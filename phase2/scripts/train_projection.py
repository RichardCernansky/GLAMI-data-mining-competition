"""
Train a multimodal projection layer (Zalando fashionID approach) on Colab T4.

Architecture (late fusion, encoders frozen):
    text (768) → Linear(768→128) → ReLU  ┐
    img  (512) → Linear(512→128) → ReLU  ├→ concat (259) → Linear(259→128) → L2-norm
    numeric(3) → [price, has_img, 0]     ┘

Loss: InfoNCE — same-product pairs are positives, all others in batch are negatives.
Batch sampling maximizes positive pairs per batch (ITEMS_PER_GRP per product).

Setup on Colab (run in a cell before this script):
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install torch numpy pandas tqdm

Upload to MyDrive/adm/phase2/:
    train_ft_embeddings.npy
    train_img_embeddings.npy
    train_img_has_image.npy
    train_ids.npy
    train_prepared.pkl

Usage:
    python train_projection.py

Output:
    MyDrive/adm/phase2/projection_model.pt

Then locally run:
    python phase2/scripts/apply_projection.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.utils import load_config

# ── Config ─────────────────────────────────────────────────────────────────────

_config  = load_config()
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR  = _config["paths"]["embeddings"]
PROC_DIR = _config["paths"]["processed_data"]
DATA_DIR = os.path.join(_base_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
TEXT_DIM      = 768
IMG_DIM       = 512
PROJ_DIM      = 128
NUMERIC_DIM   = 3
CONCAT_DIM    = PROJ_DIM * 2 + NUMERIC_DIM  # 259
OUT_DIM       = 128
ITEMS_PER_GRP = 4      # items sampled per product group per batch
BATCH_GROUPS  = 256    # groups per batch → ~1024 items, many positive pairs
EPOCHS        = 5
LR            = 3e-4
TEMPERATURE   = 0.07
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading embeddings and metadata...")
embed_tag = _config.get("embedding", {}).get("embed_tag") or "ft"
text_emb  = np.load(os.path.join(EMB_DIR, f"train_{embed_tag}_embeddings.npy")).astype(np.float32)
img_emb   = np.load(os.path.join(EMB_DIR, "train_img_embeddings.npy")).astype(np.float32)
has_img   = np.load(os.path.join(EMB_DIR, "train_img_has_image.npy")).astype(np.float32)
train_ids = np.load(os.path.join(EMB_DIR, "train_ids.npy"))

train_df = pd.read_pickle(os.path.join(PROC_DIR, "train_prepared.pkl"))
id_to_label = dict(zip(train_df["itemId"].values, train_df["label"].values))
id_to_price = dict(zip(train_df["itemId"].values, train_df["price_eur"].fillna(0).values))
labels      = np.array([id_to_label.get(i, -1) for i in train_ids])
prices      = np.array([id_to_price.get(i, 0.0) for i in train_ids], dtype=np.float32)

log_prices  = np.log1p(prices)
price_norm  = ((log_prices - log_prices.mean()) / (log_prices.std() + 1e-8)).astype(np.float32)

mask       = labels >= 0
text_emb   = text_emb[mask]
img_emb    = img_emb[mask]
has_img    = has_img[mask]
labels     = labels[mask]
price_norm = price_norm[mask]
print(f"  {len(labels):,} items, {len(set(labels)):,} product groups")

# ── Dataset ────────────────────────────────────────────────────────────────────

# Build group index once
groups = {}
for i, lbl in enumerate(labels):
    groups.setdefault(lbl, []).append(i)
group_keys = [k for k, v in groups.items() if len(v) >= 2]
print(f"  {len(group_keys):,} groups with ≥2 items")

text_t  = torch.tensor(text_emb)
img_t   = torch.tensor(img_emb)
himg_t  = torch.tensor(has_img)
price_t = torch.tensor(price_norm)


def sample_batch():
    """Sample BATCH_GROUPS groups, ITEMS_PER_GRP items each."""
    chosen_groups = np.random.choice(len(group_keys), size=BATCH_GROUPS, replace=False)
    idxs, lbls = [], []
    for gi in chosen_groups:
        gkey  = group_keys[gi]
        items = groups[gkey]
        sel   = np.random.choice(items, size=min(ITEMS_PER_GRP, len(items)), replace=False)
        idxs.extend(sel)
        lbls.extend([gkey] * len(sel))
    idxs = torch.tensor(idxs)
    return (
        text_t[idxs].to(device),
        img_t[idxs].to(device),
        himg_t[idxs].to(device),
        price_t[idxs].to(device),
        torch.tensor(lbls, device=device),
    )

# ── Model ──────────────────────────────────────────────────────────────────────

class FashionIDProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(TEXT_DIM, PROJ_DIM)
        self.img_proj  = nn.Linear(IMG_DIM,  PROJ_DIM)
        self.fusion    = nn.Linear(CONCAT_DIM, OUT_DIM)

    def forward(self, text, img, has_img, price):
        t       = F.relu(self.text_proj(text))
        i       = F.relu(self.img_proj(img)) * has_img.unsqueeze(1)
        numeric = torch.stack([price, has_img, torch.zeros_like(price)], dim=1)
        x       = torch.cat([t, i, numeric], dim=1)
        return F.normalize(self.fusion(x), dim=1)


model     = FashionIDProjection().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── InfoNCE loss ───────────────────────────────────────────────────────────────

def infonce_loss(emb, labels):
    n        = emb.size(0)
    eye      = torch.eye(n, device=emb.device)
    sim      = (emb @ emb.T) / TEMPERATURE
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * (1 - eye)

    sim_max  = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim  = torch.exp(sim - sim_max) * (1 - eye)  # no in-place ops

    numer    = (exp_sim * pos_mask).sum(dim=1)
    denom    = exp_sim.sum(dim=1)
    has_pos  = pos_mask.sum(dim=1) > 0
    loss     = -torch.log(numer[has_pos] / (denom[has_pos] + 1e-8))
    return loss.mean()

# ── Training ───────────────────────────────────────────────────────────────────

steps_per_epoch = len(group_keys) // BATCH_GROUPS
print(f"\nTraining {EPOCHS} epochs  (~{steps_per_epoch} steps/epoch)...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n = 0.0, 0

    for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{EPOCHS}"):
        text, img, has_img_b, price, lbls = sample_batch()
        emb  = model(text, img, has_img_b, price)
        loss = infonce_loss(emb, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1

    scheduler.step()
    print(f"  Epoch {epoch}  loss={total_loss/max(n,1):.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

# ── Save ───────────────────────────────────────────────────────────────────────

out_path = os.path.join(DATA_DIR, "sep_projection_model.pt")
torch.save(model.state_dict(), out_path)
print(f"\nSaved to {out_path}")
print("Download projection_model.pt then run:")
print("  python phase2/scripts/apply_projection.py")
