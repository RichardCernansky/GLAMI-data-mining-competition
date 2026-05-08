"""
Joint training of text encoder + multimodal projection (Zalando-style).

Text encoder (SentenceTransformer) and projection layers train together.
Image encoder stays frozen — pre-computed image embeddings are used directly.

Architecture:
    text → SentenceTransformer (trainable) → 768
    img  → pre-computed CLIP embeddings   → 512 (frozen)

    text_proj (768→128) → ReLU  ┐
    img_proj  (512→128) → ReLU  ├→ concat (259) → Linear(259→128) → L2-norm
    numeric   (3)               ┘

Loss: InfoNCE with batch sampling that maximizes positive pairs.

Setup on Colab (run in a cell before this script):
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install sentence-transformers torch numpy pandas tqdm

Files needed in MyDrive/adm/phase2/:
    train_prepared.parquet         (convert locally: train_prepared.pkl -> parquet)
    finetuned-product-encoder/     (folder — starting checkpoint)
    train_img_embeddings.npy
    train_img_has_image.npy
    train_ids.npy

Usage:
    python train_joint.py

Output in MyDrive/adm/phase2/:
    jointly_trained_encoder/       (updated SentenceTransformer — download this)
    joint_projection_model.pt      (projection weights — download this)

Locally after downloading:
    Update configs/default.yaml:
        embedding:
          model_name: "data/jointly_trained_encoder"
          embed_tag: "joint"
    python phase2/scripts/extract_embeddings.py --text-only
    python phase2/scripts/apply_projection.py   (set MODEL_PATH to joint_projection_model.pt)
    python phase2/scripts/train_edge_scorer.py
    python phase2/scripts/run.py --validate
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.utils import load_config

# ── Config ─────────────────────────────────────────────────────────────────────

_config   = load_config()
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR   = _config["paths"]["embeddings"]
PROC_DIR  = _config["paths"]["processed_data"]
DATA_DIR  = os.path.join(_base_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_MODEL_DIR = os.path.join(_base_dir, _config["embedding"]["model_name"])
IMG_DIM        = 512
TEXT_DIM       = 768
PROJ_DIM       = 128
CONCAT_DIM     = PROJ_DIM * 2 + 3   # 259
OUT_DIM        = 128

ITEMS_PER_GRP  = 2      # items per product group per batch
BATCH_GROUPS   = 64     # groups per batch → ~128 items (encoder is memory-heavy)
ENCODER_BATCH  = 256    # sentence-transformer encoding batch size
EPOCHS         = 3
LR_ENCODER     = 2e-5   # small LR for encoder — don't destroy pretrained weights
LR_PROJ        = 3e-4   # larger LR for projection layers
TEMPERATURE    = 0.07
WARMUP_RATIO   = 0.1
SEED           = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load metadata and image embeddings ─────────────────────────────────────────

print("Loading metadata and image embeddings...")
train_df    = pd.read_pickle(os.path.join(PROC_DIR, "train_prepared.pkl"))
train_ids   = np.load(os.path.join(EMB_DIR, "train_ids.npy"))
img_emb     = np.load(os.path.join(EMB_DIR, "train_img_embeddings.npy")).astype(np.float32)
has_img     = np.load(os.path.join(EMB_DIR, "train_img_has_image.npy")).astype(np.float32)

id_to_label = dict(zip(train_df["itemId"].values, train_df["label"].values))
id_to_price = dict(zip(train_df["itemId"].values, train_df["price_eur"].fillna(0).values))
id_to_title = dict(zip(train_df["itemId"].values, train_df["title"].fillna("").values))
id_to_desc  = dict(zip(train_df["itemId"].values, train_df["description"].fillna("").values))

labels  = np.array([id_to_label.get(i, -1) for i in train_ids])
prices  = np.array([id_to_price.get(i, 0.0) for i in train_ids], dtype=np.float32)

def build_text(item_id, max_desc=500):
    title = str(id_to_title.get(item_id, ""))
    desc  = str(id_to_desc.get(item_id, ""))[:max_desc]
    return f"{title}. {desc}" if desc else title

# Normalize price
log_prices = np.log1p(prices)
price_norm = ((log_prices - log_prices.mean()) / (log_prices.std() + 1e-8)).astype(np.float32)

# Filter to labeled items only
mask       = labels >= 0
train_ids  = train_ids[mask]
img_emb    = img_emb[mask]
has_img    = has_img[mask]
labels     = labels[mask]
price_norm = price_norm[mask]
print(f"  {len(labels):,} items, {len(set(labels)):,} product groups")

# Build group index
groups = {}
for i, lbl in enumerate(labels):
    groups.setdefault(lbl, []).append(i)
group_keys = [k for k, v in groups.items() if len(v) >= 2]
print(f"  {len(group_keys):,} groups with ≥2 items")

# Move image embeddings to GPU tensors once
img_t   = torch.tensor(img_emb).to(device)
himg_t  = torch.tensor(has_img).to(device)
price_t = torch.tensor(price_norm).to(device)

# ── Model ──────────────────────────────────────────────────────────────────────

print(f"\nLoading text encoder from {BASE_MODEL_DIR}...")
st_model  = SentenceTransformer(BASE_MODEL_DIR)
encoder   = st_model._first_module().auto_model.to(device)
tokenizer = st_model.tokenizer
encoder.gradient_checkpointing_enable()  # recompute activations during backward to save memory

class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(TEXT_DIM, PROJ_DIM)
        self.img_proj  = nn.Linear(IMG_DIM,  PROJ_DIM)
        self.fusion    = nn.Linear(CONCAT_DIM, OUT_DIM)

    def forward(self, text_emb, img, has_img, price):
        t       = F.relu(self.text_proj(text_emb))
        i       = F.relu(self.img_proj(img)) * has_img.unsqueeze(1)
        numeric = torch.stack([price, has_img, torch.zeros_like(price)], dim=1)
        x       = torch.cat([t, i, numeric], dim=1)
        return F.normalize(self.fusion(x), dim=1)

proj = ProjectionHead().to(device)

# Separate LRs: small for encoder, larger for projection
optimizer = torch.optim.AdamW([
    {"params": encoder.parameters(),  "lr": LR_ENCODER},
    {"params": proj.parameters(),     "lr": LR_PROJ},
])
steps_per_epoch = len(group_keys) // BATCH_GROUPS
total_steps     = steps_per_epoch * EPOCHS
warmup_steps    = math.ceil(total_steps * WARMUP_RATIO)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[LR_ENCODER, LR_PROJ],
    total_steps=total_steps, pct_start=WARMUP_RATIO,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_texts(item_ids):
    """Encode a list of item_ids through the trainable text encoder."""
    texts  = [build_text(i) for i in item_ids]
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out    = encoder(**inputs)
    # mean pooling
    mask_expanded = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
    return F.normalize(emb, dim=1)


def sample_batch():
    chosen = np.random.choice(len(group_keys), size=BATCH_GROUPS, replace=False)
    idxs, lbls, item_ids = [], [], []
    for gi in chosen:
        gkey  = group_keys[gi]
        items = groups[gkey]
        sel   = np.random.choice(items, size=min(ITEMS_PER_GRP, len(items)), replace=False)
        for s in sel:
            idxs.append(s)
            lbls.append(gkey)
            item_ids.append(train_ids[s])
    return idxs, lbls, item_ids


def infonce_loss(emb, labels):
    n        = emb.size(0)
    eye      = torch.eye(n, device=device)
    sim      = (emb @ emb.T) / TEMPERATURE
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * (1 - eye)
    sim_max  = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim  = torch.exp(sim - sim_max) * (1 - eye)
    numer    = (exp_sim * pos_mask).sum(dim=1)
    denom    = exp_sim.sum(dim=1)
    has_pos  = pos_mask.sum(dim=1) > 0
    loss     = -torch.log(numer[has_pos] / (denom[has_pos] + 1e-8))
    return loss.mean()

# ── Training ───────────────────────────────────────────────────────────────────

print(f"\nJoint training {EPOCHS} epochs  (~{steps_per_epoch} steps/epoch)...")
print(f"  Encoder LR={LR_ENCODER}  Projection LR={LR_PROJ}")

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    proj.train()
    total_loss, n = 0.0, 0

    for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{EPOCHS}"):
        idxs, lbls, item_ids = sample_batch()
        idxs_t = torch.tensor(idxs)
        lbls_t = torch.tensor(lbls, device=device)

        text_emb = encode_texts(item_ids)
        emb = proj(text_emb, img_t[idxs_t], himg_t[idxs_t], price_t[idxs_t])
        loss = infonce_loss(emb, lbls_t)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n += 1

    avg_loss = total_loss / max(n, 1)
    print(f"  Epoch {epoch}  loss={avg_loss:.4f}")

    # Save checkpoint after every epoch
    ckpt_dir  = os.path.join(DATA_DIR, f"joint_checkpoint_epoch{epoch}")
    ckpt_proj = os.path.join(DATA_DIR, f"joint_projection_epoch{epoch}.pt")
    st_model._first_module().auto_model = encoder.cpu()
    st_model.save(ckpt_dir)
    encoder.to(device)   # move back to GPU for next epoch
    torch.save({
        "epoch":      epoch,
        "loss":       avg_loss,
        "proj":       proj.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
    }, ckpt_proj)
    print(f"  Checkpoint saved: {ckpt_dir}/  +  {ckpt_proj}")

# ── Final save ─────────────────────────────────────────────────────────────────

encoder_out = os.path.join(DATA_DIR, "jointly_trained_encoder")
proj_out    = os.path.join(DATA_DIR, "joint_projection_model.pt")

st_model._first_module().auto_model = encoder.cpu()
st_model.save(encoder_out)
torch.save(proj.state_dict(), proj_out)

print(f"\nSaved encoder to:    {encoder_out}/")
print(f"Saved projection to: {proj_out}")
print("\nDownload both, then locally:")
print("  1. Update configs/default.yaml:")
print('       embedding:')
print('         model_name: "data/jointly_trained_encoder"')
print('         embed_tag: "joint"')
print("  2. python phase2/scripts/extract_embeddings.py --text-only")
print("  3. In apply_projection.py set MODEL_PATH = 'data/joint_projection_model.pt'")
print("  4. python phase2/scripts/apply_projection.py")
print("  5. python phase2/scripts/train_edge_scorer.py")
print("  6. python phase2/scripts/run.py --validate")
