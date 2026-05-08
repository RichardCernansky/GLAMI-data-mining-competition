"""
Apply trained projection model to generate multimodal embeddings locally.

Runs on CPU in seconds — the projection layers are tiny linear layers,
no encoder inference needed.

Place projection_model.pt at phase2/data/projection_model.pt before running.

Usage:
    python phase2/scripts/apply_projection.py

Output:
    phase1_proj_embeddings.npy
    train_proj_embeddings.npy

Then update configs/default.yaml:
    embedding:
      embed_tag: "proj"
    faiss:
      use_image: false   # projection already bakes in image signal

Then run:
    python phase2/scripts/train_edge_scorer.py
    python phase2/scripts/run.py --validate
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import load_config

# ── Must match train_projection.py exactly ─────────────────────────────────────

TEXT_DIM   = 768
IMG_DIM    = 512
PROJ_DIM   = 128
CONCAT_DIM = PROJ_DIM * 2 + 3   # 259
OUT_DIM    = 128


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


def apply_projection(model, text_emb, img_emb, has_img, price_norm, batch_size=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(text_emb), batch_size):
            end = min(start + batch_size, len(text_emb))
            emb = model(
                torch.tensor(text_emb[start:end]),
                torch.tensor(img_emb[start:end]),
                torch.tensor(has_img[start:end]),
                torch.tensor(price_norm[start:end]),
            )
            out.append(emb.numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def main():
    config   = load_config()
    emb_dir  = config["paths"]["embeddings"]
    proc_dir = config["paths"]["processed_data"]

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(script_dir, "data", "projection_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"projection_model.pt not found at {model_path}\n"
            f"Download it from Google Drive and place it at phase2/data/projection_model.pt"
        )

    print(f"Loading projection model...")
    model = FashionIDProjection()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print(f"  Loaded from {model_path}")

    for prefix, pkl_name in [("phase1", "phase1_prepared.pkl"), ("train", "train_prepared.pkl")]:
        pkl_path = os.path.join(proc_dir, pkl_name)
        if not os.path.exists(pkl_path):
            print(f"\nSkipping {prefix} — {pkl_name} not found")
            continue

        print(f"\nProcessing {prefix}...")
        df  = pd.read_pickle(pkl_path)
        ids = df["itemId"].values

        text_emb = np.load(os.path.join(emb_dir, f"{prefix}_ft_embeddings.npy")).astype(np.float32)
        img_emb  = np.load(os.path.join(emb_dir, f"{prefix}_img_embeddings.npy")).astype(np.float32)
        has_img  = np.load(os.path.join(emb_dir, f"{prefix}_img_has_image.npy")).astype(np.float32)

        prices     = df["price_eur"].fillna(0).values.astype(np.float32)
        log_prices = np.log1p(prices)
        price_norm = ((log_prices - log_prices.mean()) / (log_prices.std() + 1e-8)).astype(np.float32)

        print(f"  {len(ids):,} items, applying projection...")
        proj_emb = apply_projection(model, text_emb, img_emb, has_img, price_norm)
        print(f"  Output shape: {proj_emb.shape}")

        out_path = os.path.join(emb_dir, f"{prefix}_proj_embeddings.npy")
        np.save(out_path, proj_emb)
        print(f"  Saved {out_path}")

    print("\nDone. Now update configs/default.yaml:")
    print("  embedding:")
    print('    embed_tag: "proj"')
    print("  faiss:")
    print("    use_image: false")
    print("\nThen run:")
    print("  python phase2/scripts/train_edge_scorer.py")
    print("  python phase2/scripts/run.py --validate")


if __name__ == "__main__":
    main()
