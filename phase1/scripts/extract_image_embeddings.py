"""
Extract image embeddings using CLIP for all items.

Run from project root:
    python scripts/extract_image_embeddings.py

Saves embeddings as numpy arrays to data/processed/embeddings/
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config

import torch
from transformers import CLIPModel, CLIPProcessor


def main():
    config = load_config()
    proc_dir = config["paths"]["processed_data"]
    emb_dir = config["paths"]["embeddings"]
    img_dir = config["paths"].get("images", "data/raw/fit_dataset_images")
    os.makedirs(emb_dir, exist_ok=True)

    # --- Load prepared data (just need itemIds) ---
    print("Loading prepared data...")
    train = pd.read_pickle(os.path.join(proc_dir, "train_prepared.pkl"))
    phase1 = pd.read_pickle(os.path.join(proc_dir, "phase1_prepared.pkl"))

    train_ids = train["itemId"].values
    phase1_ids = phase1["itemId"].values

    print(f"  Train: {len(train_ids)} items")
    print(f"  Phase1: {len(phase1_ids)} items")

    # --- Check how many images exist ---
    available_images = set(
        int(f.replace(".jpg", "")) for f in os.listdir(img_dir) if f.endswith(".jpg")
    )
    train_has_img = sum(1 for i in train_ids if i in available_images)
    phase1_has_img = sum(1 for i in phase1_ids if i in available_images)
    print(f"\n  Images available: {len(available_images)}")
    print(f"  Train items with images: {train_has_img}/{len(train_ids)} ({train_has_img/len(train_ids)*100:.1f}%)")
    print(f"  Phase1 items with images: {phase1_has_img}/{len(phase1_ids)} ({phase1_has_img/len(phase1_ids)*100:.1f}%)")

    # --- Load CLIP model ---
    model_name = "openai/clip-vit-base-patch32"
    print(f"\nLoading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Use MPS if available (M1 GPU)
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using M1 GPU (MPS)")
    model = model.to(device)
    model.eval()

    # --- Encode function ---
    def encode_images(item_ids, batch_size=64):
        """Encode a list of item images. Returns embeddings array and valid mask."""
        n = len(item_ids)
        embed_dim = 512  # CLIP ViT-B/32 output dimension
        embeddings = np.zeros((n, embed_dim), dtype=np.float32)
        has_image = np.zeros(n, dtype=bool)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = item_ids[start:end]

            # Load images that exist
            images = []
            indices = []
            for i, item_id in enumerate(batch_ids):
                img_path = os.path.join(img_dir, f"{item_id}.jpg")
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                        indices.append(start + i)
                    except Exception as e:
                        pass  # Skip corrupted images

            if images:
                inputs = processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                with torch.no_grad():
                    vision_outputs = model.vision_model(pixel_values=pixel_values)
                    outputs = model.visual_projection(vision_outputs.pooler_output)
                    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                    outputs = outputs.cpu().numpy()

                for j, idx in enumerate(indices):
                    embeddings[idx] = outputs[j]
                    has_image[idx] = True

            if (start + batch_size) % (batch_size * 50) == 0 or end == n:
                pct = end / n * 100
                print(f"  {end}/{n} ({pct:.1f}%)")

        return embeddings, has_image

    # --- Encode train ---
    print(f"\nEncoding {len(train_ids)} train images...")
    t0 = time.time()
    train_img_emb, train_has_img = encode_images(train_ids)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Images found: {train_has_img.sum()}/{len(train_ids)}")

    # --- Encode phase1 ---
    print(f"\nEncoding {len(phase1_ids)} phase1 images...")
    t0 = time.time()
    phase1_img_emb, phase1_has_img = encode_images(phase1_ids)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Images found: {phase1_has_img.sum()}/{len(phase1_ids)}")

    # --- Save ---
    np.save(os.path.join(emb_dir, "train_img_embeddings.npy"), train_img_emb)
    np.save(os.path.join(emb_dir, "train_img_has_image.npy"), train_has_img)
    np.save(os.path.join(emb_dir, "phase1_img_embeddings.npy"), phase1_img_emb)
    np.save(os.path.join(emb_dir, "phase1_img_has_image.npy"), phase1_has_img)

    print(f"\nSaved to {emb_dir}/")
    for f in sorted(os.listdir(emb_dir)):
        size_mb = os.path.getsize(os.path.join(emb_dir, f)) / 1024 / 1024
        print(f"  {f:45s} {size_mb:8.1f} MB")

    # --- Sanity check ---
    print("\n--- Sanity check ---")
    label_counts = train["label"].value_counts()
    test_label = label_counts.index[0]
    idxs = np.where((train["label"] == test_label).values)[0]

    # Find two items with same label that both have images
    pair_found = False
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            if train_has_img[idxs[i]] and train_has_img[idxs[j]]:
                a, b = idxs[i], idxs[j]
                sim = np.dot(train_img_emb[a], train_img_emb[b])
                print(f"Same product (label {test_label}):")
                print(f"  Item 1: {train_ids[a]} ({train.iloc[a]['title'][:60]})")
                print(f"  Item 2: {train_ids[b]} ({train.iloc[b]['title'][:60]})")
                print(f"  Image cosine sim: {sim:.4f}")
                pair_found = True
                break
        if pair_found:
            break

    # Random pair
    rng = np.random.RandomState(42)
    valid_indices = np.where(train_has_img)[0]
    if len(valid_indices) >= 2:
        ri, rj = rng.choice(valid_indices, size=2, replace=False)
        sim_rand = np.dot(train_img_emb[ri], train_img_emb[rj])
        print(f"\nRandom pair:")
        print(f"  Item 1: {train_ids[ri]} ({train.iloc[ri]['title'][:60]})")
        print(f"  Item 2: {train_ids[rj]} ({train.iloc[rj]['title'][:60]})")
        print(f"  Image cosine sim: {sim_rand:.4f}")


if __name__ == "__main__":
    main()