import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from src.utils import load_config


def build_text(row, max_desc):
    parts = [str(row["title"])]
    if pd.notna(row["description"]):
        parts.append(str(row["description"])[:max_desc])
    return ". ".join(parts)


def encode_images(item_ids, img_dir, model, processor, device, batch_size=64):
    n = len(item_ids)
    embeddings = np.zeros((n, 512), dtype=np.float32)
    has_image = np.zeros(n, dtype=bool)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        images, indices = [], []
        for i, item_id in enumerate(item_ids[start:end]):
            path = os.path.join(img_dir, f"{item_id}.jpg")
            if os.path.exists(path):
                try:
                    images.append(Image.open(path).convert("RGB"))
                    indices.append(start + i)
                except Exception:
                    pass
        if images:
            inputs = processor(images=images, return_tensors="pt")
            with torch.no_grad():
                out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
                out = model.visual_projection(out.pooler_output)
                out = (out / out.norm(dim=-1, keepdim=True)).cpu().numpy()
            for j, idx in enumerate(indices):
                embeddings[idx] = out[j]
                has_image[idx] = True

        if end % (batch_size * 20) == 0 or end == n:
            print(f"  images {end}/{n}")

    return embeddings, has_image


def main():
    config = load_config()
    proc_dir = config["paths"]["processed_data"]
    emb_dir = config["paths"]["embeddings"]
    img_dir = config["paths"]["images"]
    emb_cfg = config["embedding"]
    os.makedirs(emb_dir, exist_ok=True)

    print("Loading phase2_prepared.pkl...")
    phase2 = pd.read_pickle(os.path.join(proc_dir, "phase2_prepared.pkl"))
    ids = phase2["itemId"].values
    print(f"  {len(ids)} items")

    # --- Text embeddings ---
    print(f"\nEncoding text with {emb_cfg['model_name']}...")
    texts = phase2.apply(lambda r: build_text(r, emb_cfg["max_description_length"]), axis=1).tolist()
    model = SentenceTransformer(emb_cfg["model_name"])
    t0 = time.time()
    text_emb = model.encode(texts, batch_size=emb_cfg["batch_size"],
                            show_progress_bar=True, normalize_embeddings=True)
    print(f"  Done in {time.time()-t0:.1f}s, shape: {text_emb.shape}")
    np.save(os.path.join(emb_dir, "phase2_text_embeddings.npy"), text_emb.astype(np.float32))
    np.save(os.path.join(emb_dir, "phase2_ids.npy"), ids)

    # --- Image embeddings ---
    print("\nEncoding images with CLIP...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    t0 = time.time()
    img_emb, has_img = encode_images(ids, img_dir, clip, proc, device)
    print(f"  Done in {time.time()-t0:.1f}s, images found: {has_img.sum()}/{len(ids)}")
    np.save(os.path.join(emb_dir, "phase2_img_embeddings.npy"), img_emb)
    np.save(os.path.join(emb_dir, "phase2_img_has_image.npy"), has_img)
    print(f"\nSaved to {emb_dir}/")

if __name__ == "__main__":
    main()
