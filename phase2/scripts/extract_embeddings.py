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


def main(text_only=False):
    config = load_config()
    proc_dir = config["paths"]["processed_data"]
    emb_dir = config["paths"]["embeddings"]
    img_dir = config["paths"]["images"]
    emb_cfg = config["embedding"]
    os.makedirs(emb_dir, exist_ok=True)

    embed_tag  = emb_cfg.get("embed_tag")
    tag_suffix = f"_{embed_tag}" if embed_tag else ""
    print(f"  embed_tag={embed_tag!r}  → text files will be {{prefix}}{tag_suffix}_embeddings.npy")

    for prefix, pkl_name in [("phase1", "phase1_prepared.pkl"), ("phase2", "phase2_prepared.pkl"), ("train", "train_prepared.pkl")]:
        pkl_path = os.path.join(proc_dir, pkl_name)
        if not os.path.exists(pkl_path):
            print(f"\nSkipping {prefix} — {pkl_name} not found")
            continue

        print(f"\nLoading {pkl_name}...")
        df  = pd.read_pickle(pkl_path)
        ids = df["itemId"].values
        print(f"  {len(ids)} items")

        print(f"Encoding text with {emb_cfg['model_name']}...")
        texts    = df.apply(lambda r: build_text(r, emb_cfg["max_description_length"]), axis=1).tolist()
        st_model = SentenceTransformer(emb_cfg["model_name"])
        t0       = time.time()
        text_emb = st_model.encode(texts, batch_size=emb_cfg["batch_size"],
                                   show_progress_bar=True, normalize_embeddings=True)
        print(f"  Done in {time.time()-t0:.1f}s, shape: {text_emb.shape}")

        emb_file = f"{prefix}{tag_suffix}_embeddings.npy"
        np.save(os.path.join(emb_dir, emb_file), text_emb.astype(np.float32))
        np.save(os.path.join(emb_dir, f"{prefix}_ids.npy"), ids)
        print(f"  Saved {emb_file}")

        # Image embeddings don't depend on text model — skip if already exist or --text-only
        img_out = os.path.join(emb_dir, f"{prefix}_img_embeddings.npy")
        if text_only:
            print("  Skipping image encoding (--text-only)")
        elif os.path.exists(img_out):
            print(f"  Skipping image encoding — {prefix}_img_embeddings.npy already exists")
        else:
            print("  Encoding images with CLIP...")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            clip   = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            proc   = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            t0     = time.time()
            img_emb, has_img = encode_images(ids, img_dir, clip, proc, device)
            print(f"  Done in {time.time()-t0:.1f}s, images found: {has_img.sum()}/{len(ids)}")
            np.save(img_out, img_emb)
            np.save(os.path.join(emb_dir, f"{prefix}_img_has_image.npy"), has_img)

    print(f"\nAll embeddings saved to {emb_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-only", action="store_true", help="Skip image encoding (faster, text embeddings only)")
    args = parser.parse_args()
    main(text_only=args.text_only)
