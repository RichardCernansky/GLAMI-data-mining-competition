"""
Fine-tune SentenceTransformer with contrastive loss on product pairs.

Runs on the school server (GPU) or Google Colab.

Prerequisites:
    python phase2/scripts/build_pairs.py   (generates contrastive_pairs.parquet)

Usage:
    python phase2/scripts/finetune_embeddings.py

Resuming after a crash:
    Set BASE_MODEL to the latest checkpoint folder, e.g.:
        BASE_MODEL = "data/checkpoints/500"

Output:
    phase2/data/finetuned-product-encoder/

After training, update configs/default.yaml:
    embedding:
      model_name: "data/finetuned-product-encoder"
      embed_tag: "ft"

Then run:
    python phase2/scripts/extract_embeddings.py
    python phase2/scripts/train_edge_scorer.py
    python phase2/scripts/run.py --validate
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, evaluation
from sentence_transformers.losses import MultipleNegativesRankingLoss
from src.utils import load_config

# ── Config ─────────────────────────────────────────────────────────────────────

_config   = load_config()
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = _config["paths"]["processed_data"]
DATA_DIR  = os.path.join(_base_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_MODEL   = "paraphrase-multilingual-mpnet-base-v2"
PAIRS_FILE   = os.path.join(PROC_DIR, "contrastive_pairs.parquet")
OUTPUT_DIR   = os.path.join(DATA_DIR, "finetuned-product-encoder")
CKPT_DIR     = os.path.join(DATA_DIR, "checkpoints")
BATCH_SIZE   = 64
EPOCHS       = 3
WARMUP_RATIO = 0.1

# ── Load pairs ─────────────────────────────────────────────────────────────────

print(f"Loading pairs from {PAIRS_FILE}...")
df = pd.read_parquet(PAIRS_FILE)
print(f"  {len(df):,} positive pairs")

split    = int(len(df) * 0.9)
train_df = df.iloc[:split]
eval_df  = df.iloc[split:]

train_examples = [
    InputExample(texts=[row.text_a, row.text_b])
    for row in train_df.itertuples()
]

# ── Model ──────────────────────────────────────────────────────────────────────
# cache_folder on Drive so re-runs don't re-download 1GB

print(f"Loading base model: {BASE_MODEL}")
model = SentenceTransformer(BASE_MODEL)

# ── Loss ───────────────────────────────────────────────────────────────────────

loss = MultipleNegativesRankingLoss(model)

# ── Evaluator ──────────────────────────────────────────────────────────────────

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1=eval_df["text_a"].tolist(),
    sentences2=eval_df["text_b"].tolist(),
    scores=[1.0] * len(eval_df),
    name="product-pairs",
)

# ── Train ──────────────────────────────────────────────────────────────────────

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
steps_per_epoch  = len(train_dataloader)
warmup_steps     = math.ceil(steps_per_epoch * EPOCHS * WARMUP_RATIO)
ckpt_every       = max(1, steps_per_epoch // 4)  # ~4 checkpoints per epoch

print(f"\nSteps per epoch: {steps_per_epoch}  checkpoint every: {ckpt_every} steps")
print(f"Fine-tuning for {EPOCHS} epochs  (warmup={warmup_steps} steps)...")
print(f"Output:      {OUTPUT_DIR}")
print(f"Checkpoints: {CKPT_DIR}")

model.fit(
    train_objectives=[(train_dataloader, loss)],
    evaluator=evaluator,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path=OUTPUT_DIR,
    save_best_model=True,
    evaluation_steps=steps_per_epoch,      # eval once per epoch
    checkpoint_path=CKPT_DIR,
    checkpoint_save_steps=ckpt_every,      # ~4x per epoch to Drive
    checkpoint_save_total_limit=4,
    show_progress_bar=True,
)

print(f"\nDone. Model saved to: {OUTPUT_DIR}")
print("Update configs/default.yaml:")
print('  embedding:')
print('    model_name: "data/finetuned-product-encoder"')
print('    embed_tag: "ft"')

