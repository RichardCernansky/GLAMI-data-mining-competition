import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils import load_config, prepare_items

def main():
    config = load_config()
    raw_dir = config["paths"]["raw_data"]
    out_dir = config["paths"]["processed_data"]
    os.makedirs(out_dir, exist_ok=True)

    print("Loading items_phase_2.csv...")
    phase2 = pd.read_csv(os.path.join(raw_dir, "items_phase_2.csv"))
    print(f"  {len(phase2)} items, columns: {list(phase2.columns)}")

    print("Preparing items...")
    phase2_prep = prepare_items(phase2, config)
    phase2_prep.to_pickle(os.path.join(out_dir, "phase2_prepared.pkl"))
    print(f"  Saved phase2_prepared.pkl ({len(phase2_prep)} items)")

if __name__ == "__main__":
    main()
