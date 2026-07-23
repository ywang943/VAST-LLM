"""
Fix ablation condition E: use the exact same model as run_csaf_frozen_htsat.py
(CSAFLinearProbe with trainable pool1/pool2 + CSAF + head).

This ensures the ablation E result matches the main TPA-CSAF experiment.
"""

import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
from scripts.run_csaf_frozen_htsat import CSAFLinearProbe, run_seed
from data.respvoice_datasets import CachedMelDataset
from scripts.run_opera_icbhi_disease import official_split

SEEDS = [0, 1, 2, 3, 4]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "./checkpoints/csaf_ablation_e_fixed"
    print("=== Ablation E (FIXED): Same model as main TPA-CSAF experiment ===")

    results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        auc = run_seed(seed, device, out_dir)
        print(f"  seed {seed}: test AUROC = {auc:.4f}")
        results.append(auc)

    m_auc = float(np.mean(results))
    s_auc = float(np.std(results))
    print(f"\n{'='*55}")
    print(f"  Ablation E (FIXED): {m_auc:.3f} +- {s_auc:.3f}")
    print(f"  per-seed: {[round(a,4) for a in results]}")
    print(f"{'='*55}")

    # Update the ablation results file
    ablation_path = Path("checkpoints/csaf_ablation/csaf_ablation_results.json")
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text())
    else:
        ablation = {}

    ablation["E_csaf"] = {
        "description": "TPA-CSAF (FIXED: same model as main experiment, trainable pool1/pool2)",
        "auroc_mean": round(m_auc, 4),
        "auroc_std": round(s_auc, 4),
        "per_seed": [round(a, 4) for a in results],
    }
    ablation_path.write_text(json.dumps(ablation, indent=2))
    print(f"  Updated: {ablation_path}")


if __name__ == "__main__":
    main()
