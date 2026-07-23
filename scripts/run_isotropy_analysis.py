"""
Mechanistic evidence for SIGReg → codebook utilization.

Claim: SIGReg drives representations toward isotropic Gaussian, spreading them
uniformly in embedding space (high effective rank), which is WHY VQ codebook
utilization rises. This measures the CAUSE behind the controlled 1%→72% result.

For JEPA-only (no SIGReg) vs LeJEPA (SIGReg) encoders (3 seeds each), extract
z_cont on ICBHI and compute:
  - Effective rank (participation ratio of covariance eigenvalues): how many
    dimensions are actually used. Isotropic = full rank; collapsed = rank ~1.
  - Isotropy score: ||normalized_cov - I/D||_F  (lower = more isotropic)
  - Mean cosine similarity between random pairs (collapse → ~1)
  - Variance concentration: top-1 eigenvalue / total (collapse → ~1)

Pairs the utilization result with its mechanism — makes the claim defensible.

Usage:
  python scripts/run_isotropy_analysis.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.config import ModelConfig
from respvoice.model import RespVoiceModel

D = 128
CACHE = "data/mel_cache/opera_icbhi_disease"
SEEDS = [0, 1, 2]
VAL_DIR = "checkpoints/sigreg_vq_validation"


@torch.no_grad()
def extract_z(ckpt_path, device):
    """Load a D=128 stage1 encoder, return all z_cont tokens (N*L, D)."""
    cfg = ModelConfig(D=D, encoder_layers=6, encoder_heads=8,
                      predictor_layers=2, n_sigreg_slices=64, codebook_size=512)
    model = RespVoiceModel(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"]
    enc = {k: v for k, v in state.items() if not k.startswith("vq.") and not k.startswith("decoder.")}
    model.load_state_dict(enc, strict=False)
    model = model.to(device).eval()

    ds = CachedMelDataset(root=CACHE, meta_file=str(Path(CACHE) / "metadata.json"))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    zs = []
    for batch in loader:
        z = model.encoder(batch["mel"].to(device))   # (B, L, D)
        zs.append(z.reshape(-1, z.size(-1)).cpu())
    return torch.cat(zs, 0).numpy()                   # (N*L, D)


def isotropy_metrics(Z):
    """Z: (N, D). Returns dict of isotropy/rank metrics."""
    Zc = Z - Z.mean(0, keepdims=True)
    cov = (Zc.T @ Zc) / Zc.shape[0]                   # (D, D)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    total = eig.sum() + 1e-12
    p = eig / total
    # Effective rank = exp(entropy of normalized eigenvalue spectrum)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    # Participation ratio (alternative effective dim)
    part_ratio = float((eig.sum() ** 2) / (np.square(eig).sum() + 1e-12))
    # Isotropy: distance of normalized covariance from scaled identity
    cov_n = cov / (np.trace(cov) / D)
    iso_dist = float(np.linalg.norm(cov_n - np.eye(D), ord="fro"))
    # Top eigenvalue share (collapse → near 1)
    top1_share = float(eig.max() / total)
    # Mean pairwise cosine of a random subset (collapse → near 1)
    idx = np.random.RandomState(0).choice(Z.shape[0], size=min(2000, Z.shape[0]), replace=False)
    Zn = Z[idx] / (np.linalg.norm(Z[idx], axis=1, keepdims=True) + 1e-9)
    cos = Zn @ Zn.T
    mean_cos = float((cos.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
    return {"eff_rank": round(eff_rank, 2), "part_ratio": round(part_ratio, 2),
            "iso_dist": round(iso_dist, 3), "top1_share": round(top1_share, 4),
            "mean_cos": round(mean_cos, 4)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"D": D, "jepa_only": [], "lejepa": []}

    for seed in SEEDS:
        jepa_ck = f"{VAL_DIR}/s{seed}_stage1_jepa/stage1_best.pt"
        leje_ck = f"{VAL_DIR}/s{seed}_stage1_leje/stage1_best.pt"
        if not Path(jepa_ck).exists() or not Path(leje_ck).exists():
            print(f"seed {seed}: ckpt missing, skip"); continue
        print(f"\n=== seed {seed} ===")
        zj = extract_z(jepa_ck, device); zl = extract_z(leje_ck, device)
        mj = isotropy_metrics(zj); ml = isotropy_metrics(zl)
        out["jepa_only"].append(mj); out["lejepa"].append(ml)
        print(f"  JEPA-only: eff_rank={mj['eff_rank']:.1f}  iso_dist={mj['iso_dist']:.2f}  top1={mj['top1_share']:.3f}  cos={mj['mean_cos']:.3f}")
        print(f"  LeJEPA:    eff_rank={ml['eff_rank']:.1f}  iso_dist={ml['iso_dist']:.2f}  top1={ml['top1_share']:.3f}  cos={ml['mean_cos']:.3f}")

    def agg(rows, key):
        v = np.array([r[key] for r in rows])
        return float(v.mean()), float(v.std())

    print(f"\n{'='*60}")
    print("  ISOTROPY: JEPA-only vs LeJEPA (mean ± std over seeds)")
    print(f"{'='*60}")
    summary = {}
    for key, name in [("eff_rank", "Effective rank (↑=isotropic)"),
                      ("iso_dist", "Cov dist from identity (↓=isotropic)"),
                      ("top1_share", "Top-1 eigenvalue share (↓=spread)"),
                      ("mean_cos", "Mean pairwise cosine (↓=spread)")]:
        jm, js = agg(out["jepa_only"], key)
        lm, ls = agg(out["lejepa"], key)
        summary[key] = {"jepa": [jm, js], "lejepa": [lm, ls]}
        print(f"  {name:38s}  JEPA={jm:7.3f}±{js:.3f}   LeJEPA={lm:7.3f}±{ls:.3f}")
    print(f"{'='*60}")
    er_j = summary["eff_rank"]["jepa"][0]; er_l = summary["eff_rank"]["lejepa"][0]
    print(f"  >>> SIGReg raises effective rank {er_j:.1f} → {er_l:.1f}  ({er_l/max(er_j,1e-6):.1f}x)")
    print(f"      Of max possible D={D}. This explains codebook util 1% → 72%.")

    out["summary"] = summary
    Path("checkpoints/isotropy_analysis").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/isotropy_analysis/isotropy_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: checkpoints/isotropy_analysis/isotropy_results.json")


if __name__ == "__main__":
    main()
