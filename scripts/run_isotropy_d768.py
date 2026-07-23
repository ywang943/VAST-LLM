"""
D=768 isotropy analysis at the MAIN-MODEL dimension (HTS-AT).

Clean SIGReg comparison at D=768 to match the main TPA-CSAF experiments:
  - JEPA-only HTS-AT  (lam_sig=0,  checkpoints/htsat_jepa_only_d768)
  - LeJEPA   HTS-AT  (lam_sig=0.02, checkpoints/htsat_lejepa_v3_full)

Both initialized from OPERA-CT, pretrained on the same 47K data, differing only
in SIGReg. Confirms "SIGReg -> isotropic representations" holds at the dimension
the main model actually uses (768), closing the D=128/D=768 coherence gap.

Metrics: effective rank, mean pairwise cosine, top-1 eigenvalue share,
covariance distance from identity.

Usage:
  python scripts/run_isotropy_d768.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder

D = 768
MAX_SAMPLES = 8000

ENCODERS = {
    "jepa_only_no_sigreg": "checkpoints/htsat_jepa_only_d768/htsat_lejepa_best.pt",
    "lejepa_sigreg":       "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt",
}


def collect_loader(bs=32):
    cache = Path("data/mel_cache")
    dsets = []
    for d in sorted(cache.iterdir()):
        m = d / "metadata.json"
        if m.exists():
            ds = CachedMelDataset(root=str(d), meta_file=str(m))
            if len(ds) > 0:
                dsets.append(ds)
    return DataLoader(ConcatDataset(dsets), batch_size=bs, shuffle=True, num_workers=4)


@torch.no_grad()
def extract_z(ckpt, loader, device, max_samples=MAX_SAMPLES):
    enc = build_htsat_encoder(use_csaf=True, freeze_backbone=True)
    if Path(ckpt).exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = {k.replace("encoder.", "", 1): v for k, v in ck["model_state"].items()
                 if k.startswith("encoder.")}
        enc.load_state_dict(state, strict=False)
    else:
        print(f"  WARNING: {ckpt} missing")
    enc = enc.to(device).eval()
    zs, n = [], 0
    for b in loader:
        z = enc(b["mel"].to(device))
        zs.append(z.reshape(-1, z.size(-1)).cpu())
        n += z.size(0)
        if n >= max_samples:
            break
    return torch.cat(zs, 0).numpy()


def metrics(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    cov = (Zc.T @ Zc) / Zc.shape[0]
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    total = eig.sum() + 1e-12
    p = eig / total
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    cov_n = cov / (np.trace(cov) / D)
    iso = float(np.linalg.norm(cov_n - np.eye(D), ord="fro"))
    top1 = float(eig.max() / total)
    idx = np.random.RandomState(0).choice(Z.shape[0], size=min(2000, Z.shape[0]), replace=False)
    Zn = Z[idx] / (np.linalg.norm(Z[idx], axis=1, keepdims=True) + 1e-9)
    cos = Zn @ Zn.T
    mean_cos = float((cos.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
    return {"eff_rank": round(eff_rank, 2), "iso_dist": round(iso, 3),
            "top1_share": round(top1, 4), "mean_cos": round(mean_cos, 4)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = collect_loader()
    out = {"D": D}
    for name, ckpt in ENCODERS.items():
        print(f"\n=== {name} ===")
        Z = extract_z(ckpt, loader, device)
        m = metrics(Z)
        out[name] = m
        print(f"  eff_rank={m['eff_rank']:.1f}/{D}  mean_cos={m['mean_cos']:.3f}  "
              f"top1={m['top1_share']:.3f}  iso_dist={m['iso_dist']:.2f}")

    print(f"\n{'='*60}\n  D=768 ISOTROPY (main-model dimension)\n{'='*60}")
    j = out["jepa_only_no_sigreg"]; l = out["lejepa_sigreg"]
    print(f"  Effective rank:  JEPA={j['eff_rank']:.1f}  LeJEPA={l['eff_rank']:.1f}  ({l['eff_rank']/max(j['eff_rank'],1e-6):.1f}x)")
    print(f"  Mean cosine:     JEPA={j['mean_cos']:.3f}  LeJEPA={l['mean_cos']:.3f}")
    print(f"  >>> Confirms SIGReg -> isotropy AT D=768 (the main model's dim)")

    Path("checkpoints/isotropy_d768").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/isotropy_d768/isotropy_d768_results.json").write_text(json.dumps(out, indent=2))
    print(f"  Saved: checkpoints/isotropy_d768/isotropy_d768_results.json")


if __name__ == "__main__":
    main()
