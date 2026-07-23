"""
D=768 VQ ablation with a STRONG encoder + large VQ-training data.

Goal: confirm whether EMA / L2 help when the encoder is strong (HTS-AT LeJEPA
pretrained on 47K) and VQ has lots of data — under rigorous FULL-DATASET
utilization (not per-batch best).

Encoders compared:
  - LeJEPA HTS-AT  (SIGReg-on, from htsat_lejepa_v3_full)  → main
  - OPERA-CT raw   (COLA, no SIGReg)                        → no-SIGReg anchor

VQ conditions on each encoder:
  - B: no EMA, no L2  (plain gradient VQ)
  - C: + EMA
  - D: + EMA + L2  (ours)

Metric: full-dataset codebook utilization + perplexity (all tokens).
Codebook sizes: 512, 1024, 2048.

Usage:
  python scripts/run_vq_d768_ablation.py
"""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.vq import VectorQuantizer

D = 768
OUT_DIR = "checkpoints/vq_d768_ablation"
LEJEPA_CKPT = "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt"


def collect_loader(batch_size=32, max_per_set=None):
    cache_dir = Path("data/mel_cache")
    dsets = []
    for d in sorted(cache_dir.iterdir()):
        meta = d / "metadata.json"
        if meta.exists():
            ds = CachedMelDataset(root=str(d), meta_file=str(meta))
            if len(ds) > 0:
                dsets.append(ds)
    combined = ConcatDataset(dsets)
    return DataLoader(combined, batch_size=batch_size, shuffle=False, num_workers=4)


@torch.no_grad()
def extract_features(encoder, loader, device, max_samples=20000):
    """Run frozen encoder over data, return cached z_cont (N, L, D) on CPU."""
    encoder.eval()
    feats = []
    n = 0
    for batch in loader:
        mel = batch["mel"].to(device)
        z = encoder(mel)                # (B, 64, D)
        feats.append(z.cpu())
        n += z.size(0)
        if n >= max_samples:
            break
    return torch.cat(feats, dim=0)      # (N, 64, D)


@torch.no_grad()
def vq_full_util(vq, feats, device, K):
    """Compute full-dataset utilization after training."""
    vq.eval()
    counts = torch.zeros(K, device=device)
    for i in range(0, feats.size(0), 256):
        z = feats[i:i+256].to(device)
        out = vq(z)
        counts += torch.bincount(out["ids"].reshape(-1), minlength=K).float()
    util = (counts > 0).sum().item() / K
    probs = counts / (counts.sum() + 1e-10)
    perp = (-(probs * (probs + 1e-10).log()).sum()).exp().item()
    return util, perp


def train_vq(feats, use_ema, l2, K, device, epochs=15):
    """Train a VQ on cached features. Returns full-dataset util/perp."""
    vq = VectorQuantizer(codebook_size=K, D=D, beta=0.25, use_ema=use_ema,
                         ema_decay=0.99, restart_threshold=1, restart_every=1,
                         l2_normalize=l2).to(device)
    opt = torch.optim.AdamW(vq.parameters(), lr=1e-3) if not use_ema else None
    N = feats.size(0)
    idx = torch.randperm(N)
    for ep in range(epochs):
        vq.train()
        for i in range(0, N, 256):
            z = feats[idx[i:i+256]].to(device)
            out = vq(z)
            if opt is not None:
                opt.zero_grad(); out["loss"].backward(); opt.step()
    return vq_full_util(vq, feats, device, K)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codebooks", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--vq-epochs", type=int, default=15)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    loader = collect_loader()

    # --- Encoder 1: LeJEPA HTS-AT (SIGReg-on) ---
    print("Building LeJEPA HTS-AT encoder...")
    enc_lejepa = build_htsat_encoder(use_csaf=True, freeze_backbone=True)
    if Path(LEJEPA_CKPT).exists():
        ck = torch.load(LEJEPA_CKPT, map_location="cpu", weights_only=False)
        state = {k.replace("encoder.", "", 1): v for k, v in ck["model_state"].items()
                 if k.startswith("encoder.")}
        miss, unexp = enc_lejepa.load_state_dict(state, strict=False)
        print(f"  loaded LeJEPA weights ({len(state)-len(miss)}/{len(state)})")
    enc_lejepa = enc_lejepa.to(device)

    # --- Encoder 2: OPERA-CT raw (COLA, no SIGReg) ---
    print("Building OPERA-CT raw encoder (no SIGReg anchor)...")
    enc_opera = build_htsat_encoder(use_csaf=True, freeze_backbone=True).to(device)
    # (CSAF random-init but HTS-AT = OPERA-CT COLA weights; no LeJEPA/SIGReg applied)

    encoders = {"lejepa_sigreg": enc_lejepa, "operaCT_no_sigreg": enc_opera}

    results = {}
    for enc_name, enc in encoders.items():
        print(f"\n=== Encoder: {enc_name} — extracting features ===")
        feats = extract_features(enc, loader, device, max_samples=args.max_samples)
        print(f"  features: {tuple(feats.shape)}")
        results[enc_name] = {}
        for K in args.codebooks:
            row = {}
            for cond, (ema, l2) in [("B_no_ema_no_l2", (False, False)),
                                     ("C_ema", (True, False)),
                                     ("D_ema_l2", (True, True))]:
                util, perp = train_vq(feats, ema, l2, K, device, epochs=args.vq_epochs)
                row[cond] = {"util": round(util, 4), "perp": round(perp, 1)}
                print(f"  K={K:5d} {cond:16s}: util={util*100:5.1f}%  perp={perp:.0f}")
            results[enc_name][f"K{K}"] = row

    out = {
        "experiment": "D=768 VQ ablation, strong encoder + 20K VQ data, full-dataset util",
        "encoders": list(encoders.keys()),
        "codebooks": args.codebooks,
        "results": results,
    }
    (Path(OUT_DIR) / "vq_d768_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {OUT_DIR}/vq_d768_results.json")

    # Quick interpretation
    print("\n=== Does EMA/L2 help? (LeJEPA encoder, K=512) ===")
    r = results["lejepa_sigreg"]["K512"]
    print(f"  B (none):   {r['B_no_ema_no_l2']['util']*100:.1f}%")
    print(f"  C (+EMA):   {r['C_ema']['util']*100:.1f}%")
    print(f"  D (+EMA+L2):{r['D_ema_l2']['util']*100:.1f}%")


if __name__ == "__main__":
    main()
