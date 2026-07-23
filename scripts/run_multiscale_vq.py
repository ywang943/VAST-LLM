"""
Scale-wise Multi-Codebook VQ: each HTS-AT stage gets its own codebook.

  e1 (B,64,192) → proj → VQ1 (K=512, D=128) → ids1
  e2 (B,64,384) → proj → VQ2 (K=512, D=128) → ids2
  e3 (B,64,768) → proj → VQ3 (K=512, D=128) → ids3
  e4 (B,64,768) → proj → VQ4 (K=512, D=128) → ids4

Compare with single-codebook baseline:
  e1,e2,e3,e4 → CSAF → single VQ (K=512, D=768) → ids

Measures per-codebook utilization + perplexity to test:
  "SIGReg stabilizes codebooks across ALL acoustic scales"

Uses frozen HTS-AT (OPERA-CT) features, trains VQ only.
Two encoder sources:
  - OPERA-CT raw (no SIGReg)
  - LeJEPA V3 (with SIGReg)

Usage:
  python scripts/run_multiscale_vq.py
"""

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.vq import VectorQuantizer

STAGE_DIMS = (192, 384, 768, 768)
PROJ_D = 128
K = 512
OUT_DIR = "checkpoints/multiscale_vq"
LEJEPA_CKPT = "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt"


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
def extract_4stages(encoder, loader, device, max_samples=10000):
    """Extract features from all 4 HTS-AT stages. Returns list of 4 tensors."""
    encoder.eval()
    stages = [[] for _ in range(4)]
    n = 0
    for batch in loader:
        mel = batch["mel"].to(device)
        x = encoder._preprocess(mel)
        x = encoder.htsat.patch_embed(x)
        if encoder.htsat.ape:
            x = x + encoder.htsat.absolute_pos_embed
        x = encoder.htsat.pos_drop(x)

        x, _ = encoder.htsat.layers[0](x)
        stages[0].append(encoder.pool1(x).cpu())
        x, _ = encoder.htsat.layers[1](x)
        stages[1].append(encoder.pool2(x).cpu())
        x, _ = encoder.htsat.layers[2](x)
        stages[2].append(x.cpu())
        x, _ = encoder.htsat.layers[3](x)
        stages[3].append(encoder.htsat.norm(x).cpu())

        n += mel.size(0)
        if n >= max_samples:
            break
    return [torch.cat(s, 0) for s in stages]


@torch.no_grad()
def full_util(vq, feats, proj, device, K_size):
    vq.eval(); proj.eval()
    counts = torch.zeros(K_size, device=device)
    for i in range(0, feats.size(0), 128):
        z = proj(feats[i:i+128].to(device))
        out = vq(z)
        counts += torch.bincount(out["ids"].reshape(-1), minlength=K_size).float()
    util = (counts > 0).sum().item() / K_size
    probs = counts / (counts.sum() + 1e-10)
    perp = (-(probs * (probs + 1e-10).log()).sum()).exp().item()
    return round(util, 4), round(perp, 1)


def train_scale_vq(feats, in_dim, device, epochs=15):
    """Train a VQ on one stage's features. Returns util, perp."""
    proj = nn.Sequential(nn.Linear(in_dim, PROJ_D), nn.LayerNorm(PROJ_D)).to(device)
    vq = VectorQuantizer(codebook_size=K, D=PROJ_D, beta=0.25,
                         use_ema=True, ema_decay=0.99, restart_threshold=1,
                         restart_every=1, l2_normalize=True).to(device)
    N = feats.size(0)
    for ep in range(epochs):
        vq.train(); proj.train()
        idx = torch.randperm(N)
        for i in range(0, N, 128):
            z = proj(feats[idx[i:i+128]].to(device))
            out = vq(z)
    return full_util(vq, feats, proj, device, K)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    loader = collect_loader()

    encoders = {}
    # OPERA-CT raw (no SIGReg)
    enc_opera = build_htsat_encoder(use_csaf=True, freeze_backbone=True).to(device)
    encoders["operaCT_no_sigreg"] = enc_opera

    # LeJEPA (with SIGReg)
    enc_lejepa = build_htsat_encoder(use_csaf=True, freeze_backbone=True)
    if Path(LEJEPA_CKPT).exists():
        ck = torch.load(LEJEPA_CKPT, map_location="cpu", weights_only=False)
        state = {k.replace("encoder.", "", 1): v for k, v in ck["model_state"].items()
                 if k.startswith("encoder.")}
        enc_lejepa.load_state_dict(state, strict=False)
    enc_lejepa = enc_lejepa.to(device)
    encoders["lejepa_sigreg"] = enc_lejepa

    results = {}
    stage_names = ["Stage1 (~1s jitter)", "Stage2 (~2s HNR)",
                   "Stage3 (~4s F0)", "Stage4 (global)"]

    for enc_name, enc in encoders.items():
        print(f"\n{'='*55}\n  Encoder: {enc_name}\n{'='*55}")
        feats = extract_4stages(enc, loader, device)
        results[enc_name] = {}
        for i, (feat, dim, name) in enumerate(zip(feats, STAGE_DIMS, stage_names)):
            util, perp = train_scale_vq(feat, dim, device, epochs=15)
            results[enc_name][f"stage{i+1}"] = {
                "name": name, "dim": dim, "util": util, "perp": perp
            }
            print(f"  {name:25s} dim={dim:4d}  util={util*100:5.1f}%  perp={perp:.0f}")

    # Summary
    print(f"\n{'='*65}")
    print("  SCALE-WISE MULTI-CODEBOOK VQ SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Stage':<28s} {'No SIGReg':>12s}  {'SIGReg':>12s}")
    for i in range(4):
        k = f"stage{i+1}"
        u_no = results["operaCT_no_sigreg"][k]["util"]
        u_si = results["lejepa_sigreg"][k]["util"]
        name = results["operaCT_no_sigreg"][k]["name"]
        print(f"  {name:<28s} {u_no*100:>10.1f}%  {u_si*100:>10.1f}%")
    print(f"{'='*65}")
    print("  If SIGReg lifts util at ALL scales → paper claim upgrade:")
    print('  "SIGReg stabilizes codebooks across all acoustic scales"')

    out = {"experiment": "Scale-wise Multi-Codebook VQ",
           "K": K, "proj_D": PROJ_D, "results": results}
    (Path(OUT_DIR) / "multiscale_vq_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {OUT_DIR}/multiscale_vq_results.json")


if __name__ == "__main__":
    main()
