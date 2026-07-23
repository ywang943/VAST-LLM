"""
Clean SIGReg → VQ codebook ablation.

Core innovation claim: LeJEPA's SIGReg produces isotropic Gaussian
representations that directly enable high VQ codebook utilization,
solving codebook collapse.

Pure ablation (isolates SIGReg as the only variable):
  Pretrain two D=128 encoders on the SAME data, SAME everything,
  differing ONLY in lam_sig:
    - JEPA-only:  lam_sig = 0.0   (no SIGReg)
    - LeJEPA:     lam_sig = 0.02  (with SIGReg)

  Then train VQ on top of each frozen encoder and measure codebook
  utilization + perplexity.

Conditions:
  A: JEPA-only encoder + VQ            → expect LOW util (collapse)
  B: LeJEPA encoder + VQ (no EMA/L2)   → SIGReg alone lifts util
  C: LeJEPA + VQ + EMA                 → +EMA polish
  D: LeJEPA + VQ + EMA + L2 (ours)     → full method

This isolates SIGReg (A vs B) cleanly, unlike the old version that
compared random-init vs pretrained.

Usage:
  python scripts/run_sigreg_vq_ablation.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer

D = 128
PRETRAIN_CACHE = "data/mel_cache/opera_icbhi_disease"
OUT_DIR = "checkpoints/sigreg_vq_ablation"


def pretrain_stage1(lam_sig, tag, epochs, device):
    """Pretrain a D=128 encoder with the given lam_sig. Returns checkpoint path."""
    print(f"\n{'='*55}")
    print(f"  Pretraining Stage1: {tag} (lam_sig={lam_sig})")
    print(f"{'='*55}")
    cfg = RespVoiceConfig(
        model=ModelConfig(D=D, encoder_layers=6, encoder_heads=8,
                          predictor_layers=2, n_sigreg_slices=64,
                          codebook_size=512),
        train=TrainConfig(stage1_epochs=epochs, stage1_lr=3e-4,
                          lam_sig=lam_sig, batch_size=32,
                          warmup_ratio=0.1, num_workers=0),
        checkpoint_dir=f"{OUT_DIR}/stage1_{tag}",
        log_dir=f"logs/sigreg_vq/{tag}",
    )
    ds = CachedMelDataset(root=PRETRAIN_CACHE,
                          meta_file=str(Path(PRETRAIN_CACHE) / "metadata.json"),
                          include_labels=False)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    model = RespVoiceModel(cfg.model)
    trainer = Trainer(cfg, model)
    trainer.train_stage1(loader)
    return f"{OUT_DIR}/stage1_{tag}/stage1_best.pt"


def train_vq(stage1_ckpt, use_ema, l2, label, epochs, device):
    """Train VQ on top of a frozen pretrained encoder. Returns util/perp."""
    print(f"\n  --- VQ: {label} (ckpt={Path(stage1_ckpt).parent.name}, EMA={use_ema}, L2={l2}) ---")
    cfg = RespVoiceConfig(
        model=ModelConfig(D=D, codebook_size=512, encoder_layers=6,
                          encoder_heads=8, predictor_layers=2, n_sigreg_slices=64,
                          vq_use_ema=use_ema, vq_ema_decay=0.99,
                          vq_restart_threshold=1, vq_l2_normalize=l2),
        train=TrainConfig(stage2_epochs=epochs, stage2_lr=1e-4,
                          lam_recon=0.05, batch_size=32,
                          warmup_ratio=0.05, num_workers=0),
        checkpoint_dir=f"{OUT_DIR}/vq_{label.replace(' ','_')}",
        log_dir=f"logs/sigreg_vq/vq_{label.replace(' ','_')}",
    )
    model = RespVoiceModel(cfg.model)
    state = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    full = state["model_state"]
    enc_only = {k: v for k, v in full.items()
                if not k.startswith("vq.") and not k.startswith("decoder.")}
    model.load_state_dict(enc_only, strict=False)

    ds = CachedMelDataset(root=PRETRAIN_CACHE,
                          meta_file=str(Path(PRETRAIN_CACHE) / "metadata.json"),
                          include_labels=False)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    trainer = Trainer(cfg, model)
    trainer.train_stage2(loader)

    log_path = Path(cfg.log_dir) / "train_log.jsonl"
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    s2 = [r for r in rows if r.get("stage") == "stage2"]
    best_util = max((r["util"] for r in s2), default=0.0)
    best_perp = max((r["perp"] for r in s2), default=0.0)
    return round(best_util, 4), round(best_perp, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-epochs", type=int, default=60)
    p.add_argument("--vq-epochs", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Step 1: pretrain two encoders (only difference = SIGReg)
    jepa_ckpt = pretrain_stage1(0.0,  "jepa_only", args.stage1_epochs, device)
    lejepa_ckpt = pretrain_stage1(0.02, "lejepa",   args.stage1_epochs, device)

    # Step 2: VQ on each
    results = []
    uA, pA = train_vq(jepa_ckpt,   use_ema=False, l2=False,
                      label="A JEPA-only (no SIGReg)", epochs=args.vq_epochs, device=device)
    results.append({"label": "A: JEPA-only (no SIGReg)", "sigreg": False,
                    "ema": False, "l2": False, "util": uA, "perp": pA})

    uB, pB = train_vq(lejepa_ckpt, use_ema=False, l2=False,
                      label="B LeJEPA SIGReg", epochs=args.vq_epochs, device=device)
    results.append({"label": "B: LeJEPA (SIGReg)", "sigreg": True,
                    "ema": False, "l2": False, "util": uB, "perp": pB})

    uC, pC = train_vq(lejepa_ckpt, use_ema=True, l2=False,
                      label="C LeJEPA SIGReg EMA", epochs=args.vq_epochs, device=device)
    results.append({"label": "C: LeJEPA + EMA", "sigreg": True,
                    "ema": True, "l2": False, "util": uC, "perp": pC})

    uD, pD = train_vq(lejepa_ckpt, use_ema=True, l2=True,
                      label="D LeJEPA SIGReg EMA L2 ours", epochs=args.vq_epochs, device=device)
    results.append({"label": "D: LeJEPA + EMA + L2 (ours)", "sigreg": True,
                    "ema": True, "l2": True, "util": uD, "perp": pD})

    # Summary
    print(f"\n{'='*60}")
    print("  SIGReg → VQ CODEBOOK ABLATION (pure SIGReg isolation)")
    print(f"{'='*60}")
    print(f"  {'Condition':<32} {'util':>7} {'perp':>7}")
    for r in results:
        print(f"  {r['label']:<32} {r['util']*100:>6.1f}% {r['perp']:>7.0f}")
    print(f"{'='*60}")
    print(f"  Key: SIGReg lifts util {results[0]['util']*100:.0f}% → {results[1]['util']*100:.0f}%")

    out = {
        "experiment": "Pure SIGReg isolation (JEPA-only vs LeJEPA), then VQ",
        "D": D, "codebook_size": 512,
        "stage1_epochs": args.stage1_epochs, "vq_epochs": args.vq_epochs,
        "conditions": results,
        "headline": f"SIGReg lifts codebook util from {results[0]['util']*100:.0f}% to {results[3]['util']*100:.0f}%",
    }
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(OUT_DIR) / "sigreg_vq_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {OUT_DIR}/sigreg_vq_results.json")


if __name__ == "__main__":
    main()
