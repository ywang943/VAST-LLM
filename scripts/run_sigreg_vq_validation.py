"""
Multi-seed validation of the SIGReg → VQ codebook claim (headline evidence).

The single-seed pure ablation showed JEPA-only (0.2%) vs LeJEPA (99.8%).
This validates robustness across seeds: for each seed, pretrain BOTH a
JEPA-only and a LeJEPA encoder (identical except lam_sig), then run VQ and
record codebook utilization + perplexity. Report mean ± std across seeds.

Also computes FULL-DATASET utilization (not just per-batch best) for a more
rigorous, defensible paper number.

Conditions:
  A: JEPA-only (no SIGReg) + VQ
  B: LeJEPA (SIGReg) + VQ
  C: LeJEPA + EMA
  D: LeJEPA + EMA + L2 (ours)

Usage:
  python scripts/run_sigreg_vq_validation.py --seeds 0 1 2
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer

D = 128
PRETRAIN_CACHE = "data/mel_cache/opera_icbhi_disease"
OUT_DIR = "checkpoints/sigreg_vq_validation"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def pretrain(lam_sig, tag, seed, epochs):
    set_seed(seed)
    cfg = RespVoiceConfig(
        model=ModelConfig(D=D, encoder_layers=6, encoder_heads=8,
                          predictor_layers=2, n_sigreg_slices=64, codebook_size=512),
        train=TrainConfig(stage1_epochs=epochs, stage1_lr=3e-4, lam_sig=lam_sig,
                          batch_size=32, warmup_ratio=0.1, num_workers=0),
        checkpoint_dir=f"{OUT_DIR}/s{seed}_stage1_{tag}",
        log_dir=f"logs/sigreg_vq_val/s{seed}_{tag}",
    )
    ds = CachedMelDataset(root=PRETRAIN_CACHE,
                          meta_file=str(Path(PRETRAIN_CACHE) / "metadata.json"))
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    model = RespVoiceModel(cfg.model)
    Trainer(cfg, model).train_stage1(loader)
    return f"{OUT_DIR}/s{seed}_stage1_{tag}/stage1_best.pt"


@torch.no_grad()
def full_dataset_util(model, loader, device):
    """Codebook utilization + perplexity over the ENTIRE dataset (not per-batch)."""
    model.eval()
    K = model.vq.codebook_size
    counts = torch.zeros(K, device=device)
    for batch in loader:
        mel = batch["mel"].to(device)
        z = model.encoder(mel)
        out = model.vq(z)
        ids = out["ids"].reshape(-1)
        counts += torch.bincount(ids, minlength=K).float()
    util = (counts > 0).sum().item() / K
    probs = counts / (counts.sum() + 1e-10)
    perp = (-(probs * (probs + 1e-10).log()).sum()).exp().item()
    return util, perp


def train_vq(stage1_ckpt, use_ema, l2, tag, seed, epochs, device):
    set_seed(seed)
    cfg = RespVoiceConfig(
        model=ModelConfig(D=D, codebook_size=512, encoder_layers=6, encoder_heads=8,
                          predictor_layers=2, n_sigreg_slices=64,
                          vq_use_ema=use_ema, vq_ema_decay=0.99,
                          vq_restart_threshold=1, vq_l2_normalize=l2),
        train=TrainConfig(stage2_epochs=epochs, stage2_lr=1e-4, lam_recon=0.05,
                          batch_size=32, warmup_ratio=0.05, num_workers=0),
        checkpoint_dir=f"{OUT_DIR}/s{seed}_vq_{tag}",
        log_dir=f"logs/sigreg_vq_val/s{seed}_vq_{tag}",
    )
    model = RespVoiceModel(cfg.model)
    state = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)["model_state"]
    enc = {k: v for k, v in state.items() if not k.startswith("vq.") and not k.startswith("decoder.")}
    model.load_state_dict(enc, strict=False)
    ds = CachedMelDataset(root=PRETRAIN_CACHE,
                          meta_file=str(Path(PRETRAIN_CACHE) / "metadata.json"))
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    Trainer(cfg, model).train_stage2(loader)
    # rigorous full-dataset metric
    eval_loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    util, perp = full_dataset_util(model, eval_loader, model.device if hasattr(model, "device") else device)
    return util, perp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--stage1-epochs", type=int, default=60)
    p.add_argument("--vq-epochs", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    conds = {"A_jepa_only": [], "B_lejepa": [], "C_lejepa_ema": [], "D_lejepa_ema_l2": []}
    perps = {k: [] for k in conds}

    for seed in args.seeds:
        print(f"\n{'#'*60}\n#  SEED {seed}\n{'#'*60}")
        jepa = pretrain(0.0,  "jepa", seed, args.stage1_epochs)
        leje = pretrain(0.02, "leje", seed, args.stage1_epochs)

        uA, pA = train_vq(jepa, False, False, "A", seed, args.vq_epochs, device)
        uB, pB = train_vq(leje, False, False, "B", seed, args.vq_epochs, device)
        uC, pC = train_vq(leje, True,  False, "C", seed, args.vq_epochs, device)
        uD, pD = train_vq(leje, True,  True,  "D", seed, args.vq_epochs, device)

        for k, u, pp in [("A_jepa_only", uA, pA), ("B_lejepa", uB, pB),
                         ("C_lejepa_ema", uC, pC), ("D_lejepa_ema_l2", uD, pD)]:
            conds[k].append(u); perps[k].append(pp)
        print(f"  seed{seed}: A={uA*100:.1f}% B={uB*100:.1f}% C={uC*100:.1f}% D={uD*100:.1f}%")

    # Aggregate
    print(f"\n{'='*65}")
    print("  SIGReg → VQ VALIDATION (full-dataset util, mean ± std)")
    print(f"{'='*65}")
    labels = {"A_jepa_only": "A: JEPA-only (no SIGReg)",
              "B_lejepa": "B: LeJEPA (SIGReg)",
              "C_lejepa_ema": "C: LeJEPA + EMA",
              "D_lejepa_ema_l2": "D: LeJEPA + EMA + L2 (ours)"}
    summary = {}
    for k in conds:
        u = np.array(conds[k]); pp = np.array(perps[k])
        summary[k] = {"util_mean": float(u.mean()), "util_std": float(u.std()),
                      "perp_mean": float(pp.mean()), "perp_std": float(pp.std()),
                      "util_per_seed": [round(x, 4) for x in conds[k]]}
        print(f"  {labels[k]:<30} util={u.mean()*100:5.1f}±{u.std()*100:.1f}%  perp={pp.mean():.0f}±{pp.std():.0f}")
    print(f"{'='*65}")
    a, b = summary["A_jepa_only"]["util_mean"], summary["D_lejepa_ema_l2"]["util_mean"]
    print(f"  SIGReg lifts util: {a*100:.1f}% → {b*100:.1f}%")

    out = {"experiment": "Multi-seed pure SIGReg→VQ validation (full-dataset util)",
           "D": D, "codebook_size": 512, "seeds": args.seeds,
           "conditions": summary,
           "headline": f"SIGReg lifts codebook util from {a*100:.1f}% to {b*100:.1f}% (mean over {len(args.seeds)} seeds)"}
    (Path(OUT_DIR) / "validation_results.json").write_text(json.dumps(out, indent=2))
    print(f"  Saved: {OUT_DIR}/validation_results.json")


if __name__ == "__main__":
    main()
