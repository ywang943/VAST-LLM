"""
Downstream AUROC ablation: tests contribution of each component.

Conditions (all D=128, official ICBHI split, 5 seeds):
  A) Random encoder (no pretraining)                       -- baseline
  B) JEPA only (no SIGReg, lam_sig=0)                     -- removes SIGReg
  C) Full LeJEPA (JEPA + SIGReg) → z_cont (MAIN RESULT)   -- already done
  D) Full LeJEPA → z_q (with VQ, EMA+L2)                  -- VQ for downstream

This shows which components drive AUROC improvements.

Usage:
    python scripts/run_downstream_ablation.py --conditions A B D
    python scripts/run_downstream_ablation.py --conditions A B C D
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from data.respvoice_datasets import CachedMelDataset
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.downstream import DownstreamHead
from respvoice.trainer import Trainer
from scripts.run_full_local import (
    class_weights_from_labels, evaluate_binary, labels_from_subset
)
from scripts.run_opera_icbhi_disease import (
    official_split, train_stage3_auc
)


# ---------------------------------------------------------------------------
# Common config and helpers
# ---------------------------------------------------------------------------

PRETRAIN_CACHE  = "./data/mel_cache/pretrain_zenodo_no_icbhi"
LABEL_CACHE     = "./data/mel_cache/opera_icbhi_disease"
SEEDS           = [0, 1, 2, 3, 4]
BATCH_SIZE      = 64
EPOCHS_S1       = 5
EPOCHS_S2       = 5
EPOCHS_S3       = 64
D               = 128
ENCODER_LAYERS  = 2
ENCODER_HEADS   = 4


def make_cfg(checkpoint_dir, log_dir, lam_sig=0.01, vq_l2=True, vq_ema=True,
             codebook_size=512, stage1_epochs=EPOCHS_S1, stage2_epochs=EPOCHS_S2):
    return RespVoiceConfig(
        model=ModelConfig(
            D=D,
            codebook_size=codebook_size,
            encoder_layers=ENCODER_LAYERS,
            encoder_heads=ENCODER_HEADS,
            predictor_layers=1,
            n_sigreg_slices=32,
            vq_use_ema=vq_ema,
            vq_l2_normalize=vq_l2,
            vq_restart_threshold=1,
        ),
        train=TrainConfig(
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            stage3_epochs=EPOCHS_S3,
            batch_size=BATCH_SIZE,
            stage1_lr=3e-4,
            stage2_lr=1e-4,
            stage3_lr=3e-4,
            lam_sig=lam_sig,
            lam_recon=0.05,
            warmup_ratio=0.1,
            num_workers=0,
        ),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )


def run_stage3_seeds(stage2_ckpt_path, cfg, label_cache, seeds,
                     use_quantized, device, out_dir, linear_probe=False):
    label_ds = CachedMelDataset(
        root=label_cache,
        meta_file=str(Path(label_cache) / "metadata.json"),
        include_labels=True,
    )
    train_ds, val_ds, test_ds = official_split(label_ds)
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    weights = class_weights_from_labels(labels_from_subset(train_ds), 2)

    seed_results = []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        seed_cfg = RespVoiceConfig(
            model=cfg.model,
            train=cfg.train,
            checkpoint_dir=f"{out_dir}/seed{seed}",
            log_dir=f"{out_dir}/log_seed{seed}",
        )
        model = RespVoiceModel(cfg.model)

        # Load encoder weights from stage 2 checkpoint
        if stage2_ckpt_path and Path(stage2_ckpt_path).exists():
            raw = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
            full = raw["model_state"]
            current = model.state_dict()
            compat = {k: v for k, v in full.items()
                      if k in current and current[k].shape == v.shape}
            model.load_state_dict(compat, strict=False)

        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, generator=g
        )
        result = train_stage3_auc(
            seed_cfg, model, train_loader, val_loader,
            n_classes=2, device=device,
            use_quantized=use_quantized,
            linear_probe=linear_probe,
            class_weights=weights,
        )
        # Evaluate on test
        ckpt = torch.load(result["best_path"], map_location="cpu", weights_only=False)
        m_eval = RespVoiceModel(ckpt["config"].model)
        m_eval.set_downstream_head(
            DownstreamHead(ckpt["config"].model.D, n_classes=2, use_regression=False)
        )
        m_eval.load_state_dict(ckpt["model_state"], strict=False)
        m_eval.to(device)
        test_res = evaluate_binary(m_eval, test_loader, device, use_quantized=use_quantized)
        test_auroc = float(test_res.get("auroc", 0.0))
        print(f"    seed {seed}: val={result['best_auc']:.4f}  test={test_auroc:.4f}")
        seed_results.append({
            "seed": seed, "val_auc": result["best_auc"], "test_auroc": test_auroc
        })

    aurocs = [r["test_auroc"] for r in seed_results]
    return {
        "seed_results": seed_results,
        "auroc_mean": round(float(np.mean(aurocs)), 4),
        "auroc_std":  round(float(np.std(aurocs)), 4),
        "per_seed": [round(a, 4) for a in aurocs],
    }


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def run_A_random_encoder(out_dir, device, linear_probe=False):
    """No pretraining: random encoder + downstream."""
    mode = "Linear Probe" if linear_probe else "Fine-tune"
    print(f"\n=== Condition A: Random Encoder ({mode}) ===")
    cfg = make_cfg(f"{out_dir}/A_random", f"{out_dir}/A_random_log")
    return run_stage3_seeds(
        stage2_ckpt_path=None, cfg=cfg,
        label_cache=LABEL_CACHE, seeds=SEEDS,
        use_quantized=False, device=device,
        out_dir=f"{out_dir}/A_random",
        linear_probe=linear_probe,
    )


def run_B_jepa_only(out_dir, device, linear_probe=False):
    """JEPA pretraining without SIGReg (lam_sig=0), then z_cont downstream."""
    mode = "Linear Probe" if linear_probe else "Fine-tune"
    print(f"\n=== Condition B: JEPA Only (no SIGReg, {mode}) ===")
    cfg = make_cfg(f"{out_dir}/B_jepa_only", f"{out_dir}/B_jepa_only_log",
                   lam_sig=0.0)
    pretrain_ds = CachedMelDataset(
        root=PRETRAIN_CACHE,
        meta_file=str(Path(PRETRAIN_CACHE) / "metadata.json"),
        include_labels=False,
    )
    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True, num_workers=0)
    model = RespVoiceModel(cfg.model)
    trainer = Trainer(cfg, model)
    trainer.train_stage1(pretrain_loader)
    stage1_ckpt = str(Path(cfg.checkpoint_dir) / "stage1_final.pt")
    return run_stage3_seeds(
        stage2_ckpt_path=stage1_ckpt, cfg=cfg,
        label_cache=LABEL_CACHE, seeds=SEEDS,
        use_quantized=False, device=device,
        out_dir=f"{out_dir}/B_jepa_only",
        linear_probe=linear_probe,
    )


def run_D_lejepa_zq(out_dir, device, linear_probe=False):
    """Full LeJEPA → Stage 2 VQ (EMA+L2) → z_q downstream."""
    print("\n=== Condition D: Full LeJEPA + VQ → z_q downstream ===")
    # Use the existing Stage 1 encoder from the main OPERA run
    stage1_ckpt = "./checkpoints/opera_official_icbhi_cont_ft/stage2_final.pt"
    cfg = make_cfg(f"{out_dir}/D_lejepa_zq", f"{out_dir}/D_lejepa_zq_log",
                   codebook_size=512, vq_ema=True, vq_l2=True)

    # Stage 2: VQ tokenizer on top of existing Stage 1 encoder
    pretrain_ds = CachedMelDataset(
        root=PRETRAIN_CACHE,
        meta_file=str(Path(PRETRAIN_CACHE) / "metadata.json"),
        include_labels=False,
    )
    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True, num_workers=0)
    model = RespVoiceModel(cfg.model)
    # Load Stage 1 encoder
    raw = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    current = model.state_dict()
    compat = {k: v for k, v in raw["model_state"].items()
              if k in current and current[k].shape == v.shape}
    model.load_state_dict(compat, strict=False)

    trainer = Trainer(cfg, model)
    trainer.train_stage2(pretrain_loader)  # only VQ, encoder frozen
    stage2_ckpt_path = str(Path(cfg.checkpoint_dir) / "stage2_final.pt")
    print(f"  Stage 2 complete: {stage2_ckpt_path}")

    return run_stage3_seeds(
        stage2_ckpt_path=stage2_ckpt_path, cfg=cfg,
        label_cache=LABEL_CACHE, seeds=SEEDS,
        use_quantized=True,  # USE z_q!
        device=device,
        out_dir=f"{out_dir}/D_lejepa_zq",
        linear_probe=linear_probe,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=["A", "B", "D"],
                   choices=["A", "B", "D"],
                   help="A=random, B=jepa-only, D=lejepa-zq")
    p.add_argument("--linear-probe", action="store_true",
                   help="Freeze encoder, only train downstream head (shows pretraining benefit)")
    p.add_argument("--out-dir", default="./checkpoints/downstream_ablation")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running conditions: {args.conditions}")
    print(f"NOTE: Condition C (LeJEPA z_cont) = existing result 0.917 ± 0.037")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "C_lejepa_zcont": {
            "description": "Full LeJEPA + z_cont (main result, already done)",
            "auroc_mean": 0.917, "auroc_std": 0.037,
            "per_seed": [0.919, 0.856, 0.932, 0.909, 0.970],
        }
    }

    lp = args.linear_probe
    if "A" in args.conditions:
        r = run_A_random_encoder(out_dir, device, linear_probe=lp)
        results["A_random"] = {
            "description": "Random encoder (no pretraining)", "linear_probe": lp, **r
        }
        print(f"  A (random): {r['auroc_mean']:.3f} +- {r['auroc_std']:.3f}")

    if "B" in args.conditions:
        r = run_B_jepa_only(out_dir, device, linear_probe=lp)
        results["B_jepa_only"] = {
            "description": "JEPA only (no SIGReg, lam_sig=0)", "linear_probe": lp, **r
        }
        print(f"  B (JEPA only): {r['auroc_mean']:.3f} +- {r['auroc_std']:.3f}")

    if "D" in args.conditions:
        r = run_D_lejepa_zq(out_dir, device, linear_probe=lp)
        results["D_lejepa_zq"] = {
            "description": "Full LeJEPA + VQ (EMA+L2) -> z_q downstream", "linear_probe": lp, **r
        }
        print(f"  D (LeJEPA+VQ z_q): {r['auroc_mean']:.3f} +- {r['auroc_std']:.3f}")

    # Summary table
    print("\n" + "="*65)
    print("  DOWNSTREAM AUROC ABLATION SUMMARY")
    print("="*65)
    order = ["A_random", "B_jepa_only", "C_lejepa_zcont", "D_lejepa_zq"]
    labels = {
        "A_random":       "A: Random encoder (no pretraining)",
        "B_jepa_only":    "B: JEPA only (no SIGReg)",
        "C_lejepa_zcont": "C: Full LeJEPA, z_cont  [main]",
        "D_lejepa_zq":    "D: Full LeJEPA, z_q (VQ)",
    }
    for key in order:
        if key in results:
            r = results[key]
            print(f"  {labels[key]:<42}  {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")

    out_path = Path(out_dir) / "downstream_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
