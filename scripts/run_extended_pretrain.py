"""
Extended LeJEPA pretraining: warm-start from existing checkpoint, run N more epochs.

Goal: improve LP AUROC from 0.770 (5 epoch) toward ~0.80+ (15 epoch).

Strategy:
  - Warm-start from Stage1 checkpoint (5 epochs already done)
  - Run 10 more epochs with cosine LR from warm start
  - Use pretrain_zenodo_no_icbhi (strict no-leakage)
  - Re-run VQ Stage2 + Linear Probe evaluation (5 seeds)

Usage:
    python scripts/run_extended_pretrain.py
    python scripts/run_extended_pretrain.py --extra-epochs 15
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
from scripts.run_opera_icbhi_disease import (
    official_split, train_stage3_auc
)
from scripts.run_full_local import (
    class_weights_from_labels, evaluate_binary, labels_from_subset
)
from respvoice.downstream import DownstreamHead

PRETRAIN_CACHE = "./data/mel_cache/pretrain_zenodo_no_icbhi"
LABEL_CACHE    = "./data/mel_cache/opera_icbhi_disease"
SEEDS = [0, 1, 2, 3, 4]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-s1", default="./checkpoints/opera_official_icbhi_cont_ft/stage1_final.pt",
                   help="Existing Stage1 checkpoint to warm-start from")
    p.add_argument("--extra-epochs", type=int, default=10,
                   help="Additional LeJEPA pretraining epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--encoder-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4,  # lower LR for warm-start
                   help="Learning rate for continued pretraining (lower than initial)")
    p.add_argument("--lam-sig", type=float, default=0.01)
    p.add_argument("--out-dir", default="./checkpoints/lejepa_extended")
    p.add_argument("--log-dir", default="./logs/lejepa_extended")
    p.add_argument("--epochs-s3", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Warm-starting from: {args.init_s1}")
    print(f"Additional epochs: {args.extra_epochs}")
    print(f"Continued pretraining LR: {args.lr}")

    cfg = RespVoiceConfig(
        model=ModelConfig(
            D=args.dim,
            codebook_size=512,
            encoder_layers=args.encoder_layers,
            encoder_heads=args.encoder_heads,
            predictor_layers=1,
            n_sigreg_slices=32,
        ),
        train=TrainConfig(
            stage1_epochs=args.extra_epochs,
            stage1_lr=args.lr,
            stage2_epochs=5,
            stage2_lr=1e-4,
            stage3_epochs=args.epochs_s3,
            batch_size=args.batch_size,
            lam_sig=args.lam_sig,
            lam_recon=0.05,
            warmup_ratio=0.05,  # short warmup for continued training
            num_workers=0,
        ),
        checkpoint_dir=args.out_dir,
        log_dir=args.log_dir,
    )

    model = RespVoiceModel(cfg.model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Warm-start: load existing Stage1 encoder
    if Path(args.init_s1).exists():
        raw = torch.load(args.init_s1, map_location="cpu", weights_only=False)
        full = raw["model_state"]
        current = model.state_dict()
        compat = {k: v for k, v in full.items()
                  if k in current and current[k].shape == v.shape}
        model.load_state_dict(compat, strict=False)
        loaded = len(compat)
        print(f"Loaded {loaded}/{len(full)} keys from warm-start checkpoint")
    else:
        print(f"WARNING: checkpoint not found, starting from scratch")

    # Pretraining data
    pretrain_ds = CachedMelDataset(
        root=PRETRAIN_CACHE,
        meta_file=str(Path(PRETRAIN_CACHE) / "metadata.json"),
        include_labels=False,
    )
    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    print(f"Pretrain windows: {len(pretrain_ds)}")

    trainer = Trainer(cfg, model)

    # Extended Stage1
    print(f"\n--- Extended LeJEPA Stage1 ({args.extra_epochs} more epochs) ---")
    trainer.train_stage1(pretrain_loader)

    # Stage2 VQ
    print("\n--- Stage2 VQ (5 epochs) ---")
    trainer.train_stage2(pretrain_loader)

    stage2_ckpt = str(Path(args.out_dir) / "stage2_final.pt")
    print(f"Stage2 checkpoint: {stage2_ckpt}")

    # Stage3 linear probe, 5 seeds
    label_ds = CachedMelDataset(
        root=LABEL_CACHE,
        meta_file=str(Path(LABEL_CACHE) / "metadata.json"),
        include_labels=True,
    )
    train_ds, val_ds, test_ds = official_split(label_ds)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    weights = class_weights_from_labels(labels_from_subset(train_ds), 2)

    print("\n--- Stage3 Linear Probe (5 seeds) ---")
    seed_results = []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        seed_cfg = RespVoiceConfig(
            model=cfg.model, train=cfg.train,
            checkpoint_dir=f"{args.out_dir}/lp_seed{seed}",
            log_dir=f"{args.log_dir}/lp_seed{seed}",
        )
        m = RespVoiceModel(cfg.model)
        raw = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
        cur = m.state_dict()
        compat = {k: v for k, v in raw["model_state"].items()
                  if k in cur and cur[k].shape == v.shape}
        m.load_state_dict(compat, strict=False)

        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, generator=g)

        result = train_stage3_auc(
            seed_cfg, m, train_loader, val_loader,
            n_classes=2, device=device,
            use_quantized=False, linear_probe=True,
            class_weights=weights,
        )

        ckpt = torch.load(result["best_path"], map_location="cpu", weights_only=False)
        m_eval = RespVoiceModel(ckpt["config"].model)
        m_eval.set_downstream_head(DownstreamHead(cfg.model.D, n_classes=2, use_regression=False))
        m_eval.load_state_dict(ckpt["model_state"], strict=False)
        m_eval.to(device)
        test_res = evaluate_binary(m_eval, test_loader, device, use_quantized=False)
        test_auroc = float(test_res.get("auroc", 0.0))
        print(f"  seed {seed}: val={result['best_auc']:.4f}  test={test_auroc:.4f}")
        seed_results.append({"seed": seed, "test_auroc": test_auroc})

    aurocs = [r["test_auroc"] for r in seed_results]
    mean_auc = float(np.mean(aurocs))
    std_auc  = float(np.std(aurocs))

    summary = {
        "description": f"LeJEPA warm-start +{args.extra_epochs} epochs → total ~15 epochs",
        "init_checkpoint": args.init_s1,
        "extra_epochs": args.extra_epochs,
        "lr_continued": args.lr,
        "auroc_mean": round(mean_auc, 4),
        "auroc_std":  round(std_auc, 4),
        "per_seed": [round(a, 4) for a in aurocs],
        "seed_results": seed_results,
        "protocol": "linear_probe",
    }
    out = Path(args.out_dir) / "extended_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  EXTENDED PRETRAIN LP AUROC: {mean_auc:.3f} +- {std_auc:.3f}")
    print(f"  (vs 5-epoch baseline: 0.770 +- 0.034)")
    print(f"  per-seed: {[round(a,3) for a in aurocs]}")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
