"""
Modality Ablation: Does adding voice data (Coswara) help respiratory downstream tasks?

Compares:
  A) Respiratory-only pretraining: CoughVID Zenodo only (~30K windows)
  B) Respiratory + Voice pretraining: CoughVID + Coswara (~34K windows)  [current]

Both use the same Stage 3 protocol (ICBHI official split, 5 seeds, z_cont).

Usage:
    python scripts/run_modality_ablation.py --condition respiratory-only
    python scripts/run_modality_ablation.py --condition with-voice
    python scripts/run_modality_ablation.py --condition both
"""

import argparse
import json
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


def run_condition(
    condition_name: str,
    pretrain_cache: str,
    label_cache: str,
    seeds: list,
    checkpoint_base: str,
    log_base: str,
    dim: int = 128,
    encoder_layers: int = 2,
    encoder_heads: int = 4,
    epochs_s1: int = 5,
    epochs_s2: int = 5,
    epochs_s3: int = 64,
    batch_size: int = 64,
):
    from scripts.run_opera_icbhi_disease import (
        official_split, train_stage3_auc, build_model
    )
    from scripts.run_full_local import (
        class_weights_from_labels, evaluate_binary, labels_from_subset
    )
    from sklearn.model_selection import train_test_split

    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}")
    print(f"  Pretrain cache: {pretrain_cache}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    cfg = RespVoiceConfig(
        model=ModelConfig(
            D=dim,
            codebook_size=512,
            encoder_layers=encoder_layers,
            encoder_heads=encoder_heads,
            predictor_layers=1,
            n_sigreg_slices=32,
        ),
        train=TrainConfig(
            stage1_epochs=epochs_s1,
            stage2_epochs=epochs_s2,
            stage3_epochs=epochs_s3,
            batch_size=batch_size,
            stage1_lr=3e-4,
            stage2_lr=1e-4,
            stage3_lr=3e-4,
            lam_sig=0.01,
            lam_recon=0.05,
            warmup_ratio=0.1,
            num_workers=0,
        ),
        checkpoint_dir=f"{checkpoint_base}/stage12",
        log_dir=f"{log_base}/stage12",
    )

    # Stage 1+2: Train once on the pretrain cache
    pretrain_ds = CachedMelDataset(
        root=pretrain_cache,
        meta_file=str(Path(pretrain_cache) / "metadata.json"),
        include_labels=False,
    )
    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )
    print(f"  Pretrain windows: {len(pretrain_ds)}")

    model = RespVoiceModel(cfg.model)
    trainer = Trainer(cfg, model)
    trainer.train_stage1(pretrain_loader)
    trainer.train_stage2(pretrain_loader)
    stage2_ckpt = str(Path(cfg.checkpoint_dir) / "stage2_final.pt")
    print(f"  Stage 1+2 complete. Checkpoint: {stage2_ckpt}")

    # Stage 3: 5 seeds on ICBHI
    label_ds = CachedMelDataset(
        root=label_cache,
        meta_file=str(Path(label_cache) / "metadata.json"),
        include_labels=True,
    )
    train_ds, val_ds, test_ds = official_split(label_ds)
    from torch.utils.data import Subset
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    weights = class_weights_from_labels(labels_from_subset(train_ds), 2)

    seed_results = []
    for seed in seeds:
        import random; random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        seed_cfg = RespVoiceConfig(
            model=cfg.model,
            train=cfg.train,
            checkpoint_dir=f"{checkpoint_base}/seed{seed}",
            log_dir=f"{log_base}/seed{seed}",
        )
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, generator=g)

        # Load Stage 2 checkpoint
        import argparse as _ap
        _args = _ap.Namespace(init_checkpoint=stage2_ckpt, no_ema=False, no_l2_norm=False)
        m = build_model(_args, seed_cfg)

        result = train_stage3_auc(
            seed_cfg, m, train_loader, val_loader,
            n_classes=2, device=trainer.device,
            use_quantized=False, linear_probe=False,
            class_weights=weights,
        )

        # Evaluate best checkpoint on test
        ckpt = torch.load(result["best_path"], map_location="cpu", weights_only=False)
        m_eval = RespVoiceModel(ckpt["config"].model)
        from respvoice.downstream import DownstreamHead
        m_eval.set_downstream_head(DownstreamHead(ckpt["config"].model.D, n_classes=2, use_regression=False))
        m_eval.load_state_dict(ckpt["model_state"], strict=False)
        m_eval.to(trainer.device)
        test_res = evaluate_binary(m_eval, test_loader, trainer.device, use_quantized=False)
        test_auroc = test_res.get("auroc", 0.0)
        print(f"  seed {seed}: val_best_auc={result['best_auc']:.4f}  test_auroc={test_auroc:.4f}")
        seed_results.append({"seed": seed, "val_auc": result["best_auc"], "test_auroc": test_auroc})

    aurocs = [r["test_auroc"] for r in seed_results]
    mean_auc = float(np.mean(aurocs))
    std_auc  = float(np.std(aurocs))
    print(f"\n  {condition_name}: AUROC {mean_auc:.3f} ± {std_auc:.3f}")

    out = {
        "condition": condition_name,
        "pretrain_cache": pretrain_cache,
        "n_pretrain_windows": len(pretrain_ds),
        "seeds": seed_results,
        "auroc_mean": round(mean_auc, 4),
        "auroc_std":  round(std_auc, 4),
    }
    out_path = Path(checkpoint_base) / "modality_ablation_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="both",
                   choices=["respiratory-only", "with-voice", "both"])
    p.add_argument("--label-cache", default="./data/mel_cache/opera_icbhi_disease")
    p.add_argument("--coughvid-only-cache", default="./data/mel_cache/pretrain_coughvid_only")
    p.add_argument("--with-voice-cache",    default="./data/mel_cache/pretrain_zenodo_no_icbhi")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs-s1", type=int, default=5)
    p.add_argument("--epochs-s2", type=int, default=5)
    p.add_argument("--epochs-s3", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--out-dir", default="./checkpoints/modality_ablation")
    args = p.parse_args()

    results = {}

    if args.condition in ("respiratory-only", "both"):
        r = run_condition(
            condition_name="Respiratory-only (CoughVID)",
            pretrain_cache=args.coughvid_only_cache,
            label_cache=args.label_cache,
            seeds=args.seeds,
            checkpoint_base=f"{args.out_dir}/respiratory_only",
            log_base=f"./logs/modality_ablation/respiratory_only",
            dim=args.dim,
            epochs_s1=args.epochs_s1,
            epochs_s2=args.epochs_s2,
            epochs_s3=args.epochs_s3,
            batch_size=args.batch_size,
        )
        results["respiratory_only"] = r

    if args.condition in ("with-voice", "both"):
        r = run_condition(
            condition_name="Respiratory + Voice (CoughVID + Coswara)",
            pretrain_cache=args.with_voice_cache,
            label_cache=args.label_cache,
            seeds=args.seeds,
            checkpoint_base=f"{args.out_dir}/with_voice",
            log_base=f"./logs/modality_ablation/with_voice",
            dim=args.dim,
            epochs_s1=args.epochs_s1,
            epochs_s2=args.epochs_s2,
            epochs_s3=args.epochs_s3,
            batch_size=args.batch_size,
        )
        results["with_voice"] = r

    print("\n" + "="*60)
    print("  MODALITY ABLATION SUMMARY")
    print("="*60)
    for k, v in results.items():
        print(f"  {v['condition']:<45}  AUROC {v['auroc_mean']:.3f} ± {v['auroc_std']:.3f}")

    out_path = Path(args.out_dir) / "modality_ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary saved: {out_path}")


if __name__ == "__main__":
    main()
