"""
Stage 3: Downstream task fine-tuning.

Supports both:
  (a) Linear Probe 鈥?encoder + VQ frozen (matches OPERA benchmark protocol)
  (b) Full Fine-tune 鈥?entire model with small lr

Usage:
  python scripts/stage3_downstream.py --dataset svd --checkpoint ./checkpoints/stage2_best.pt
  python scripts/stage3_downstream.py --dummy --linear-probe
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split
from respvoice.config import RespVoiceConfig, ModelConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer
from respvoice.preprocessing import AudioPreprocessor
from data.respvoice_datasets import LabeledDataset, DummyDataset, DOWNSTREAM_CONFIGS


def parse_args():
    p = argparse.ArgumentParser(description="RespVoice Stage 3: Downstream Fine-tuning")
    p.add_argument("--checkpoint", default=None,
                   help="Path to Stage 2 checkpoint")
    p.add_argument("--dataset", default=None,
                   choices=list(DOWNSTREAM_CONFIGS.keys()),
                   help="Downstream dataset name")
    p.add_argument("--data-dir", default="./data/audio",
                   help="Root directory for audio files")
    p.add_argument("--meta-file", default=None,
                   help="Path to dataset metadata JSON")
    p.add_argument("--n-classes", type=int, default=2,
                   help="Number of output classes")
    p.add_argument("--linear-probe", action="store_true",
                   help="Linear probe protocol (encoder frozen)")
    p.add_argument("--use-quantized", action="store_true", default=True,
                   help="Use quantized z_q instead of continuous z_cont")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--dummy", action="store_true",
                   help="Smoke-test with synthetic data")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    n_classes = args.n_classes
    if args.dataset and args.dataset in DOWNSTREAM_CONFIGS:
        n_classes = DOWNSTREAM_CONFIGS[args.dataset]["n_classes"]
        print(f"Dataset '{args.dataset}': {n_classes} classes")

    cfg = RespVoiceConfig(
        model=ModelConfig(),
        train=TrainConfig(
            stage3_epochs=args.epochs,
            stage3_lr=args.lr,
            batch_size=args.batch_size,
        ),
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    model = RespVoiceModel(cfg.model)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)

    # --- Dataset ---
    if args.dummy:
        full_dataset = DummyDataset(n_samples=200, n_mels=cfg.model.n_mels,
                                    T=250, n_classes=n_classes)
    else:
        if args.meta_file is None:
            raise ValueError("Provide --meta-file for labeled dataset, or use --dummy.")
        preprocessor = AudioPreprocessor(
            sr=cfg.model.sr, n_mels=cfg.model.n_mels,
            win_ms=cfg.model.win_ms, hop_ms=cfg.model.hop_ms,
            target_sec=cfg.model.target_sec,
        )
        full_dataset = LabeledDataset(
            root=args.data_dir, meta_file=args.meta_file,
            preprocessor=preprocessor,
        )

    # Train / val split
    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers)

    print(f"\n=== Stage 3: Downstream Fine-tuning ===")
    print(f"  protocol:      {'Linear Probe' if args.linear_probe else 'Full Fine-tune'}")
    print(f"  use_quantized: {args.use_quantized}")
    print(f"  n_classes:     {n_classes}")
    print(f"  train/val:     {n_train}/{n_val}")

    trainer = Trainer(cfg, model)
    trainer.train_stage3(
        train_loader, val_loader,
        n_classes=n_classes,
        linear_probe=args.linear_probe,
        use_quantized=args.use_quantized,
    )
    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
