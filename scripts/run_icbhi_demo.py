"""
Small real-data RespVoice demo on ICBHI.

This is intentionally smaller than the paper-scale default config. It uses the
downloaded 174 ICBHI wav files to verify that the full pipeline trains on real
audio in this local Windows/Python 3.13 environment.

Usage:
  python scripts/run_icbhi_demo.py
  python scripts/run_icbhi_demo.py --max-samples 64 --epochs 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split

from data.respvoice_datasets import LabeledDataset, RespVoiceDataset
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.preprocessing import AudioPreprocessor
from respvoice.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Run a small real-data ICBHI demo")
    p.add_argument("--data-dir", default="./data/audio/icbhi")
    p.add_argument("--meta-file", default="./data/audio/icbhi/metadata.json")
    p.add_argument("--checkpoint-dir", default="./checkpoints/run_icbhi")
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--target-sec", type=float, default=8.0)
    p.add_argument("--n-classes", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = RespVoiceConfig(
        model=ModelConfig(
            D=args.dim,
            codebook_size=args.codebook_size,
            target_sec=args.target_sec,
            encoder_layers=2,
            encoder_heads=4,
            predictor_layers=1,
            n_sigreg_slices=32,
            backbone="custom",
        ),
        train=TrainConfig(
            stage1_epochs=args.epochs,
            stage2_epochs=args.epochs,
            stage3_epochs=args.epochs,
            batch_size=args.batch_size,
            stage1_lr=3e-4,
            stage2_lr=1e-4,
            stage3_lr=1e-3,
            lam_sig=0.01,
            lam_recon=0.05,
            warmup_ratio=0.2,
            num_workers=0,
        ),
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir="./logs/icbhi_demo",
    )

    preprocessor = AudioPreprocessor(
        sr=cfg.model.sr,
        n_mels=cfg.model.n_mels,
        win_ms=cfg.model.win_ms,
        hop_ms=cfg.model.hop_ms,
        target_sec=cfg.model.target_sec,
    )

    pretrain_ds = RespVoiceDataset(
        root=args.data_dir,
        preprocessor=preprocessor,
        max_samples=args.max_samples,
    )
    pretrain_loader = DataLoader(
        pretrain_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    labeled_ds = LabeledDataset(
        root=args.data_dir,
        meta_file=args.meta_file,
        preprocessor=preprocessor,
    )
    if args.max_samples:
        labeled_ds.samples = labeled_ds.samples[: args.max_samples]

    n_val = max(8, int(len(labeled_ds) * 0.2))
    n_train = len(labeled_ds) - n_val
    train_ds, val_ds = random_split(
        labeled_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(2026),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = RespVoiceModel(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())

    print("=== RespVoice ICBHI real-data demo ===")
    print(f"  device cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  samples: {len(pretrain_ds)} pretrain, {n_train}/{n_val} train/val")
    print(f"  model: D={args.dim}, layers=2, codebook={args.codebook_size}")
    print(f"  parameters: {n_params:,}")

    trainer = Trainer(cfg, model)
    trainer.train_stage1(pretrain_loader)
    trainer.train_stage2(pretrain_loader)
    trainer.train_stage3(
        train_loader,
        val_loader=val_loader,
        n_classes=args.n_classes,
        linear_probe=True,
        use_quantized=True,
    )

    model.eval()
    batch = next(iter(pretrain_loader))["mel"].to(trainer.device)
    with torch.no_grad():
        ids = model.encode_to_tokens(batch[:2])
    print("\nToken demo:")
    print(f"  ids shape: {tuple(ids.shape)}")
    print(f"  min/max id: {ids.min().item()} / {ids.max().item()}")
    print(f"  checkpoints: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
