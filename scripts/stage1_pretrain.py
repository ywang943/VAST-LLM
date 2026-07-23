"""
Stage 1: LeJEPA self-supervised pretraining.

Usage:
  python scripts/stage1_pretrain.py
  python scripts/stage1_pretrain.py --data-dir ./data/audio --epochs 100 --batch-size 32
  python scripts/stage1_pretrain.py --dummy   # smoke-test with synthetic data (no GPU needed)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader
from respvoice.config import RespVoiceConfig, ModelConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer
from respvoice.preprocessing import AudioPreprocessor
from data.respvoice_datasets import RespVoiceDataset, DummyDataset


def parse_args():
    p = argparse.ArgumentParser(description="RespVoice Stage 1: LeJEPA Pretraining")
    p.add_argument("--data-dir", default="./data/audio")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lam-sig", type=float, default=0.02,
                   help="LeJEPA trade-off 位 (single hyperparameter)")
    p.add_argument("--backbone", default="custom",
                   choices=["custom", "opera_ct", "ast"])
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap dataset size for quick runs")
    p.add_argument("--dummy", action="store_true",
                   help="Use synthetic data for smoke-testing (no audio files needed)")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = RespVoiceConfig(
        model=ModelConfig(backbone=args.backbone),
        train=TrainConfig(
            stage1_epochs=args.epochs,
            stage1_lr=args.lr,
            lam_sig=args.lam_sig,
            batch_size=args.batch_size,
        ),
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("=== RespVoice Stage 1 Configuration ===")
    print(f"  backbone:   {cfg.model.backbone}")
    print(f"  epochs:     {cfg.train.stage1_epochs}")
    print(f"  lr:         {cfg.train.stage1_lr}")
    print(f"  lam_sig:    {cfg.train.lam_sig}  鈫?only LeJEPA hyperparameter")
    print(f"  batch_size: {cfg.train.batch_size}")

    if args.dummy:
        print("\n[Dummy mode] Using synthetic data 鈥?no audio files required.")
        dataset = DummyDataset(n_samples=128, n_mels=cfg.model.n_mels, T=250)
    else:
        preprocessor = AudioPreprocessor(
            sr=cfg.model.sr,
            n_mels=cfg.model.n_mels,
            win_ms=cfg.model.win_ms,
            hop_ms=cfg.model.hop_ms,
            target_sec=cfg.model.target_sec,
        )
        dataset = RespVoiceDataset(
            root=args.data_dir,
            preprocessor=preprocessor,
            max_samples=args.max_samples,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers,
    )

    model = RespVoiceModel(cfg.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total_params:,}")

    trainer = Trainer(cfg, model)
    trainer.train_stage1(loader)
    print("\nStage 1 complete. Checkpoint saved to:", cfg.checkpoint_dir)


if __name__ == "__main__":
    main()
