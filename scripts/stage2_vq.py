"""
Stage 2: VQ tokenizer training.

Encoder is frozen (loaded from Stage 1 checkpoint).
Only the VQ codebook and optional decoder are trained.

Critical result: codebook utilization with vs. without SIGReg.

Usage:
  python scripts/stage2_vq.py --checkpoint ./checkpoints/stage1_best.pt
  python scripts/stage2_vq.py --dummy  # smoke-test
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from respvoice.config import RespVoiceConfig, ModelConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer
from respvoice.preprocessing import AudioPreprocessor
from data.respvoice_datasets import RespVoiceDataset, DummyDataset


def parse_args():
    p = argparse.ArgumentParser(description="RespVoice Stage 2: VQ Tokenizer Training")
    p.add_argument("--checkpoint", default=None,
                   help="Path to Stage 1 checkpoint (optional, otherwise random init)")
    p.add_argument("--data-dir", default="./data/audio")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--lam-recon", type=float, default=0.1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = RespVoiceConfig(
        model=ModelConfig(codebook_size=args.codebook_size),
        train=TrainConfig(
            stage2_epochs=args.epochs,
            stage2_lr=args.lr,
            lam_recon=args.lam_recon,
            batch_size=args.batch_size,
        ),
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    model = RespVoiceModel(cfg.model)

    if args.checkpoint:
        print(f"Loading Stage 1 weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print("  Encoder weights loaded.")
    else:
        print("No Stage 1 checkpoint provided 鈥?using random initialization.")

    if args.dummy:
        dataset = DummyDataset(n_samples=128, n_mels=cfg.model.n_mels, T=250)
    else:
        preprocessor = AudioPreprocessor(
            sr=cfg.model.sr, n_mels=cfg.model.n_mels,
            win_ms=cfg.model.win_ms, hop_ms=cfg.model.hop_ms,
            target_sec=cfg.model.target_sec,
        )
        dataset = RespVoiceDataset(
            root=args.data_dir, preprocessor=preprocessor,
            max_samples=args.max_samples,
        )

    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size,
        shuffle=True, drop_last=True,
        num_workers=cfg.train.num_workers,
    )

    print(f"\n=== Stage 2: VQ Tokenizer ===")
    print(f"  codebook_size: {args.codebook_size}")
    print(f"  lam_recon:     {args.lam_recon}")
    print(f"  encoder:       FROZEN")
    print(f"\n  Key metric: codebook utilization (goal: > 0.5, ideally > 0.9)")

    trainer = Trainer(cfg, model)
    trainer.train_stage2(loader)
    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
