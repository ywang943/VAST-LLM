"""
Scaled-up LeJEPA pretraining: larger backbone + larger dataset.

Previous: D=128, 2-layer Transformer, 34K windows, RTX 5060 Ti 8GB
Now:      D=768, 12-layer Transformer, ~30K+ windows, A800 80GB

Scaling changes:
  1. Backbone: 6→12 Transformer layers, D=128→768, 8→12 heads
  2. Data: ICBHI + CoughVID + LibriSpeech mel caches (all available)
  3. Batch size: 32→128 (A800 80GB allows much larger batches)
  4. Training: 100 epochs with cosine LR + warmup

Usage:
  python scripts/run_scaled_pretrain.py
  python scripts/run_scaled_pretrain.py --dim 1024 --layers 16 --heads 16  # even larger
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer
from data.respvoice_datasets import CachedMelDataset, RespVoiceDataset
from respvoice.preprocessing import AudioPreprocessor


def collect_mel_caches(cache_dir: str = "data/mel_cache"):
    """Find all available mel cache directories with metadata.json."""
    cache_dir = Path(cache_dir)
    datasets = []
    total = 0
    for d in sorted(cache_dir.iterdir()):
        meta = d / "metadata.json"
        if meta.exists():
            ds = CachedMelDataset(root=str(d), meta_file=str(meta), include_labels=False)
            if len(ds) > 0:
                datasets.append(ds)
                total += len(ds)
                print(f"  {d.name}: {len(ds)} windows")
    return datasets, total


def collect_audio_dirs(audio_dir: str = "data/audio"):
    """Find all wav directories for on-the-fly preprocessing."""
    audio_dir = Path(audio_dir)
    dirs = []
    for d in sorted(audio_dir.iterdir()):
        if d.is_dir():
            wav_count = len(list(d.glob("*.wav")))
            if wav_count > 0:
                dirs.append((str(d), wav_count))
                print(f"  {d.name}: {wav_count} wav files")
    return dirs


def main():
    p = argparse.ArgumentParser(description="Scaled-up LeJEPA pretraining")
    p.add_argument("--dim", type=int, default=768, help="Embedding dimension")
    p.add_argument("--layers", type=int, default=12, help="Transformer layers")
    p.add_argument("--heads", type=int, default=12, help="Attention heads")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lam-sig", type=float, default=0.02)
    p.add_argument("--checkpoint-dir", default="./checkpoints/scaled_pretrain")
    p.add_argument("--use-audio-dirs", action="store_true",
                   help="Also load raw wav files (slower, on-the-fly preprocessing)")
    args = p.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("SCALED-UP LeJEPA PRETRAINING")
    print("=" * 60)
    print(f"  Backbone: D={args.dim}, {args.layers} layers, {args.heads} heads")
    n_params_approx = args.dim * args.dim * 4 * args.layers * 2 / 1e6
    print(f"  ~{n_params_approx:.0f}M encoder parameters (approx)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print()

    # Collect all data
    print("=== Collecting mel caches ===")
    mel_datasets, mel_total = collect_mel_caches()

    audio_datasets = []
    if args.use_audio_dirs:
        print("\n=== Collecting audio directories ===")
        preprocessor = AudioPreprocessor(sr=16000, n_mels=64, win_ms=64.0, hop_ms=32.0, target_sec=8.0)
        audio_dirs = collect_audio_dirs()
        for audio_dir, count in audio_dirs:
            ds = RespVoiceDataset(root=audio_dir, preprocessor=preprocessor)
            if len(ds) > 0:
                audio_datasets.append(ds)

    all_datasets = mel_datasets + audio_datasets
    if not all_datasets:
        print("ERROR: No data found! Run data preparation scripts first.")
        return

    combined = ConcatDataset(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
    print(f"\n  Total training samples: {len(combined)}")

    # Configure model
    cfg = RespVoiceConfig(
        model=ModelConfig(
            D=args.dim,
            backbone="custom",
            encoder_layers=args.layers,
            encoder_heads=args.heads,
            n_mels=64,
            patch_h=8,
            patch_w=8,
            mask_ratio=0.60,
            n_sigreg_slices=256,
            codebook_size=8192,
        ),
        train=TrainConfig(
            stage1_epochs=args.epochs,
            stage1_lr=args.lr,
            lam_sig=args.lam_sig,
            batch_size=args.batch_size,
            weight_decay=0.05,
            warmup_ratio=0.1,
            num_workers=4,
        ),
        checkpoint_dir=args.checkpoint_dir,
    )

    loader = DataLoader(
        combined,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.train.num_workers > 0 else False,
    )

    print(f"  Batches per epoch: {len(loader)}")
    print(f"  Total steps: {len(loader) * args.epochs}")

    # Build model
    model = RespVoiceModel(cfg.model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    print()

    # Train
    trainer = Trainer(cfg, model)
    t0 = time.time()
    trainer.train_stage1(loader)
    elapsed = time.time() - t0

    print(f"\nPretraining complete in {elapsed/60:.1f} minutes")
    print(f"Checkpoint: {args.checkpoint_dir}")

    # Save config
    summary = {
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "total_samples": len(combined),
        "total_params": total_params,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    with open(Path(args.checkpoint_dir) / "pretrain_config.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
