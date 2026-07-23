"""
Stage 2 VQ verification: load existing D768 Stage 1 checkpoint,
run 5 epochs of Stage 2 with EMA+restart, check utilization stays high.

Usage:
    python scripts/verify_vq_stage2.py
    python scripts/verify_vq_stage2.py --codebook-size 8192
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", default="./checkpoints/opera_official_icbhi_large768_full/stage1_final.pt")
    p.add_argument("--pretrain-cache", default="./data/mel_cache/pretrain_zenodo_no_icbhi")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--encoder-layers", type=int, default=6)
    p.add_argument("--encoder-heads", type=int, default=12)
    p.add_argument("--codebook-size", type=int, default=2048)
    p.add_argument("--no-ema", action="store_true", help="Disable EMA (reproduce collapse)")
    p.add_argument("--checkpoint-dir", default="./checkpoints/vq_verify_d768")
    args = p.parse_args()

    use_ema = not args.no_ema
    print(f"=== Stage 2 VQ Verification ===")
    print(f"  D={args.dim}  codebook={args.codebook_size}  EMA={'ON' if use_ema else 'OFF (collapse mode)'}  epochs={args.epochs}")

    cfg = RespVoiceConfig(
        model=ModelConfig(
            D=args.dim,
            codebook_size=args.codebook_size,
            encoder_layers=args.encoder_layers,
            encoder_heads=args.encoder_heads,
            predictor_layers=2,
            n_sigreg_slices=64,
            vq_use_ema=use_ema,
            vq_ema_decay=0.99,
            vq_restart_threshold=1,
            vq_restart_every=1,
        ),
        train=TrainConfig(
            stage2_epochs=args.epochs,
            stage2_lr=1e-4,
            lam_recon=0.05,
            batch_size=args.batch_size,
            warmup_ratio=0.05,
            num_workers=0,
        ),
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.checkpoint_dir,
    )

    model = RespVoiceModel(cfg.model)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load Stage 1 checkpoint (encoder + predictor only, skip VQ in case of size mismatch)
    ckpt_path = args.stage1_ckpt
    if Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        full_state = state["model_state"]
        # Filter to only encoder and predictor weights (skip vq.* to avoid size mismatch)
        enc_state = {k: v for k, v in full_state.items() if not k.startswith("vq.") and not k.startswith("decoder.")}
        missing, unexpected = model.load_state_dict(enc_state, strict=False)
        vq_skipped = [k for k in full_state if k.startswith("vq.")]
        print(f"  Loaded Stage 1 from: {ckpt_path}")
        print(f"    encoder keys loaded: {len(enc_state)}, VQ keys skipped: {len(vq_skipped)}")
    else:
        print(f"  WARNING: Stage 1 checkpoint not found at {ckpt_path}, using random init")

    # Load pretraining data
    cache = Path(args.pretrain_cache)
    pretrain_ds = CachedMelDataset(
        root=str(cache),
        meta_file=str(cache / "metadata.json"),
        include_labels=False,
    )
    loader = DataLoader(pretrain_ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)
    print(f"  Pretraining windows: {len(pretrain_ds)}  batches/epoch: {len(loader)}")
    print()

    trainer = Trainer(cfg, model)
    trainer.train_stage2(loader)

    print("\n=== Summary ===")
    print("If EMA+restart is working: final util > 0.3, perp > 200")
    print("Old behavior (no fix): util collapses to < 0.01 by epoch 2")


if __name__ == "__main__":
    main()
