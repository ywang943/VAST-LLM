"""
Train VQ codebook on frozen dual-input encoder outputs.

Freezes the pretrained encoder, runs all data through it to get z_cont,
then trains the VQ codebook using EMA updates + dead code restart.

Output: a trained VQ module that converts z_cont (B,64,768) → token IDs (B,64)
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from respvoice.dual_input_encoder import build_dual_input_encoder
from respvoice.vq import VectorQuantizer
from scripts.run_dual_lejepa_pretrain import (
    WavOnlyDataset, DualInputDataset, collate_dual, collect_datasets,
)


def load_encoder(checkpoint, device):
    """Load pretrained dual-input encoder, freeze all params."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k.replace("encoder.", "", 1): v
             for k, v in ckpt["model_state"].items()
             if k.startswith("encoder.")}
    encoder = build_dual_input_encoder(
        ckpt_path=None, freeze_backbone=True, freeze_cnn=True, use_csaf=True,
    )
    encoder.load_state_dict(state, strict=False)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Pretrained dual LeJEPA checkpoint")
    p.add_argument("--codebook-size", type=int, default=8192)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="checkpoints/vq/vq_codebook.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen encoder
    print("Loading frozen encoder...")
    encoder = load_encoder(args.checkpoint, device)

    # Data
    print("\nLoading data...")
    datasets, total = collect_datasets(None)
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    loader = DataLoader(
        combined, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
        collate_fn=collate_dual,
    )
    print(f"  Total: {total} samples, {len(loader)} batches")

    # VQ module
    vq = VectorQuantizer(
        codebook_size=args.codebook_size,
        D=768,
        beta=0.25,
        use_ema=True,
        ema_decay=0.99,
        l2_normalize=True,
    ).to(device)

    print(f"\nVQ: K={args.codebook_size}, D=768, L2-normalized")
    print(f"Training for {args.steps} steps...")

    step = 0
    stats = {"loss": 0, "util": 0, "perp": 0, "n": 0}

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            mel = batch["mel"].to(device)
            wav = batch["wav"].to(device) if batch["wav"] is not None else None

            with torch.no_grad():
                z_cont = encoder(mel, wav)  # (B, 64, 768)

            vq.train()
            vq_out = vq(z_cont)

            stats["loss"] += vq_out["loss"].item()
            stats["util"] += vq_out["util"]
            stats["perp"] += vq_out["perplexity"]
            stats["n"] += 1

            step += 1
            if step % 500 == 0:
                n = stats["n"]
                print(f"  Step {step}: loss={stats['loss']/n:.4f} "
                      f"util={stats['util']/n:.3f} "
                      f"perp={stats['perp']/n:.0f} "
                      f"restart={vq_out['n_restarted']}")
                stats = {"loss": 0, "util": 0, "perp": 0, "n": 0}

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vq_state": vq.state_dict(),
        "codebook_size": args.codebook_size,
        "D": 768,
        "steps": args.steps,
        "encoder_checkpoint": args.checkpoint,
    }, str(out_path))
    print(f"\nSaved: {out_path}")

    # Final stats
    vq.eval()
    all_ids = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 50:
                break
            mel = batch["mel"].to(device)
            wav = batch["wav"].to(device) if batch["wav"] is not None else None
            z = encoder(mel, wav)
            out = vq(z)
            all_ids.append(out["ids"].cpu())
    all_ids = torch.cat(all_ids, dim=0)  # (N, 64)
    unique = all_ids.unique().numel()
    print(f"\nFinal codebook utilization: {unique}/{args.codebook_size} "
          f"({unique/args.codebook_size*100:.1f}%)")


if __name__ == "__main__":
    main()
