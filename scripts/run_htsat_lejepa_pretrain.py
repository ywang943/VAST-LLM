"""
HTS-AT + CSAF full-parameter LeJEPA pretraining.

This is the key scaling experiment: instead of a custom lightweight encoder,
use the full HTS-AT backbone (31M params, 4 Swin stages) initialized from
OPERA-CT, plus TPA-CSAF (9.9M params), and do LeJEPA pretraining on all
available data with the A800.

Goal: beat OPERA-CT fine-tune (0.957) by combining:
  - HTS-AT multi-scale backbone (initialized from OPERA-CT COLA weights)
  - TPA-CSAF cross-scale attention fusion
  - LeJEPA + SIGReg self-supervised objective
  - Larger pretraining dataset (~40K windows)

Architecture:
  Input → HTS-AT (4 Swin stages, unfrozen) → CSAF fusion → z_cont (B,64,768)
  LeJEPA predictor predicts masked z_cont from visible z_cont
  SIGReg regularizes z_cont toward N(0,I)

Usage:
  python scripts/run_htsat_lejepa_pretrain.py
  python scripts/run_htsat_lejepa_pretrain.py --epochs 50 --batch-size 64
  python scripts/run_htsat_lejepa_pretrain.py --init scratch --backbone-lr 5e-5
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.jepa import JEPAPredictor, jepa_loss
from respvoice.sigreg import SIGReg


class HTSATLeJEPAModel(nn.Module):
    """HTS-AT + CSAF encoder with LeJEPA + SIGReg pretraining."""

    def __init__(self, D=768, mask_ratio=0.60, freeze_backbone=False,
                 init="opera"):
        super().__init__()
        self.D = D
        self.mask_ratio = mask_ratio

        self.encoder = build_htsat_encoder(
            ckpt_path=("checkpoints/opera_cache/encoder-operaCT.ckpt"
                       if init == "opera" else None),
            freeze_backbone=freeze_backbone,
            use_csaf=True,
        )

        self.predictor = JEPAPredictor(D=D, depth=2, num_heads=12)
        self.sigreg = SIGReg(n_slices=256)

        max_seq = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, D))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, mel, lam_sig=0.02):
        B = mel.size(0)
        z_cont = self.encoder(mel)  # (B, 64, 768)
        L = z_cont.size(1)
        pos = self.pos_embed[:, :L, :].expand(B, -1, -1)

        # JEPA loss: predict masked from visible
        n_mask = max(1, int(L * self.mask_ratio))
        perm = torch.randperm(L, device=z_cont.device)
        mask_idx = perm[:n_mask]
        vis_idx = perm[n_mask:]

        target = z_cont[:, mask_idx].detach()
        vis_z = z_cont[:, vis_idx]
        mask_pos = pos[:, mask_idx]

        pred = self.predictor(vis_z, mask_pos)
        loss_jepa = F.mse_loss(pred, target)

        # SIGReg
        loss_sigreg = self.sigreg(z_cont)

        loss = loss_jepa + lam_sig * loss_sigreg
        return {
            "loss": loss,
            "loss_jepa": loss_jepa,
            "loss_sigreg": loss_sigreg,
            "z_cont": z_cont,
        }


def collect_mel_caches(cache_names=None):
    cache_dir = Path("data/mel_cache")
    selected = set(cache_names) if cache_names else None
    datasets = []
    total = 0
    for d in sorted(cache_dir.iterdir()):
        if selected is not None and d.name not in selected:
            continue
        meta = d / "metadata.json"
        if meta.exists():
            ds = CachedMelDataset(root=str(d), meta_file=str(meta), include_labels=False)
            if len(ds) > 0:
                datasets.append(ds)
                total += len(ds)
                print(f"  {d.name}: {len(ds)} windows")
    return datasets, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", choices=("opera", "scratch"), default="opera",
                   help="Initialize HTS-AT from OPERA-CT or random weights")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pretrain-caches", nargs="+", default=None,
        help="Names under data/mel_cache to use. Omit only for exploratory all-cache runs.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5,
                   help="Lower LR for pretrained HTS-AT backbone")
    p.add_argument("--lam-sig", type=float, default=0.02)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--checkpoint-dir", default="./checkpoints/htsat_lejepa_pretrain")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from", default="",
                   help="Load model weights from an arbitrary checkpoint but start a fresh schedule.")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective_bs = batch_size * grad_accum)")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("HTS-AT + CSAF LeJEPA PRETRAINING")
    print(f"Initialization: {args.init}  seed={args.seed}")
    print("=" * 60)

    # Data
    print("\n=== Data ===")
    mel_datasets, total = collect_mel_caches(args.pretrain_caches)
    if not mel_datasets:
        print("No mel caches found!"); return
    combined = ConcatDataset(mel_datasets) if len(mel_datasets) > 1 else mel_datasets[0]
    print(f"  Total: {len(combined)} windows")

    loader = DataLoader(
        combined, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Batches/epoch: {len(loader)}")

    # Model
    print("\n=== Model ===")
    model = HTSATLeJEPAModel(
        D=768, freeze_backbone=False, init=args.init
    ).to(device)

    backbone_params = list(model.encoder.htsat.parameters())
    csaf_params = list(model.encoder.csaf.parameters()) + \
                  list(model.encoder.pool1.parameters()) + \
                  list(model.encoder.pool2.parameters())
    predictor_params = list(model.predictor.parameters())
    other_params = [model.pos_embed]

    n_backbone = sum(p.numel() for p in backbone_params)
    n_csaf = sum(p.numel() for p in csaf_params)
    n_pred = sum(p.numel() for p in predictor_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  HTS-AT backbone: {n_backbone:,} params "
          f"(init={args.init}, lr={args.backbone_lr})")
    print(f"  CSAF + poolers:  {n_csaf:,} params (lr={args.lr})")
    print(f"  Predictor:       {n_pred:,} params (lr={args.lr})")
    print(f"  Total:           {n_total:,} params")

    # Differential LR: lower for pretrained backbone, higher for new modules
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": csaf_params, "lr": args.lr},
        {"params": predictor_params, "lr": args.lr},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=0.05)

    total_steps = args.epochs * len(loader)
    warmup_steps = args.warmup_epochs * len(loader)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    # Resume
    start_epoch = 1
    best_loss = float("inf")
    ckpt_path = Path(args.checkpoint_dir) / "htsat_lejepa_best.pt"
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  Loaded weights from {args.resume_from}; starting fresh schedule")
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        saved_init = ckpt.get("initialization")
        if saved_init is not None and saved_init != args.init:
            raise ValueError(
                f"Checkpoint initialization is {saved_init!r}, but --init is "
                f"{args.init!r}. Use the matching checkpoint directory."
            )
        model.load_state_dict(ckpt["model_state"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch-1}, best_loss={best_loss:.4f}")

    # Train
    grad_accum = args.grad_accum
    eff_bs = args.batch_size * grad_accum
    print(f"\n=== Training: {args.epochs} epochs, {total_steps} steps ===")
    print(f"  Effective batch size: {args.batch_size} x {grad_accum} accum = {eff_bs}")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        stats = {"loss": 0, "jepa": 0, "sigreg": 0, "n": 0}
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"Ep{epoch}/{args.epochs}", leave=False)):
            mel = batch["mel"].to(device)
            out = model(mel, lam_sig=args.lam_sig)
            # Scale loss by accum steps so effective gradient = mean over accum batches
            (out["loss"] / grad_accum).backward()

            B = mel.size(0)
            stats["loss"] += out["loss"].item() * B
            stats["jepa"] += out["loss_jepa"].item() * B
            stats["sigreg"] += out["loss_sigreg"].item() * B
            stats["n"] += B

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        n = stats["n"]
        avg_loss = stats["loss"] / n
        avg_jepa = stats["jepa"] / n
        avg_sig = stats["sigreg"] / n
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"  [Ep{epoch}] loss={avg_loss:.4f} jepa={avg_jepa:.4f} "
              f"sigreg={avg_sig:.4f} lr_bb={lr_now:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "initialization": args.init,
                "seed": args.seed,
            }, str(ckpt_path))
            print(f"    -> saved best (loss={best_loss:.4f})")

        if epoch % 10 == 0:
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "initialization": args.init,
                "seed": args.seed,
            }, str(Path(args.checkpoint_dir) / f"htsat_lejepa_ep{epoch}.pt"))

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Save summary
    summary = {
        "backbone": f"HTS-AT ({args.init} init, unfrozen)",
        "initialization": args.init,
        "seed": args.seed,
        "csaf": "TPA-CSAF (trainable)",
        "objective": "LeJEPA + SIGReg",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_backbone": args.backbone_lr,
        "lr_csaf": args.lr,
        "total_samples": len(combined),
        "pretrain_caches": args.pretrain_caches or "all",
        "total_params": n_total,
        "best_loss": best_loss,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    with open(Path(args.checkpoint_dir) / "pretrain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {args.checkpoint_dir}/pretrain_summary.json")


if __name__ == "__main__":
    main()
