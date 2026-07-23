"""
Dual-Input LeJEPA Pretraining: Mel + HuBERT CNN → HTS-AT + CSAF.

Uses the DualInputHTSATEncoder which accepts both mel spectrograms and
raw waveforms. For datasets with wav cache, both inputs are used.
Every sample is required to provide both a mel spectrogram and a waveform.

Data: Coswara + CoughVID + LibriSpeech + SVD (all recordings)
Two versions: ① scratch init  ② OPERA-CT init
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from respvoice.dual_input_encoder import build_dual_input_encoder
from respvoice.jepa import JEPAPredictor
from respvoice.sigreg import SIGReg

SR = 16000
WAV_TARGET_LEN = int(SR * 8.0)


# ── Dataset ──────────────────────────────────────────────────────────────────

class DualInputDataset(Dataset):
    """Loads strictly paired mel (.pt) and waveform (.npy) files."""

    def __init__(self, mel_root, wav_root=None, mel_meta=None):
        self.mel_root = Path(mel_root)
        self.wav_root = Path(wav_root)

        meta_path = mel_meta or (self.mel_root / "metadata.json")
        with open(meta_path) as f:
            raw = json.load(f)
        self.samples = raw.get("samples", raw if isinstance(raw, list) else [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        mel = torch.load(self.mel_root / item["path"], map_location="cpu")

        npy_name = item["path"].replace(".pt", ".npy")
        npy_path = self.wav_root / npy_name
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing paired waveform: {npy_path}")

        wav_np = np.load(str(npy_path)).astype(np.float32)
        wav_np = (wav_np - wav_np.mean()) / (wav_np.std() + 1e-8)
        if len(wav_np) >= WAV_TARGET_LEN:
            wav_np = wav_np[:WAV_TARGET_LEN]
        else:
            wav_np = np.pad(wav_np, (0, WAV_TARGET_LEN - len(wav_np)))
        wav = torch.from_numpy(wav_np)

        return {"mel": mel, "wav": wav}


class WavOnlyDataset(Dataset):
    """For SVD full which has wav but no pre-built mel - generate mel on the fly."""

    def __init__(self, wav_root, meta_path):
        self.wav_root = Path(wav_root)
        with open(meta_path) as f:
            raw = json.load(f)
        self.samples = raw.get("samples", raw if isinstance(raw, list) else [])
        from respvoice.preprocessing import AudioPreprocessor
        self.preprocessor = AudioPreprocessor(sr=SR, target_sec=8.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        npy_path = self.wav_root / item["path"]
        wav_np = np.load(str(npy_path)).astype(np.float32)

        # Generate mel from wav
        mel = self.preprocessor.to_mel(wav_np)  # (1, 64, T)

        # Normalize wav
        wav_norm = (wav_np - wav_np.mean()) / (wav_np.std() + 1e-8)
        if len(wav_norm) >= WAV_TARGET_LEN:
            wav_norm = wav_norm[:WAV_TARGET_LEN]
        else:
            wav_norm = np.pad(wav_norm, (0, WAV_TARGET_LEN - len(wav_norm)))

        return {"mel": mel, "wav": torch.from_numpy(wav_norm)}


def collate_dual(batch):
    mels = torch.stack([b["mel"] for b in batch])
    if any(b["wav"] is None for b in batch):
        raise ValueError("Dual-input pretraining received a sample without waveform")
    wavs = torch.stack([b["wav"] for b in batch])
    return {"mel": mels, "wav": wavs}


def collect_datasets(cache_names):
    """Build datasets in which every sample has both mel and waveform inputs."""
    datasets = []
    total = 0
    selected = set(cache_names) if cache_names else {
        "coswara_hf", "coughvid_zenodo", "librispeech_100h", "svd_all",
        "opera_icbhi_disease", "opera_copd", "opera_kauh",
    }

    paired_datasets = {
        "svd_all": (
            "data/mel_cache/svd_all",
            "data/wav_cache/svd_all",
        ),
        "coswara_hf": (
            "data/mel_cache/coswara_hf",
            "data/wav_cache/coswara_hf",
        ),
        "coughvid_zenodo": (
            "data/mel_cache/coughvid_zenodo",
            "data/wav_cache/coughvid_zenodo_aligned",
        ),
        "librispeech_100h": (
            "data/mel_cache/librispeech_100h",
            "data/wav_cache/librispeech_100h_aligned",
        ),
        "opera_icbhi_disease": (
            "data/mel_cache/opera_icbhi_disease",
            "data/wav_cache/opera_icbhi_disease",
        ),
        "opera_copd": (
            "data/mel_cache/opera_copd",
            "data/wav_cache/opera_copd",
        ),
        "opera_kauh": (
            "data/mel_cache/opera_kauh",
            "data/wav_cache/opera_kauh",
        ),
    }

    for name in selected:
        if name in paired_datasets:
            mel_dir, wav_dir = paired_datasets[name]
            mel_meta = Path(mel_dir) / "metadata.json"
            if mel_meta.exists():
                ds = DualInputDataset(mel_dir, wav_root=wav_dir)
                if len(ds) > 0:
                    datasets.append(ds)
                    total += len(ds)
                    print(f"  {name}: {len(ds)} (dual, cached mel+wav)")

    return datasets, total


# ── Model ────────────────────────────────────────────────────────────────────

class DualLeJEPAModel(nn.Module):
    def __init__(self, D=768, mask_ratio=0.60, init="opera", freeze_cnn=True):
        super().__init__()
        self.D = D
        self.mask_ratio = mask_ratio

        self.encoder = build_dual_input_encoder(
            ckpt_path=("checkpoints/opera_cache/encoder-operaCT.ckpt"
                       if init == "opera" else None),
            D=D, freeze_backbone=False, freeze_cnn=freeze_cnn, use_csaf=True,
        )

        self.predictor = JEPAPredictor(D=D, depth=2, num_heads=12)
        self.sigreg = SIGReg(n_slices=256)

        max_seq = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, D))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, mel, waveform=None, lam_sig=0.02):
        B = mel.size(0)
        z_cont = self.encoder(mel, waveform)  # (B, 64, 768)
        L = z_cont.size(1)
        pos = self.pos_embed[:, :L, :].expand(B, -1, -1)

        n_mask = max(1, int(L * self.mask_ratio))
        perm = torch.randperm(L, device=z_cont.device)
        mask_idx = perm[:n_mask]
        vis_idx = perm[n_mask:]

        target = z_cont[:, mask_idx].detach()
        vis_z = z_cont[:, vis_idx]
        mask_pos = pos[:, mask_idx]

        pred = self.predictor(vis_z, mask_pos)
        loss_jepa = F.mse_loss(pred, target)
        loss_sigreg = self.sigreg(z_cont)

        loss = loss_jepa + lam_sig * loss_sigreg
        return {"loss": loss, "loss_jepa": loss_jepa, "loss_sigreg": loss_sigreg}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", choices=("opera", "scratch"), default="opera")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain-caches", nargs="+",
                   default=["coswara_hf", "coughvid_zenodo", "librispeech_100h",
                            "svd_all", "opera_icbhi_disease", "opera_copd", "opera_kauh"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--lam-sig", type=float, default=0.005)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--min-epochs", type=int, default=60)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-rel-delta", type=float, default=1e-3)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--freeze-cnn", action="store_true", default=True)
    p.add_argument("--checkpoint-dir", default="./checkpoints/dual_lejepa_pretrain")
    p.add_argument("--resume", action="store_true")
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
    print("DUAL-INPUT LeJEPA PRETRAINING (Mel + HuBERT CNN)")
    print(f"Init: {args.init}  seed={args.seed}")
    print("=" * 60)

    # Data
    print("\n=== Data ===")
    datasets, total = collect_datasets(args.pretrain_caches)
    if not datasets:
        print("No datasets found!"); return
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"  Total: {len(combined)} samples")

    loader = DataLoader(
        combined, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
        persistent_workers=True, collate_fn=collate_dual,
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Batches/epoch: {len(loader)}")

    # Model
    print("\n=== Model ===")
    model = DualLeJEPAModel(
        D=768, init=args.init, freeze_cnn=args.freeze_cnn
    ).to(device)

    backbone_params = list(model.encoder.htsat.parameters())
    csaf_params = (list(model.encoder.csaf.parameters()) +
                   list(model.encoder.pool1.parameters()) +
                   list(model.encoder.pool2.parameters()))
    cnn_adapter_params = (list(model.encoder.wav_adapter.parameters()) +
                          list(model.encoder.fusion.parameters()))
    predictor_params = list(model.predictor.parameters())
    other_params = [model.pos_embed]

    # Don't include frozen CNN params
    n_backbone = sum(p.numel() for p in backbone_params)
    n_csaf = sum(p.numel() for p in csaf_params)
    n_adapter = sum(p.numel() for p in cnn_adapter_params)
    n_pred = sum(p.numel() for p in predictor_params)
    n_cnn = sum(p.numel() for p in model.encoder.hubert_cnn.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  HTS-AT backbone:   {n_backbone:,} (init={args.init}, lr={args.backbone_lr})")
    print(f"  CSAF + poolers:    {n_csaf:,} (lr={args.lr})")
    print(f"  HuBERT CNN:        {n_cnn:,} ({'frozen' if args.freeze_cnn else 'trainable'})")
    print(f"  Wav adapter+fusion:{n_adapter:,} (lr={args.lr})")
    print(f"  Predictor:         {n_pred:,} (lr={args.lr})")
    print(f"  Total:             {n_total:,} ({n_trainable:,} trainable)")

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": csaf_params, "lr": args.lr},
        {"params": cnn_adapter_params, "lr": args.lr},
        {"params": predictor_params, "lr": args.lr},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=0.05)

    grad_accum = args.grad_accum
    updates_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = args.warmup_epochs * updates_per_epoch
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    start_epoch = 1
    best_loss = float("inf")
    convergence_best = float("inf")
    epochs_without_improvement = 0
    ckpt_path = Path(args.checkpoint_dir) / "dual_lejepa_best.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch-1}, best_loss={best_loss:.4f}")

    # Train
    eff_bs = args.batch_size * grad_accum
    print(f"\n=== Training: {args.epochs} epochs, {total_steps} steps ===")
    print(f"  Effective batch size: {args.batch_size} x {grad_accum} = {eff_bs}")
    t0 = time.time()

    last_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        model.train()
        stats = {"loss": 0, "jepa": 0, "sigreg": 0, "n": 0}
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader,
                                          desc=f"Ep{epoch}/{args.epochs}",
                                          leave=False)):
            mel = batch["mel"].to(device)
            wav = batch["wav"].to(device) if batch["wav"] is not None else None

            out = model(mel, wav, lam_sig=args.lam_sig)
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

        meaningful_target = convergence_best * (1.0 - args.early_stop_rel_delta)
        if avg_loss < meaningful_target:
            convergence_best = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0:
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "initialization": args.init,
            }, str(Path(args.checkpoint_dir) / f"dual_lejepa_ep{epoch}.pt"))

        if (epoch >= args.min_epochs and
                epochs_without_improvement >= args.early_stop_patience):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: no >= "
                f"{args.early_stop_rel_delta:.2%} relative loss improvement for "
                f"{epochs_without_improvement} epochs"
            )
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    summary = {
        "model": "DualInputHTSATEncoder (Mel + HuBERT CNN)",
        "initialization": args.init,
        "seed": args.seed,
        "objective": "LeJEPA + SIGReg",
        "max_epochs": args.epochs,
        "completed_epochs": last_epoch,
        "stopped_early": stopped_early,
        "min_epochs": args.min_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_rel_delta": args.early_stop_rel_delta,
        "batch_size": args.batch_size,
        "grad_accum": grad_accum,
        "lr_backbone": args.backbone_lr,
        "lr_csaf": args.lr,
        "freeze_cnn": args.freeze_cnn,
        "total_samples": len(combined),
        "pretrain_caches": args.pretrain_caches,
        "total_params": n_total,
        "trainable_params": n_trainable,
        "best_loss": best_loss,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    with open(Path(args.checkpoint_dir) / "pretrain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {args.checkpoint_dir}/pretrain_summary.json")


if __name__ == "__main__":
    main()
