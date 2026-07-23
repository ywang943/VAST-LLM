#!/usr/bin/env python3
"""Train a matched mel-only VQ variant for VAST with full metadata.

The older RQ2 scripts either train short disposable VQs or do ablations without
saving the codebook. This script trains a reusable VQ checkpoint against a
specific encoder checkpoint, records the data roots, and reports full-token
utilization/perplexity on the sampled VQ training set.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.vq import VectorQuantizer


DEFAULT_SKIP = {
    "coswara_covid_cough",
    "coswara_covid_breathing",
    "coswara_smoker_cough",
    "coswara_smoker_breathing",
    "svd",
    "svd_full",
    "svd_healthy",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def collect_datasets(cache_root, include, skip_derived, max_samples, seed):
    cache_root = resolve_path(cache_root)
    include_set = set(include or [])
    datasets = []
    roots = []
    for meta in sorted(cache_root.glob("*/metadata.json")):
        root = meta.parent
        name = root.name
        if include_set and name not in include_set:
            continue
        if skip_derived and name in DEFAULT_SKIP:
            continue
        ds = CachedMelDataset(root=str(root), meta_file=str(meta), include_labels=False)
        if len(ds) == 0:
            continue
        datasets.append(ds)
        roots.append(str(root))

    if not datasets:
        raise RuntimeError(f"No cached mel datasets found under {cache_root}")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    if max_samples and len(combined) > max_samples:
        rng = random.Random(seed)
        idx = list(range(len(combined)))
        rng.shuffle(idx)
        combined = Subset(combined, idx[:max_samples])
    return combined, roots


def load_encoder(encoder_ckpt, device):
    ckpt = torch.load(str(encoder_ckpt), map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=True, use_csaf=True)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Encoder state mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device).eval()


@torch.no_grad()
def extract_features(encoder, loader, device):
    feats = []
    n = 0
    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        z = encoder(mel).cpu()
        feats.append(z)
        n += int(z.size(0))
        if n % 2048 == 0:
            print(f"  extracted {n} samples")
    return torch.cat(feats, dim=0)


@torch.no_grad()
def measure(vq, feats, device):
    K = vq.codebook_size
    counts = torch.zeros(K, device=device)
    vq.eval()
    for i in range(0, feats.size(0), 256):
        z = feats[i:i + 256].to(device)
        ids = vq(z)["ids"].reshape(-1)
        counts += torch.bincount(ids, minlength=K).float()
    used = int((counts > 0).sum().item())
    probs = counts / (counts.sum() + 1e-10)
    perp = float((-(probs * (probs + 1e-10).log()).sum()).exp().item())
    return used / K, perp, used, int(counts.sum().item())


def train_vq(feats, args, device):
    vq = VectorQuantizer(
        codebook_size=args.codebook_size,
        D=feats.size(-1),
        beta=0.25,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        restart_threshold=args.restart_threshold,
        restart_every=args.restart_every,
        l2_normalize=args.l2_normalize,
    ).to(device)

    opt = None
    if args.no_ema:
        opt = torch.optim.AdamW(vq.parameters(), lr=args.lr)

    n = feats.size(0)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        stats = {"loss": 0.0, "util": 0.0, "perp": 0.0, "restart": 0.0, "batches": 0}
        vq.train()
        for start in range(0, n, args.vq_batch_size):
            idx = perm[start:start + args.vq_batch_size]
            z = feats[idx].to(device)
            out = vq(z)
            if opt is not None:
                opt.zero_grad()
                out["loss"].backward()
                opt.step()
            stats["loss"] += float(out["loss"])
            stats["util"] += float(out["util"])
            stats["perp"] += float(out["perplexity"])
            stats["restart"] += float(out["n_restarted"])
            stats["batches"] += 1
        b = max(stats["batches"], 1)
        util, perp, _, _ = measure(vq, feats, device)
        print(
            f"  epoch {ep + 1:02d}/{args.epochs}: "
            f"batch_util={stats['util']/b:.3f} batch_perp={stats['perp']/b:.1f} "
            f"restart={stats['restart']/b:.1f} full_util={util:.3f} full_perp={perp:.1f}"
        )
    return vq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-ckpt", default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt")
    p.add_argument("--mel-cache-root", default="data/mel_cache")
    p.add_argument("--include", nargs="*", default=None)
    p.add_argument("--no-skip-derived", action="store_true")
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--extract-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--vq-batch-size", type=int, default=256)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--l2-normalize", action="store_true")
    p.add_argument("--restart-threshold", type=int, default=1)
    p.add_argument("--restart-every", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/vq/mel_htsat_v3_full_vq_K512_ema20k.pt")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_ckpt = resolve_path(args.encoder_ckpt)
    print(f"Device: {device}")
    print(f"Encoder: {encoder_ckpt}")

    dataset, roots = collect_datasets(
        args.mel_cache_root,
        include=args.include,
        skip_derived=not args.no_skip_derived,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.extract_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"VQ training samples: {len(dataset)} roots={len(roots)}")

    encoder = load_encoder(encoder_ckpt, device)
    print("Extracting frozen features...")
    feats = extract_features(encoder, loader, device)
    print(f"Features: {tuple(feats.shape)}")

    print(
        f"Training VQ: K={args.codebook_size} epochs={args.epochs} "
        f"ema={not args.no_ema} l2={args.l2_normalize} batch={args.vq_batch_size}"
    )
    vq = train_vq(feats, args, device)
    util, perp, used, tokens = measure(vq, feats, device)
    print(f"Final train-set utilization: {used}/{args.codebook_size} ({util:.3f})")
    print(f"Final train-set perplexity: {perp:.1f} over {tokens} tokens")

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vq_state": vq.state_dict(),
            "codebook_size": args.codebook_size,
            "D": int(feats.size(-1)),
            "steps": int(args.epochs * np.ceil(len(dataset) / args.vq_batch_size)),
            "epochs": args.epochs,
            "encoder_type": "htsat_mel",
            "encoder_checkpoint": str(Path(args.encoder_ckpt)),
            "mel_roots": roots,
            "skip_derived": not args.no_skip_derived,
            "max_samples": args.max_samples,
            "use_ema": not args.no_ema,
            "ema_decay": args.ema_decay,
            "l2_normalize": args.l2_normalize,
            "restart_threshold": args.restart_threshold,
            "restart_every": args.restart_every,
            "final_sampled_utilization": util,
            "final_sampled_perplexity": perp,
            "final_used_codes": used,
            "final_tokens": tokens,
            "seed": args.seed,
        },
        str(out_path),
    )
    print(f"Saved: {out_path}")

    summary = {
        "vq_ckpt": str(out_path),
        "utilization": util,
        "perplexity": perp,
        "used_codes": used,
        "tokens": tokens,
        "config": vars(args),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
