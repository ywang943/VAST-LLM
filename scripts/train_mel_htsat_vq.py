"""
Train a mel-only VQ codebook on frozen HTS-AT LeJEPA encoder outputs.

This is the mel-only counterpart to the dual-input SpeechGPT VQ:
  cached mel -> frozen HTS-AT/CSAF encoder -> z_cont (B,64,768)
  -> K-way EMA/L2 VQ -> discrete token IDs.

The saved checkpoint records the encoder checkpoint and mel roots so RQ3 can
avoid mixing a VQ with the wrong encoder family.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "opera_src"))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.vq import VectorQuantizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKIP = {
    # Derived Coswara task subsets; coswara_covid_all covers the full usable set.
    "coswara_covid_cough",
    "coswara_covid_breathing",
    "coswara_smoker_cough",
    "coswara_smoker_breathing",
    # Derived SVD subsets; svd_all is the broad mel-only pool.
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


def collect_mel_datasets(cache_root, include, skip_derived):
    cache_root = Path(cache_root)
    datasets = []
    roots = []
    total = 0
    include_set = set(include or [])
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
        total += len(ds)
    if not datasets:
        raise RuntimeError(f"No mel datasets found under {cache_root}")
    return datasets, roots, total


def load_encoder(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=True, use_csaf=True)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"Encoder loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-ckpt", default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt")
    p.add_argument("--mel-cache-root", default="data/mel_cache")
    p.add_argument("--include", nargs="*", default=None,
                   help="Optional mel_cache directory names to include.")
    p.add_argument("--no-skip-derived", action="store_true",
                   help="Use all mel_cache directories, including derived task subsets.")
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-l2-normalize", action="store_true")
    p.add_argument("--restart-threshold", type=int, default=1)
    p.add_argument("--restart-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--out", default="checkpoints/vq/mel_htsat_v3_full_vq_K512.pt")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder_ckpt}")

    print("\nCollecting mel caches...")
    datasets, roots, total = collect_mel_datasets(
        ROOT / args.mel_cache_root,
        args.include,
        skip_derived=not args.no_skip_derived,
    )
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"  roots={len(roots)} total_samples={total} batches/epoch={len(loader)}")
    for r in roots:
        print(f"  - {r}")

    print("\nLoading frozen mel-only encoder...")
    encoder = load_encoder(args.encoder_ckpt, device)

    vq = VectorQuantizer(
        codebook_size=args.codebook_size,
        D=768,
        beta=0.25,
        use_ema=True,
        ema_decay=0.99,
        restart_threshold=args.restart_threshold,
        restart_every=args.restart_every,
        l2_normalize=not args.no_l2_normalize,
    ).to(device)

    print(
        f"\nTraining mel-only VQ: K={args.codebook_size}, D=768, steps={args.steps}, "
        f"l2_normalize={not args.no_l2_normalize}, "
        f"restart_threshold={args.restart_threshold}, restart_every={args.restart_every}"
    )
    step = 0
    stats = {"loss": 0.0, "util": 0.0, "perp": 0.0, "restart": 0.0, "n": 0}
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            mel = batch["mel"].to(device, non_blocking=True)
            with torch.no_grad():
                z_cont = encoder(mel)
                vq.train()
                out = vq(z_cont)
            step += 1
            stats["loss"] += float(out["loss"])
            stats["util"] += float(out["util"])
            stats["perp"] += float(out["perplexity"])
            stats["restart"] += float(out["n_restarted"])
            stats["n"] += 1

            if step % args.log_every == 0:
                n = max(stats["n"], 1)
                print(
                    f"  step {step:05d}: loss={stats['loss']/n:.4f} "
                    f"util={stats['util']/n:.3f} perp={stats['perp']/n:.1f} "
                    f"restart={stats['restart']/n:.1f}"
                )
                stats = {"loss": 0.0, "util": 0.0, "perp": 0.0, "restart": 0.0, "n": 0}

    print("\nComputing final utilization on up to 100 batches...")
    counts = torch.zeros(args.codebook_size, device=device)
    seen = 0
    vq.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 100:
                break
            mel = batch["mel"].to(device, non_blocking=True)
            ids = vq(encoder(mel))["ids"].reshape(-1)
            counts += torch.bincount(ids, minlength=args.codebook_size).float()
            seen += mel.size(0)
    used = int((counts > 0).sum().item())
    probs = counts / (counts.sum() + 1e-10)
    perplexity = float((-(probs * (probs + 1e-10).log()).sum()).exp().item())
    print(f"Final sampled utilization: {used}/{args.codebook_size} ({used/args.codebook_size:.3f})")
    print(f"Final sampled perplexity: {perplexity:.1f} over {seen} samples")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vq_state": vq.state_dict(),
            "codebook_size": args.codebook_size,
            "D": 768,
            "steps": args.steps,
            "encoder_type": "htsat_mel",
            "encoder_checkpoint": args.encoder_ckpt,
            "mel_roots": roots,
            "skip_derived": not args.no_skip_derived,
            "l2_normalize": not args.no_l2_normalize,
            "restart_threshold": args.restart_threshold,
            "restart_every": args.restart_every,
            "final_sampled_utilization": used / args.codebook_size,
            "final_sampled_perplexity": perplexity,
            "seed": args.seed,
        },
        str(out_path),
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
