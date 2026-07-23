#!/usr/bin/env python3
"""Verify that a saved VQ is matched to its encoder and remeasure usage.

This is intentionally separate from the old RQ2 baseline script, which trains a
fresh short-run VQ. For paper Table 3 we need the codebook actually used by the
VAST audio-token pipeline, with explicit metadata checks.
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


def load_encoder(encoder_ckpt, device):
    ckpt = torch.load(str(encoder_ckpt), map_location="cpu", weights_only=False)
    if "model_state" not in ckpt:
        raise RuntimeError(f"{encoder_ckpt} does not look like a VAST encoder checkpoint")
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


def load_vq(vq_ckpt, encoder_ckpt, device):
    data = torch.load(str(vq_ckpt), map_location="cpu", weights_only=False)

    vq_encoder_type = data.get("encoder_type")
    if vq_encoder_type != "htsat_mel":
        raise RuntimeError(
            f"VQ encoder_type mismatch: expected htsat_mel, got {vq_encoder_type}"
        )

    recorded_encoder = data.get("encoder_checkpoint")
    if not recorded_encoder:
        raise RuntimeError("VQ checkpoint has no encoder_checkpoint metadata")
    if resolve_path(recorded_encoder) != resolve_path(encoder_ckpt):
        raise RuntimeError(
            "VQ/encoder checkpoint mismatch: "
            f"VQ records {recorded_encoder}, requested {encoder_ckpt}"
        )

    K = int(data["codebook_size"])
    D = int(data["D"])
    if D != 768:
        raise RuntimeError(f"Unexpected VQ dimension D={D}; expected 768")

    l2_normalize = bool(data.get("l2_normalize", True))
    vq = VectorQuantizer(codebook_size=K, D=D, l2_normalize=l2_normalize)
    missing, unexpected = vq.load_state_dict(data["vq_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"VQ state mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    for p in vq.parameters():
        p.requires_grad = False
    return vq.to(device).eval(), data


def build_loader(roots, batch_size, num_workers, max_samples_per_root, seed):
    datasets = []
    rng = random.Random(seed)
    for root in roots:
        root = resolve_path(root)
        meta = root / "metadata.json"
        if not meta.exists():
            raise FileNotFoundError(meta)
        ds = CachedMelDataset(root=str(root), meta_file=str(meta), include_labels=False)
        if len(ds) == 0:
            continue
        if max_samples_per_root and len(ds) > max_samples_per_root:
            idx = list(range(len(ds)))
            rng.shuffle(idx)
            ds = Subset(ds, idx[:max_samples_per_root])
        datasets.append(ds)
    if not datasets:
        raise RuntimeError("No non-empty mel roots for VQ verification")
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    ), sum(len(ds) for ds in datasets)


@torch.no_grad()
def measure_usage(encoder, vq, loader, max_batches, codebook_size, device):
    counts = torch.zeros(codebook_size, device=device)
    seen_samples = 0
    seen_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        mel = batch["mel"].to(device, non_blocking=True)
        ids = vq(encoder(mel))["ids"].reshape(-1)
        counts += torch.bincount(ids, minlength=codebook_size).float()
        seen_samples += int(mel.size(0))
        seen_tokens += int(ids.numel())

    used = int((counts > 0).sum().item())
    probs = counts / (counts.sum() + 1e-10)
    perplexity = float((-(probs * (probs + 1e-10).log()).sum()).exp().item())
    return {
        "seen_samples": seen_samples,
        "seen_tokens": seen_tokens,
        "used_codes": used,
        "codebook_size": int(codebook_size),
        "utilization": float(used / codebook_size),
        "perplexity": perplexity,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-ckpt", default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt")
    p.add_argument("--vq-ckpt", default="checkpoints/vq/mel_htsat_v3_full_vq_K512_ema.pt")
    p.add_argument("--roots", nargs="*", default=None,
                   help="Override mel roots. Default: roots recorded in the VQ checkpoint.")
    p.add_argument("--max-batches", type=int, default=100,
                   help="Number of batches to measure; <=0 means full loader.")
    p.add_argument("--max-samples-per-root", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="checkpoints/rq2_matched_vq/vast_vq_verification.json")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_ckpt = resolve_path(args.encoder_ckpt)
    vq_ckpt = resolve_path(args.vq_ckpt)

    print(f"Device: {device}")
    print(f"Encoder: {encoder_ckpt}")
    print(f"VQ: {vq_ckpt}")

    encoder = load_encoder(encoder_ckpt, device)
    vq, vq_data = load_vq(vq_ckpt, encoder_ckpt, device)

    roots = args.roots or vq_data.get("mel_roots")
    if not roots:
        raise RuntimeError("No roots provided and VQ checkpoint has no mel_roots metadata")

    loader, n_samples = build_loader(
        roots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples_per_root=args.max_samples_per_root,
        seed=args.seed,
    )
    print(f"Verification roots: {len(roots)} total_samples={n_samples}")
    print(
        f"VQ metadata: K={vq_data['codebook_size']} D={vq_data['D']} "
        f"steps={vq_data.get('steps')} l2_normalize={vq_data.get('l2_normalize', True)}"
    )
    print(
        "Saved final metrics: "
        f"util={vq_data.get('final_sampled_utilization')} "
        f"perp={vq_data.get('final_sampled_perplexity')}"
    )

    max_batches = None if args.max_batches <= 0 else args.max_batches
    measured = measure_usage(
        encoder,
        vq,
        loader,
        max_batches=max_batches,
        codebook_size=int(vq_data["codebook_size"]),
        device=device,
    )
    print(
        "Remeasured metrics: "
        f"util={measured['utilization']:.3f} "
        f"perp={measured['perplexity']:.1f} "
        f"samples={measured['seen_samples']} tokens={measured['seen_tokens']}"
    )

    out = {
        "encoder_ckpt": str(encoder_ckpt),
        "vq_ckpt": str(vq_ckpt),
        "matched": True,
        "vq_metadata": {
            "codebook_size": int(vq_data["codebook_size"]),
            "D": int(vq_data["D"]),
            "steps": vq_data.get("steps"),
            "encoder_type": vq_data.get("encoder_type"),
            "encoder_checkpoint": vq_data.get("encoder_checkpoint"),
            "l2_normalize": vq_data.get("l2_normalize", True),
            "final_sampled_utilization": vq_data.get("final_sampled_utilization"),
            "final_sampled_perplexity": vq_data.get("final_sampled_perplexity"),
            "mel_roots": roots,
        },
        "remeasured": measured,
    }
    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
