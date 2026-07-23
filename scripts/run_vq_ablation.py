"""
VQ Codebook Utilization Ablation for the paper.

Compares SIGReg vs No-SIGReg at Stage 2, using the D128 pipeline.
This is the core experiment from audio.md Section 9.

Conditions:
  A) SIGReg OFF (random encoder)  + VQ: baseline collapse
  B) SIGReg ON  (Stage1 encoder)  + VQ: our method
  C) SIGReg ON + VQ (no L2 norm) : intermediate
  D) SIGReg ON + VQ + L2 norm    : full method (default)

Usage:
  .venv\Scripts\python.exe scripts\run_vq_ablation.py
  .venv\Scripts\python.exe scripts\run_vq_ablation.py --stage1-ckpt ./checkpoints/full_local_weighted_lp/stage1_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer


def run_stage2(
    label: str,
    pretrain_cache: str,
    stage1_ckpt=None,
    codebook_size: int,
    use_ema: bool,
    l2_normalize: bool,
    epochs: int,
    batch_size: int,
    D: int,
    encoder_layers: int,
    encoder_heads: int,
) -> dict:
    cfg = RespVoiceConfig(
        model=ModelConfig(
            D=D,
            codebook_size=codebook_size,
            encoder_layers=encoder_layers,
            encoder_heads=encoder_heads,
            predictor_layers=1,
            n_sigreg_slices=32,
            vq_use_ema=use_ema,
            vq_ema_decay=0.99,
            vq_restart_threshold=1,
            vq_l2_normalize=l2_normalize,
        ),
        train=TrainConfig(
            stage2_epochs=epochs,
            stage2_lr=1e-4,
            lam_recon=0.05,
            batch_size=batch_size,
            warmup_ratio=0.05,
            num_workers=0,
        ),
        checkpoint_dir=f"./checkpoints/vq_ablation/{label.replace(' ','_').replace(':','').replace('(','').replace(')','').replace('/','_')}",
        log_dir=f"./logs/vq_ablation/{label.replace(' ','_').replace(':','').replace('(','').replace(')','').replace('/','_')}",
    )

    model = RespVoiceModel(cfg.model)

    if stage1_ckpt and Path(stage1_ckpt).exists():
        state = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
        full = state["model_state"]
        # Load only encoder+predictor, skip VQ (may have different codebook_size)
        enc_only = {k: v for k, v in full.items() if not k.startswith("vq.") and not k.startswith("decoder.")}
        model.load_state_dict(enc_only, strict=False)
        print(f"  Loaded Stage1 encoder: {stage1_ckpt}")
    else:
        print("  Using random encoder (no Stage1 pretraining)")

    cache = Path(pretrain_cache)
    ds = CachedMelDataset(
        root=str(cache), meta_file=str(cache / "metadata.json"), include_labels=False
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    trainer = Trainer(cfg, model)
    trainer.train_stage2(loader)

    # Read epoch-level stats from log
    log_path = Path(cfg.log_dir) / "train_log.jsonl"
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    s2_rows = [r for r in rows if r.get("stage") == "stage2"]
    best_util = max((r["util"] for r in s2_rows), default=0.0)
    best_perp = max((r["perp"] for r in s2_rows), default=0.0)
    last = s2_rows[-1] if s2_rows else {}

    result = {
        "label": label,
        "codebook_size": codebook_size,
        "l2_normalize": l2_normalize,
        "use_ema": use_ema,
        "has_sigreg": stage1_ckpt is not None,
        "best_util": round(best_util, 4),
        "best_perp": round(best_perp, 1),
        "final_util": round(last.get("util", 0), 4),
        "final_perp": round(last.get("perp", 0.0), 1),
        "epochs": epochs,
    }
    print(f"\n  [{label}]  best_util={best_util:.3f}  best_perp={best_perp:.1f}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", default="./checkpoints/opera_official_icbhi_cont_ft/stage2_final.pt",
                   help="D128 Stage1 checkpoint for SIGReg-ON conditions")
    p.add_argument("--pretrain-cache", default="./data/mel_cache/pretrain_full")
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--encoder-heads", type=int, default=4)
    p.add_argument("--out", default="./checkpoints/vq_ablation/ablation_results.json")
    args = p.parse_args()

    print("=" * 60)
    print("  VQ Codebook Utilization Ablation")
    print(f"  D={args.dim}  K={args.codebook_size}  epochs={args.epochs}")
    print("=" * 60)

    conditions = [
        # (label, stage1_ckpt, use_ema, l2_normalize)
        ("A: No-SIGReg No-EMA No-L2",  None,            False, False),
        ("B: SIGReg No-EMA No-L2",     args.stage1_ckpt, False, False),
        ("C: SIGReg EMA No-L2",        args.stage1_ckpt, True,  False),
        ("D: SIGReg EMA L2 (ours)",    args.stage1_ckpt, True,  True),
    ]

    results = []
    for label, ckpt, use_ema, l2n in conditions:
        print(f"\n{'='*60}")
        print(f"  Running: {label}")
        print(f"{'='*60}")
        try:
            r = run_stage2(
                label=label,
                stage1_ckpt=ckpt,
                pretrain_cache=args.pretrain_cache,
                codebook_size=args.codebook_size,
                use_ema=use_ema,
                l2_normalize=l2n,
                epochs=args.epochs,
                batch_size=args.batch_size,
                D=args.dim,
                encoder_layers=args.encoder_layers,
                encoder_heads=args.encoder_heads,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR in {label}: {e}")
            results.append({"label": label, "error": str(e)})

    print("\n" + "=" * 60)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Condition':<35}  {'util':>6}  {'perp':>8}")
    print("  " + "-" * 55)
    for r in results:
        if "error" not in r:
            print(f"  {r['label']:<35}  {r['best_util']:>6.3f}  {r['best_perp']:>8.1f}")
        else:
            print(f"  {r['label']:<35}  ERROR: {r['error']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out}")


if __name__ == "__main__":
    main()
