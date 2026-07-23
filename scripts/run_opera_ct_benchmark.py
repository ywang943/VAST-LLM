"""
Run frozen multitask benchmark and CSAF adaptation using OPERA-CT backbone.

Same protocol as run_frozen_multitask_benchmark.py and
run_multitask_csaf_adaptation.py, but loads OPERA-CT weights directly
instead of a LeJEPA checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch

from respvoice.htsat_encoder import build_htsat_encoder
from scripts.run_frozen_multitask_benchmark import (
    TASKS, MultiCacheDataset, extract_features, split_indices,
    train_linear_probe,
)
from scripts.run_multitask_csaf_adaptation import (
    build_probe, cache_stages, evaluate_probe, run_seed,
)


def load_opera_ct(device):
    encoder = build_htsat_encoder(use_csaf=True)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device).eval()


def run_frozen(encoder, args, device):
    """Frozen encoder + linear probe (same as run_frozen_multitask_benchmark)."""
    results = {
        "protocol": "frozen encoder + mean pooling + linear classifier",
        "checkpoint": "OPERA-CT (original)",
        "initialization": "opera_ct",
        "tasks": {},
    }

    for task_key in args.tasks:
        cfg = TASKS[task_key]
        dataset = MultiCacheDataset(cfg["roots"])
        splits = split_indices(dataset, cfg["split"])
        print(f"\n{cfg['name']}: n={len(dataset)} "
              f"train/val/test={tuple(map(len, splits))}")

        task_result = {"note": cfg.get("note"), "variants": {}}
        for variant in args.variants:
            parts = [
                extract_features(encoder, dataset, idx, variant, device, args.batch_size)
                for idx in splits
            ]
            features = torch.cat([p[0] for p in parts])
            labels = torch.cat([p[1] for p in parts])
            offsets = np.cumsum([0] + [len(p[1]) for p in parts])
            local_splits = tuple(
                list(range(offsets[i], offsets[i + 1])) for i in range(3)
            )
            seed_results = [
                train_linear_probe(
                    features, labels, local_splits, cfg["n_classes"], seed,
                    args.epochs, args.lr,
                )
                for seed in args.seeds
            ]
            aucs = [r["auroc"] for r in seed_results]
            task_result["variants"][variant] = {
                "auroc_mean": float(np.nanmean(aucs)),
                "auroc_std": float(np.nanstd(aucs)),
                "per_seed": seed_results,
            }
            print(f"  {variant}: AUROC={np.nanmean(aucs):.4f} "
                  f"+/- {np.nanstd(aucs):.4f}")
        results["tasks"][task_key] = task_result

    out = Path(args.frozen_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFrozen results saved: {out}")
    return results


def run_adaptation(encoder, args, device):
    """Frozen HTS-AT + task-specific CSAF/pool/head adaptation."""
    from scripts.run_frozen_multitask_benchmark import encode_stages
    from torch.utils.data import DataLoader, Subset, TensorDataset

    csaf_state = {
        k: v.detach().cpu().clone()
        for k, v in encoder.csaf.state_dict().items()
    }

    results = {
        "protocol": "frozen HTS-AT + task-specific fusion adaptation",
        "checkpoint": "OPERA-CT (original)",
        "initialization": "opera_ct",
        "tasks": {},
    }

    for task_key in args.tasks:
        cfg = TASKS[task_key]
        dataset = MultiCacheDataset(cfg["roots"])
        splits = split_indices(dataset, cfg["split"])
        print(f"\n{cfg['name']}: caching four frozen HTS-AT stages")

        stages, labels = cache_stages(encoder, dataset, device, args.batch_size)
        task_result = {"note": cfg.get("note"), "variants": {}}

        for variant in args.adapt_variants:
            seed_results = [
                run_seed(stages, labels, splits, variant, cfg["n_classes"],
                         csaf_state, seed, args.adapt_epochs, args.adapt_lr, device)
                for seed in args.seeds
            ]
            aucs = [r["auroc"] for r in seed_results]
            task_result["variants"][variant] = {
                "auroc_mean": float(np.nanmean(aucs)),
                "auroc_std": float(np.nanstd(aucs)),
                "per_seed": seed_results,
            }
            print(f"  {variant}: AUROC={np.nanmean(aucs):.4f} "
                  f"+/- {np.nanstd(aucs):.4f}")
        results["tasks"][task_key] = task_result

    out = Path(args.adapt_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nAdaptation results saved: {out}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--variants", nargs="+",
                        choices=("stage4", "concat", "tpa_csaf"),
                        default=["stage4", "concat", "tpa_csaf"])
    parser.add_argument("--adapt-variants", nargs="+",
                        choices=("stage4", "concat", "csaf"),
                        default=["stage4", "concat", "csaf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adapt-epochs", type=int, default=64)
    parser.add_argument("--adapt-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frozen-output",
                        default="checkpoints/frozen_multitask/opera_ct_results.json")
    parser.add_argument("--adapt-output",
                        default="checkpoints/csaf_multitask_adapt/opera_ct_results.json")
    parser.add_argument("--skip-frozen", action="store_true")
    parser.add_argument("--skip-adapt", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading OPERA-CT encoder...")
    encoder = load_opera_ct(device)

    if not args.skip_frozen:
        run_frozen(encoder, args, device)

    if not args.skip_adapt:
        run_adaptation(encoder, args, device)

    print("\nAll OPERA-CT benchmarks complete.")


if __name__ == "__main__":
    main()
