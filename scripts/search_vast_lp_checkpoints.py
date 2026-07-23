#!/usr/bin/env python3
"""Screen VAST checkpoints for frozen linear-probe performance on selected RQ1 tasks."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_rq1_mel_linear_probe import (  # noqa: E402
    S_TASKS,
    extract_task_features,
    fit_eval_probe,
    load_vast_like_encoder,
    seed_everything,
)


DEFAULT_CKPTS = [
    "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_ep90.pt",
    "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_ep100.pt",
    "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_ep110.pt",
    "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_ep120.pt",
    "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_ep130.pt",
    "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_ep140.pt",
    "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_ep150.pt",
    "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt",
    "checkpoints/htsat_lejepa_lamsig002/htsat_lejepa_best.pt",
    "checkpoints/htsat_lejepa_long/htsat_lejepa_best.pt",
    "checkpoints/htsat_lejepa_scratch_clean/htsat_lejepa_best.pt",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=[
        "S3_coswara_covid_exhale",
        "S4_coswara_covid_cough",
        "S5_coswara_smoker_cough",
        "S6_svd",
    ], choices=list(S_TASKS.keys()))
    parser.add_argument("--pools", nargs="+", default=["mean", "mean_std_max"],
                        choices=["mean", "mean_std", "mean_std_max"])
    parser.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default="checkpoints/rq1_mel_linear_probe/vast_lp_checkpoint_screen.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for ckpt in args.ckpts:
        ckpt_path = ROOT / ckpt
        if not ckpt_path.exists():
            print(f"Skip missing {ckpt}")
            continue
        name = str(Path(ckpt).parent.name + "/" + Path(ckpt).name)
        print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
        encoder = load_vast_like_encoder(ckpt_path, device, use_csaf=True)
        row = {}
        for task_key in args.tasks:
            cfg = S_TASKS[task_key]
            print(f"  {task_key} {cfg['name']}")
            data = extract_task_features(
                encoder, cfg, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pool_names=args.pools,
            )
            task_out = {}
            for pool in args.pools:
                res = fit_eval_probe(data["features"][pool], data["labels"], data["splits"])
                task_out[pool] = res
                print(f"    {pool}: {res['auroc']:.4f}" if res else f"    {pool}: —")
            row[task_key] = task_out
        results[name] = row
        del encoder
        torch.cuda.empty_cache()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    for pool in args.pools:
        print(f"\nPOOL {pool}")
        scored = []
        for name, row in results.items():
            vals = [row[t][pool]["auroc"] for t in args.tasks if row[t][pool]]
            scored.append((float(np.mean(vals)), name, vals))
        for mean, name, vals in sorted(scored, reverse=True):
            print(f"{mean:.4f} {name} " + " ".join(f"{v:.4f}" for v in vals))


if __name__ == "__main__":
    main()
