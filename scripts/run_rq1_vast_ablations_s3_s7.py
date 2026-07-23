#!/usr/bin/env python3
"""Run VAST LP ablations on S3 and S7.

Variants:
  - VAST full: CSAF multi-stage fusion enabled.
  - no_csaf_stage4: disables channel/stage fusion and returns stage-4 only.
  - no_pos_encoding: disables HTS-AT absolute positional embedding and zeros
    Swin relative-position bias tables at evaluation time.

All variants use the same frozen encoder -> mean pooling -> standardized
logistic probe with train-only C selection protocol as RQ1.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_rq1_mel_linear_probe as rq1  # noqa: E402


DEFAULT_ABLATION_TASKS = ["S3_coswara_covid_exhale", "S7_b2ai"]


def disable_position_encoding(encoder):
    """Remove absolute and relative position signals for ablation."""
    if hasattr(encoder.htsat, "ape"):
        encoder.htsat.ape = False
    if hasattr(encoder.htsat, "absolute_pos_embed"):
        with torch.no_grad():
            encoder.htsat.absolute_pos_embed.zero_()

    zeroed = 0
    for module in encoder.modules():
        table = getattr(module, "relative_position_bias_table", None)
        if table is not None:
            with torch.no_grad():
                table.zero_()
            zeroed += 1
    return zeroed


def load_variant(name, ckpt_path, device):
    use_csaf = name != "no_csaf_stage4"
    encoder = rq1.load_vast_like_encoder(ckpt_path, device, use_csaf=use_csaf)
    meta = {"use_csaf": use_csaf, "position_encoding": "enabled"}
    if name == "no_pos_encoding":
        n_rel = disable_position_encoding(encoder)
        meta["position_encoding"] = "absolute disabled; relative bias zeroed"
        meta["relative_position_bias_tables_zeroed"] = n_rel
    return encoder, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="checkpoints/htsat_lejepa_v3_rq1_collective_orig_e8/htsat_lejepa_best.pt",
        help="VAST checkpoint to ablate. Default matches the current RQ1 VAST LP row.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full", "no_csaf_stage4", "no_pos_encoding"],
        choices=["full", "no_csaf_stage4", "no_pos_encoding"],
    )
    parser.add_argument("--output", default="checkpoints/ablations/rq1_vast_s3_s7_ablations.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--pools",
        nargs="+",
        default=["mean"],
        choices=["mean", "mean_std", "mean_std_max"],
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_ABLATION_TASKS,
        choices=list(rq1.S_TASKS.keys()),
    )
    parser.add_argument("--c-grid", default="0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rq1.C_GRID = [float(x) for x in args.c_grid.split(",") if x.strip()]
    rq1.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / args.ckpt
    ablation_tasks = {task: rq1.S_TASKS[task] for task in args.tasks}

    results = {
        "_meta": {
            "ckpt": str(ckpt_path),
            "tasks": list(ablation_tasks.keys()),
            "c_grid": rq1.C_GRID,
            "protocol": "frozen encoder -> mean pooling -> standardized logistic regression with train-only C selection",
        }
    }

    for variant in args.variants:
        print(f"\n{'=' * 72}\n{variant}\n{'=' * 72}")
        encoder, variant_meta = load_variant(variant, ckpt_path, device)
        row = {"_meta": variant_meta}
        for task_key, cfg in ablation_tasks.items():
            print(f"  Extracting {task_key} ({cfg['name']})")
            data = rq1.extract_task_features(
                encoder,
                cfg,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pool_names=args.pools,
            )
            row[task_key] = {}
            for pool_name in args.pools:
                res = rq1.fit_eval_probe(data["features"][pool_name], data["labels"], data["splits"])
                row[task_key][pool_name] = res
                print(
                    f"    {pool_name:<12} AUROC={res['auroc']:.4f} C={res['best_c']} "
                    f"n={res['n_train']}/{res['n_test']}"
                )
        results[variant] = row
        del encoder
        torch.cuda.empty_cache()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    headers = ["Variant", *[rq1.S_TASKS[t]["name"] for t in ablation_tasks], "Avg"]
    md = []
    for pool_name in args.pools:
        if md:
            md.append("")
        md.append(f"## {pool_name}")
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for variant in args.variants:
            vals = [
                results[variant][task][pool_name]["auroc"]
                for task in ablation_tasks
            ]
            row = [variant, *[f"{v:.4f}" for v in vals], f"{float(np.mean(vals)):.4f}"]
            md.append("| " + " | ".join(row) + " |")
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nSaved: {out}")
    print(f"Saved: {md_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
