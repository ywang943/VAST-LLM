"""
Aggregate all completed experimental results into a single paper_results.json.

Reads from:
  checkpoints/opera_official_icbhi_cont_ft_auc_weighted_multiseed_thresholds.json
  checkpoints/opera_feature_baseline/operaCT_icbhidisease.json
  checkpoints/opera_official_icbhi_large768_full/opera_official_summary.json
  checkpoints/downstream_copd_d128_cont_ft/summary.json
  checkpoints/downstream_kauh_d128_cont_ft/summary.json
  checkpoints/vq_ablation/ablation_results.json
  checkpoints/d768_multiseed_s{seed}/opera_official_summary.json  (if ready)

Usage:
  python scripts/collect_paper_results.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def mean_std(values):
    arr = [v for v in values if v is not None]
    if not arr:
        return None, None
    return round(float(np.mean(arr)), 4), round(float(np.std(arr)), 4)


def main():
    results = {}

    # ---------------------------------------------------------------
    # OPERA-CT feature baseline (local reproduction)
    # ---------------------------------------------------------------
    opera_baseline = load_json("checkpoints/opera_feature_baseline/operaCT_icbhidisease.json")
    if opera_baseline:
        # Structure: {"results": [{"seed": 0, "best": {"test": {"auroc": ...}}}]}
        seed_list = opera_baseline.get("results", opera_baseline.get("seeds", []))
        aurocs = []
        for s in seed_list:
            auc = (s.get("best", {}).get("test", {}).get("auroc")
                   or s.get("test_auroc"))
            if auc is not None:
                aurocs.append(auc)
        m, s = mean_std(aurocs)
        results["opera_ct_baseline"] = {
            "description": "OPERA-CT frozen features, linear head, official ICBHI split",
            "auroc_mean": m,
            "auroc_std": s,
            "n_seeds": len(aurocs),
            "per_seed_auroc": [round(x, 4) for x in aurocs],
            "note": "Local reproduction, lower than paper (0.855) due to protocol differences",
        }

    # ---------------------------------------------------------------
    # RespVoice D128, 5-seed Stage3 z_cont (official split)
    # ---------------------------------------------------------------
    d128_thresholds = load_json(
        "checkpoints/opera_official_icbhi_cont_ft_auc_weighted_multiseed_thresholds.json"
    )
    if d128_thresholds:
        # File is a list of per-seed dicts
        if isinstance(d128_thresholds, list):
            aurocs = [s.get("test_val_tuned", s).get("auroc") for s in d128_thresholds]
            accs   = [s.get("test_val_tuned", s).get("accuracy") for s in d128_thresholds]
            baccs  = [s.get("test_val_tuned", s).get("balanced_accuracy") for s in d128_thresholds]
        else:
            aurocs = d128_thresholds.get("per_seed_test_auroc", [])
            accs   = d128_thresholds.get("per_seed_calibrated_accuracy", [])
            baccs  = d128_thresholds.get("per_seed_calibrated_balanced_accuracy", [])
        m_auc, s_auc = mean_std(aurocs)
        m_acc, s_acc = mean_std(accs)
        m_bac, s_bac = mean_std(baccs)
        results["respvoice_d128_5seed"] = {
            "description": "RespVoice D=128, z_cont downstream, official ICBHI split, 5 seeds",
            "auroc_mean": m_auc, "auroc_std": s_auc,
            "accuracy_mean": m_acc, "accuracy_std": s_acc,
            "balanced_acc_mean": m_bac, "balanced_acc_std": s_bac,
            "per_seed_auroc": [round(x, 4) if x else None for x in aurocs],
            "n_seeds": len(aurocs),
        }

    # ---------------------------------------------------------------
    # RespVoice D768 single seed (official split)
    # ---------------------------------------------------------------
    d768_single = load_json(
        "checkpoints/opera_official_icbhi_large768_full/opera_official_summary.json"
    )
    if d768_single:
        test_ckpts = d768_single.get("checkpoints", {})
        best_test = test_ckpts.get("stage3_best_auc.pt", {}).get("test", {})
        results["respvoice_d768_seed0"] = {
            "description": "RespVoice D=768, z_cont downstream, official ICBHI split, seed 0 only",
            "auroc": best_test.get("auroc"),
            "accuracy": best_test.get("accuracy"),
            "balanced_acc": best_test.get("balanced_accuracy"),
            "note": "Single seed; multi-seed run in progress",
        }

    # ---------------------------------------------------------------
    # RespVoice D768 multi-seed (if completed)
    # ---------------------------------------------------------------
    d768_aurocs = []
    for seed in range(5):
        p = Path(f"checkpoints/d768_multiseed_s{seed}/opera_official_summary.json")
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            test = data.get("checkpoints", {}).get("stage3_best_auc.pt", {}).get("test", {})
            if "auroc" in test:
                d768_aurocs.append(test["auroc"])
    if d768_aurocs:
        m, s = mean_std(d768_aurocs)
        results["respvoice_d768_multiseed"] = {
            "description": f"RespVoice D=768, z_cont, official split, {len(d768_aurocs)}/5 seeds done",
            "auroc_mean": m, "auroc_std": s,
            "per_seed_auroc": d768_aurocs,
            "n_seeds_done": len(d768_aurocs),
        }

    # ---------------------------------------------------------------
    # Transfer: COPD severity + KAUH
    # ---------------------------------------------------------------
    transfer_tasks = [
        ("downstream_copd_d128_cont_ft", "copd_severity_5class", "COPD Severity 5-class"),
        ("downstream_kauh_d128_cont_ft", "kauh_obstructive",     "KAUH Healthy vs Obstructive"),
    ]
    for task, key, desc in transfer_tasks:
        data = load_json(f"checkpoints/{task}/summary.json")
        if data:
            # Try nested checkpoints → stage3_best_auc.pt → test
            test = (data.get("checkpoints", {})
                    .get("stage3_best_auc.pt", {})
                    .get("test", {}))
            results[key] = {
                "description": desc,
                "auroc": test.get("auroc") or data.get("test_auroc"),
                "accuracy": test.get("accuracy") or data.get("test_accuracy"),
                "balanced_acc": test.get("balanced_accuracy") or data.get("test_balanced_accuracy"),
                "n_classes": data.get("n_classes"),
                "note": "Single seed, D128 z_cont fine-tune",
            }

    # ---------------------------------------------------------------
    # VQ ablation (D128)
    # ---------------------------------------------------------------
    ablation = load_json("checkpoints/vq_ablation/ablation_results.json")
    if ablation:
        results["vq_ablation_d128"] = {
            "description": "VQ codebook utilization ablation, D=128, K=512, 5 epochs",
            "conditions": [
                {k: v for k, v in cond.items() if k not in ("error",)}
                for cond in ablation if "error" not in cond
            ],
            "summary": {
                cond["label"]: {"util": cond["best_util"], "perp": cond["best_perp"]}
                for cond in ablation if "error" not in cond
            },
        }

    # ---------------------------------------------------------------
    # Print and save
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  PAPER RESULTS SUMMARY")
    print("=" * 65)

    if "opera_ct_baseline" in results:
        r = results["opera_ct_baseline"]
        print(f"\nOPERA-CT baseline (local): AUROC {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")

    if "respvoice_d128_5seed" in results:
        r = results["respvoice_d128_5seed"]
        print(f"RespVoice D128 (5-seed):   AUROC {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")
        print(f"  per-seed: {[round(x,3) for x in r['per_seed_auroc']]}")

    if "respvoice_d768_multiseed" in results:
        r = results["respvoice_d768_multiseed"]
        print(f"RespVoice D768 ({r['n_seeds_done']}/5 seeds): AUROC {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")
    elif "respvoice_d768_seed0" in results:
        r = results["respvoice_d768_seed0"]
        print(f"RespVoice D768 (seed 0):   AUROC {r['auroc']:.3f}  (multi-seed pending)")

    if "vq_ablation_d128" in results:
        print(f"\nVQ Ablation (D=128, K=512):")
        for k, v in results["vq_ablation_d128"]["summary"].items():
            print(f"  {k:<40} util={v['util']:.3f}  perp={v['perp']:.1f}")

    for key, label in [("copd_severity_5class", "COPD severity 5-class"),
                        ("kauh_obstructive", "KAUH obstructive")]:
        if key in results:
            r = results[key]
            auc = r.get("auroc")
            auc_str = f"{auc:.3f}" if auc else "N/A"
            print(f"{label}: AUROC={auc_str}  (transfer, single seed)")

    out = Path("checkpoints/paper_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
