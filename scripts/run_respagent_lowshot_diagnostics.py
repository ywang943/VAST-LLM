#!/usr/bin/env python
"""Low-shot minority augmentation diagnostics for the RespAgent-style section.

This script uses cached VAST features and compares:
  - real low-shot training only
  - naive copy + jitter augmentation
  - VAST-gen latent interpolation augmentation
  - hard VAST-gen: generate from the lowest-confidence minority examples

All evaluation is on real held-out test samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_all_baselines import S_TASKS


def load_cached(task_key: str, cache_dir: Path):
    p = cache_dir / f"{task_key}_vast_features.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p, allow_pickle=True)
    return d["features"], d["labels"].astype(int), d["splits"].astype(str)


def train_val_test(X, y, splits, seed):
    train_mask = splits == "train"
    test_mask = splits == "test"
    X_train_all, y_train_all = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(np.unique(y_train_all)) < 2:
        raise ValueError("train split has fewer than two classes")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(sss.split(X_train_all, y_train_all))
    return X_train_all[tr_idx], y_train_all[tr_idx], X_train_all[va_idx], y_train_all[va_idx], X_test, y_test


def make_lowshot(X, y, k, seed, keep_majority="all"):
    rng = np.random.default_rng(seed)
    counts = Counter(y.tolist())
    minority = min(counts, key=counts.get)
    majority = max(counts, key=counts.get)
    min_idx = np.where(y == minority)[0]
    maj_idx = np.where(y == majority)[0]
    if len(min_idx) < k:
        k = len(min_idx)
    chosen_min = rng.choice(min_idx, size=k, replace=False)
    if keep_majority == "match":
        nmaj = min(len(maj_idx), max(k * 4, k))
        chosen_maj = rng.choice(maj_idx, size=nmaj, replace=False)
    else:
        chosen_maj = maj_idx
    idx = np.concatenate([chosen_maj, chosen_min])
    rng.shuffle(idx)
    return X[idx], y[idx], int(minority), {"minority_kept": int(k), "majority_kept": int(len(chosen_maj))}


def naive_augment(X, y, minority, target_count, seed, noise_scale=0.03, scale_jitter=0.03, source_idx=None):
    rng = np.random.default_rng(seed)
    idx = np.where(y == minority)[0] if source_idx is None else np.asarray(source_idx)
    current = int((y == minority).sum())
    n_make = max(0, target_count - current)
    if n_make == 0 or len(idx) == 0:
        return X, y, 0
    class_X = X[idx]
    class_std = class_X.std(axis=0) + 1e-6
    new = []
    for _ in range(n_make):
        base = class_X[rng.integers(0, len(class_X))]
        z = base * rng.normal(1.0, scale_jitter)
        z = z + rng.normal(0.0, noise_scale, size=z.shape) * class_std
        new.append(z.astype(np.float32))
    Xn = np.concatenate([X, np.stack(new)], axis=0)
    yn = np.concatenate([y, np.full(n_make, minority, dtype=y.dtype)])
    return Xn, yn, int(n_make)


def vast_gen_augment(X, y, minority, target_count, seed, noise_scale=0.03, source_idx=None):
    rng = np.random.default_rng(seed)
    idx = np.where(y == minority)[0] if source_idx is None else np.asarray(source_idx)
    current = int((y == minority).sum())
    n_make = max(0, target_count - current)
    if n_make == 0 or len(idx) == 0:
        return X, y, 0
    class_X = X[idx]
    class_std = class_X.std(axis=0) + 1e-6
    new = []
    for _ in range(n_make):
        i, j = rng.choice(len(class_X), size=2, replace=True)
        lam = rng.uniform(0.2, 0.8)
        z = lam * class_X[i] + (1.0 - lam) * class_X[j]
        z = z + rng.normal(0.0, noise_scale, size=z.shape) * class_std
        new.append(z.astype(np.float32))
    Xn = np.concatenate([X, np.stack(new)], axis=0)
    yn = np.concatenate([y, np.full(n_make, minority, dtype=y.dtype)])
    return Xn, yn, int(n_make)


def fit_model(X_train, y_train, C=1.0):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=3000, C=C, solver="lbfgs", class_weight=None)
    clf.fit(Xt, y_train)
    return scaler, clf


def minority_probs(scaler, clf, X, minority):
    probs = clf.predict_proba(scaler.transform(X))
    cls = list(clf.classes_)
    return probs[:, cls.index(minority)]


def sensitivity_at_specificity(y_val, p_val, y_test, p_test, minority, spec_target=0.90):
    yv = (y_val == minority).astype(int)
    yt = (y_test == minority).astype(int)
    thresholds = np.unique(p_val)
    thresholds = np.r_[thresholds.max() + 1e-9, thresholds[::-1], thresholds.min() - 1e-9]
    best_thr = thresholds[0]
    best_sens = -1.0
    for thr in thresholds:
        pred = (p_val >= thr).astype(int)
        neg = yv == 0
        pos = yv == 1
        spec = ((pred[neg] == 0).sum() / max(1, neg.sum()))
        sens = ((pred[pos] == 1).sum() / max(1, pos.sum()))
        if spec >= spec_target and sens > best_sens:
            best_sens = sens
            best_thr = thr
    pred_test = (p_test >= best_thr).astype(int)
    neg = yt == 0
    pos = yt == 1
    spec = ((pred_test[neg] == 0).sum() / max(1, neg.sum()))
    sens = ((pred_test[pos] == 1).sum() / max(1, pos.sum()))
    return float(sens), float(spec), float(best_thr)


def eval_method(X_train, y_train, X_val, y_val, X_test, y_test, minority, C=1.0):
    scaler, clf = fit_model(X_train, y_train, C=C)
    p_test = minority_probs(scaler, clf, X_test, minority)
    p_val = minority_probs(scaler, clf, X_val, minority)
    y_test_bin = (y_test == minority).astype(int)
    pred = clf.predict(scaler.transform(X_test))
    sens90, spec_at, thr = sensitivity_at_specificity(y_val, p_val, y_test, p_test, minority)
    return {
        "auroc": float(roc_auc_score(y_test_bin, p_test)),
        "pr_auc": float(average_precision_score(y_test_bin, p_test)),
        "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
        "minority_recall": float(recall_score(y_test, pred, labels=[minority], average="macro", zero_division=0)),
        "sens_at_90spec": sens90,
        "spec_at_threshold": spec_at,
        "threshold": thr,
    }, (scaler, clf)


def hard_minority_indices(X_low, y_low, minority, seed, hard_fraction=0.5):
    scaler, clf = fit_model(X_low, y_low, C=1.0)
    idx = np.where(y_low == minority)[0]
    p = minority_probs(scaler, clf, X_low[idx], minority)
    n_hard = max(2, int(np.ceil(len(idx) * hard_fraction)))
    hard_local = np.argsort(p)[: min(n_hard, len(idx))]
    return idx[hard_local]


def aggregate(records):
    out = {}
    for key in records[0]:
        vals = np.array([r[key] for r in records], dtype=float)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["S3_coswara_covid_cough", "S5_coswara_smoker", "S6_icbhi_copd", "S7_b2ai"],
                   choices=list(S_TASKS.keys()))
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--cache-dir", default="checkpoints/respagent_style/feature_cache")
    p.add_argument("--out", default="checkpoints/respagent_style/lowshot_minority_diagnostics.json")
    p.add_argument("--noise-scale", type=float, default=0.03)
    p.add_argument(
        "--aug-mults",
        nargs="+",
        type=int,
        default=[2, 5, 10],
        help="Target minority count is k * multiplier, capped by majority count.",
    )
    args = p.parse_args()

    cache_dir = ROOT / args.cache_dir
    output = {
        "protocol": "low-shot minority real-only vs naive copy-jitter vs VAST latent interpolation vs hard-minority VAST generation; fixed LR C=1.0; real test only",
        "metrics": ["auroc", "pr_auc", "balanced_acc", "minority_recall", "sens_at_90spec"],
        "tasks": {},
    }

    for task in args.tasks:
        X, y, splits = load_cached(task, cache_dir)
        task_out = {"name": S_TASKS[task]["name"], "ks": {}}
        print(f"\n=== {task} {S_TASKS[task]['name']} ===")
        for k in args.ks:
            task_out["ks"].setdefault(str(k), {})
            for mult in args.aug_mults:
                by_method = defaultdict(list)
                gen_counts = defaultdict(list)
                details = []
                for seed in args.seeds:
                    Xtr, ytr, Xval, yval, Xte, yte = train_val_test(X, y, splits, seed)
                    Xlow, ylow, minority, info = make_lowshot(Xtr, ytr, k, seed)
                    majority_count = int(max(Counter(ylow.tolist()).values()))
                    target_count = int(min(majority_count, k * mult))
                    real_metrics, _ = eval_method(Xlow, ylow, Xval, yval, Xte, yte, minority)
                    by_method["real"].append(real_metrics)

                    Xn, yn, n_naive = naive_augment(Xlow, ylow, minority, target_count, seed, args.noise_scale)
                    naive_metrics, _ = eval_method(Xn, yn, Xval, yval, Xte, yte, minority)
                    by_method["naive"].append(naive_metrics)
                    gen_counts["naive"].append(n_naive)

                    Xg, yg, n_gen = vast_gen_augment(Xlow, ylow, minority, target_count, seed, args.noise_scale)
                    gen_metrics, _ = eval_method(Xg, yg, Xval, yval, Xte, yte, minority)
                    by_method["vast_gen"].append(gen_metrics)
                    gen_counts["vast_gen"].append(n_gen)

                    hard_idx = hard_minority_indices(Xlow, ylow, minority, seed)
                    Xh, yh, n_hard = vast_gen_augment(Xlow, ylow, minority, target_count, seed, args.noise_scale, source_idx=hard_idx)
                    hard_metrics, _ = eval_method(Xh, yh, Xval, yval, Xte, yte, minority)
                    by_method["vast_hard"].append(hard_metrics)
                    gen_counts["vast_hard"].append(n_hard)

                    details.append({
                        "seed": seed,
                        "minority": int(minority),
                        "target_count": int(target_count),
                        "lowshot_info": info,
                    })

                agg = {m: aggregate(v) for m, v in by_method.items()}
                for m in ["naive", "vast_gen", "vast_hard"]:
                    agg[m]["generated_mean"] = float(np.mean(gen_counts[m]))
                task_out["ks"][str(k)][str(mult)] = {"methods": agg, "details": details}
                print(
                    f"k={k} mult={mult}: "
                    f"real sens90={agg['real']['sens_at_90spec']['mean']:.3f}; "
                    f"naive={agg['naive']['sens_at_90spec']['mean']:.3f}; "
                    f"vast={agg['vast_gen']['sens_at_90spec']['mean']:.3f}; "
                    f"hard={agg['vast_hard']['sens_at_90spec']['mean']:.3f}"
                )
        output["tasks"][task] = task_out

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = [
        "# Low-Shot Minority Synthetic Augmentation Diagnostics",
        "",
        "Fixed LR C=1.0. Test set is always real. Values are mean±std across seeds.",
        "",
        "| Task | k | Method | AUROC | PR-AUC | Bal.Acc | Min Recall | Sens@90%Spec |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for task, t in output["tasks"].items():
        for k, by_mult in t["ks"].items():
            for mult, kd in by_mult.items():
                for method in ["real", "naive", "vast_gen", "vast_hard"]:
                    m = kd["methods"][method]
                    def fs(metric):
                        return f"{m[metric]['mean']:.4f}±{m[metric]['std']:.4f}"
                    lines.append(
                        f"| {t['name']} | {k}x{mult} | {method} | {fs('auroc')} | {fs('pr_auc')} | "
                        f"{fs('balanced_acc')} | {fs('minority_recall')} | {fs('sens_at_90spec')} |"
                    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
