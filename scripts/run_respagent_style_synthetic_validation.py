#!/usr/bin/env python
"""RespAgent-style synthetic augmentation validation on local VAST features.

This is a lightweight reproduction of RespAgent's downstream-utility idea:
generate class-conditional samples for under-represented classes, train the
diagnostic classifier with real + generated samples, and evaluate only on real
held-out test samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_all_baselines import S_TASKS, load_htsat_encoder


def extract_mel_features_batched(encoder, task_cfg, device, batch_size=24):
    mel_dir = ROOT / task_cfg["mel_root"]
    meta = json.loads((mel_dir / "metadata.json").read_text(encoding="utf-8"))["samples"]
    feats, labels, splits = [], [], []
    batch, batch_labels, batch_splits = [], [], []

    def flush():
        if not batch:
            return
        max_t = max(x.shape[-1] for x in batch)
        padded = []
        for x in batch:
            if x.shape[-1] < max_t:
                x = torch.nn.functional.pad(x, (0, max_t - x.shape[-1]))
            padded.append(x)
        mel = torch.stack(padded, dim=0).to(device)
        with torch.no_grad():
            z = encoder(mel).mean(dim=1).detach().cpu().numpy()
        feats.append(z)
        labels.extend(batch_labels)
        splits.extend(batch_splits)
        batch.clear()
        batch_labels.clear()
        batch_splits.clear()

    for sample in meta:
        if "label" not in sample:
            continue
        path = mel_dir / sample["path"]
        if not path.exists():
            continue
        mel = torch.load(path, map_location="cpu")
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.dim() != 3:
            continue
        batch.append(mel.float())
        batch_labels.append(int(sample["label"]))
        batch_splits.append(sample.get("split", "train"))
        if len(batch) >= batch_size:
            flush()
    flush()
    return {
        "features": np.concatenate(feats, axis=0),
        "labels": np.array(labels),
        "splits": splits,
    }


def split_arrays(data):
    X = data["features"]
    y = data["labels"]
    splits = np.array(data["splits"])
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    if not val.any():
        val = train.copy()
    return X[train], y[train], X[val], y[val], X[test], y[test]


def synthesize_minority_features(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 1.0,
    noise_scale: float = 0.03,
    seed: int = 0,
):
    """Class-conditional interpolation + small noise, similar to latent SMOTE."""
    rng = np.random.default_rng(seed)
    counts = Counter(y.tolist())
    max_count = max(counts.values())
    X_new, y_new = [], []

    for cls, count in sorted(counts.items()):
        target = int(round(max_count * target_ratio))
        n_to_make = max(0, target - count)
        if n_to_make == 0:
            continue
        idx = np.where(y == cls)[0]
        if len(idx) < 2:
            continue
        class_X = X[idx]
        class_std = class_X.std(axis=0, keepdims=True)
        for _ in range(n_to_make):
            i, j = rng.choice(len(class_X), size=2, replace=True)
            lam = rng.uniform(0.25, 0.75)
            z = lam * class_X[i] + (1.0 - lam) * class_X[j]
            z = z + rng.normal(0.0, noise_scale, size=z.shape) * (class_std.squeeze(0) + 1e-6)
            X_new.append(z.astype(np.float32))
            y_new.append(cls)

    if not X_new:
        return X, y, {"generated": 0, "class_counts_before": dict(counts), "class_counts_after": dict(counts)}

    X_aug = np.concatenate([X, np.stack(X_new)], axis=0)
    y_aug = np.concatenate([y, np.array(y_new, dtype=y.dtype)], axis=0)
    return X_aug, y_aug, {
        "generated": int(len(y_new)),
        "class_counts_before": {str(k): int(v) for k, v in counts.items()},
        "class_counts_after": {str(k): int(v) for k, v in Counter(y_aug.tolist()).items()},
    }


def naive_minority_augment(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 1.0,
    noise_scale: float = 0.03,
    scale_jitter: float = 0.03,
    seed: int = 0,
):
    """Naive copy + tiny feature jitter baseline.

    This is the feature-space analogue of simple signal transforms such as
    small amplitude scaling or adding/subtracting weak noise.  It intentionally
    does not interpolate between class examples or create new class-conditional
    directions.
    """
    rng = np.random.default_rng(seed)
    counts = Counter(y.tolist())
    max_count = max(counts.values())
    X_new, y_new = [], []

    for cls, count in sorted(counts.items()):
        target = int(round(max_count * target_ratio))
        n_to_make = max(0, target - count)
        if n_to_make == 0:
            continue
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        class_X = X[idx]
        class_std = class_X.std(axis=0, keepdims=True).squeeze(0)
        for _ in range(n_to_make):
            base = class_X[rng.integers(0, len(class_X))]
            scale = rng.normal(1.0, scale_jitter)
            z = base * scale
            z = z + rng.normal(0.0, noise_scale, size=z.shape) * (class_std + 1e-6)
            X_new.append(z.astype(np.float32))
            y_new.append(cls)

    if not X_new:
        return X, y, {"generated": 0, "class_counts_before": dict(counts), "class_counts_after": dict(counts)}

    X_aug = np.concatenate([X, np.stack(X_new)], axis=0)
    y_aug = np.concatenate([y, np.array(y_new, dtype=y.dtype)], axis=0)
    return X_aug, y_aug, {
        "generated": int(len(y_new)),
        "class_counts_before": {str(k): int(v) for k, v in counts.items()},
        "class_counts_after": {str(k): int(v) for k, v in Counter(y_aug.tolist()).items()},
    }


def fit_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, tune_c=True, c_values=None):
    if c_values is not None:
        c_grid = c_values
    else:
        c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0] if tune_c else [1.0]
    best = None
    best_score = -np.inf
    for c in c_grid:
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_train)
        Xv = scaler.transform(X_val)
        clf = LogisticRegression(max_iter=3000, C=c, solver="lbfgs")
        clf.fit(Xt, y_train)
        if len(set(y_val.tolist())) == 2:
            val_prob = clf.predict_proba(Xv)[:, 1]
            score = roc_auc_score(y_val, val_prob)
        else:
            val_prob = clf.predict_proba(Xv)
            score = roc_auc_score(y_val, val_prob, multi_class="ovr", average="macro")
        if score > best_score:
            best_score = score
            best = (c, scaler, clf)

    c, scaler, clf = best
    Xs = scaler.transform(X_test)
    pred = clf.predict(Xs)
    probs = clf.predict_proba(Xs)
    labels = sorted(set(y_test.tolist()))
    if len(labels) == 2:
        auroc = roc_auc_score(y_test, probs[:, 1])
        minority = min(Counter(y_train.tolist()), key=Counter(y_train.tolist()).get)
        minority_recall = recall_score(y_test, pred, labels=[minority], average="macro", zero_division=0)
    else:
        auroc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
        minority = min(Counter(y_train.tolist()), key=Counter(y_train.tolist()).get)
        minority_recall = recall_score(y_test, pred, labels=[minority], average="macro", zero_division=0)
    return {
        "auroc": float(auroc),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "minority_label": int(minority),
        "minority_recall": float(minority_recall),
        "best_c": float(c),
    }


def fmt(x):
    return "NA" if x is None else f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["S3_coswara_covid_cough", "S5_coswara_smoker", "S6_icbhi_copd", "S7_b2ai"],
        choices=list(S_TASKS.keys()),
    )
    parser.add_argument("--target-ratio", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--feature-cache-dir", default="checkpoints/respagent_style/feature_cache")
    parser.add_argument("--fixed-c", type=float, default=None, help="Use one fixed LR C instead of validation tuning.")
    parser.add_argument("--out", default="checkpoints/respagent_style/local_synthetic_validation.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cache_dir = ROOT / args.feature_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_cache = [
        t for t in args.tasks
        if not (cache_dir / f"{t}_vast_features.npz").exists()
    ]
    encoder = None
    if missing_cache:
        print(f"Missing feature cache for {missing_cache}; loading encoder.")
        encoder = load_htsat_encoder(ROOT / args.ckpt, device)
    else:
        print("All feature caches found; skipping encoder load.")

    output = {
        "protocol": "RespAgent-style local validation: real-only vs real+class-conditional synthetic VAST features; real test set only",
        "encoder_ckpt": args.ckpt,
        "target_ratio": args.target_ratio,
        "noise_scale": args.noise_scale,
        "tasks": {},
    }

    rows = []
    for task_key in args.tasks:
        cfg = S_TASKS[task_key]
        print(f"\n=== {task_key}: {cfg['name']} ===")
        cache_file = cache_dir / f"{task_key}_vast_features.npz"
        if cache_file.exists():
            cached = np.load(cache_file, allow_pickle=True)
            data = {
                "features": cached["features"],
                "labels": cached["labels"],
                "splits": cached["splits"].tolist(),
            }
            print(f"loaded feature cache: {cache_file}")
        else:
            data = extract_mel_features_batched(encoder, cfg, device, batch_size=args.batch_size)
            np.savez_compressed(
                cache_file,
                features=data["features"],
                labels=data["labels"],
                splits=np.array(data["splits"], dtype=object),
            )
            print(f"saved feature cache: {cache_file}")
        Xtr, ytr, Xval, yval, Xte, yte = split_arrays(data)
        print(f"train={len(ytr)} val={len(yval)} test={len(yte)} train_counts={dict(Counter(ytr.tolist()))}")
        c_values = [args.fixed_c] if args.fixed_c is not None else None
        tune_c = (len(set(yval.tolist())) > 1) and args.fixed_c is None
        real = fit_and_eval(Xtr, ytr, Xval, yval, Xte, yte, tune_c=tune_c, c_values=c_values)
        Xnaive, ynaive, naive_info = naive_minority_augment(
            Xtr, ytr, target_ratio=args.target_ratio, noise_scale=args.noise_scale, seed=args.seed
        )
        naive = fit_and_eval(Xnaive, ynaive, Xval, yval, Xte, yte, tune_c=tune_c, c_values=c_values)
        Xaug, yaug, synth_info = synthesize_minority_features(
            Xtr, ytr, target_ratio=args.target_ratio, noise_scale=args.noise_scale, seed=args.seed
        )
        aug = fit_and_eval(Xaug, yaug, Xval, yval, Xte, yte, tune_c=tune_c, c_values=c_values)
        delta_naive = {k: naive[k] - real[k] for k in ["auroc", "accuracy", "macro_f1", "minority_recall"]}
        delta = {k: aug[k] - real[k] for k in ["auroc", "accuracy", "macro_f1", "minority_recall"]}
        output["tasks"][task_key] = {
            "name": cfg["name"],
            "n_train": int(len(ytr)),
            "n_val": int(len(yval)),
            "n_test": int(len(yte)),
            "real_only": real,
            "naive_copy_jitter": naive,
            "real_plus_synthetic": aug,
            "delta_naive": delta_naive,
            "delta": delta,
            "naive": naive_info,
            "synthetic": synth_info,
        }
        rows.append((cfg["name"], real, naive, aug, delta_naive, delta, naive_info["generated"], synth_info["generated"]))
        print(
            f"real AUROC={real['auroc']:.4f} rec_min={real['minority_recall']:.4f}; "
            f"naive AUROC={naive['auroc']:.4f} rec_min={naive['minority_recall']:.4f}; "
            f"synthetic AUROC={aug['auroc']:.4f} rec_min={aug['minority_recall']:.4f}; "
            f"naive={naive_info['generated']} generated={synth_info['generated']}"
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    lines = [
        "# RespAgent-Style Local Synthetic Validation",
        "",
        "| Task | Naive N | VAST-gen N | Real AUROC | Naive AUROC | VAST-gen AUROC | Real min-recall | Naive min-recall | VAST-gen min-recall | Real Macro-F1 | Naive Macro-F1 | VAST-gen Macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, real, naive, aug, delta_naive, delta, naive_n, gen in rows:
        lines.append(
            f"| {name} | {naive_n} | {gen} | {real['auroc']:.4f} | {naive['auroc']:.4f} | {aug['auroc']:.4f} | "
            f"{real['minority_recall']:.4f} | {naive['minority_recall']:.4f} | {aug['minority_recall']:.4f} | "
            f"{real['macro_f1']:.4f} | {naive['macro_f1']:.4f} | {aug['macro_f1']:.4f} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
