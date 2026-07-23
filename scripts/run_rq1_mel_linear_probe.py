#!/usr/bin/env python3
"""RQ1 mel-input linear-probe evaluation with consistent probe protocol.

This script focuses on the mel-input methods whose protocol previously had two
issues: OPERA-CT was loaded with a random CSAF head, and the logistic probe used
a fixed C despite very different feature geometries. It evaluates OPERA, JEPA
w/o SIGReg, and VAST with the same pooling and train-only C selection.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from respvoice.htsat_encoder import build_htsat_encoder


S_TASKS = {
    "S1_icbhi_copd": {"mel_root": "data/mel_cache/opera_icbhi_disease", "name": "ICBHI COPD"},
    "S2_copd_severity": {"mel_root": "data/mel_cache/opera_copd", "name": "COPD Sev."},
    "S3_coswara_covid_exhale": {"mel_root": "data/mel_cache/coswara_covid_exhale", "name": "Covid Exhale"},
    "S4_coswara_covid_cough": {"mel_root": "data/mel_cache/coswara_covid_cough", "name": "Covid Cough"},
    "S5_coswara_smoker_cough": {"mel_root": "data/mel_cache/coswara_smoker_cough", "name": "Smoker Cough"},
    "S6_svd": {"mel_root": "data/mel_cache/svd_full", "name": "SVD V+S"},
    "S7_b2ai": {"mel_root": "data/mel_cache/b2ai_voice_pathology", "name": "Bridge2AI"},
}

C_GRID = [0.01, 0.1, 1.0, 10.0]


class LabeledCachedMel(Dataset):
    def __init__(self, mel_root):
        self.root = ROOT / mel_root
        meta_path = self.root / "metadata.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        samples = raw.get("samples", raw if isinstance(raw, list) else [])
        self.samples = [
            s for s in samples
            if "label" in s and (self.root / s["path"]).exists()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mel = torch.load(str(self.root / s["path"]), map_location="cpu")
        return {
            "mel": mel,
            "label": int(s["label"]),
            "split": s.get("split", "train"),
        }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch):
    return {
        "mel": torch.stack([b["mel"] for b in batch], dim=0),
        "label": np.array([b["label"] for b in batch], dtype=np.int64),
        "split": [b["split"] for b in batch],
    }


def load_vast_like_encoder(ckpt_path, device, use_csaf=True):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=True, use_csaf=use_csaf)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{ckpt_path} load mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device).eval()


def load_opera_ct(device):
    ckpt_path = ROOT / "opera_src/cks/model/encoder-operaCT.ckpt"
    # OPERA-CT checkpoint does not contain the CSAF fusion module.
    # The local encoder loader maps encoder.encoder.htsat.* directly into
    # self.htsat; loading the whole wrapper would leave duplicate htsat.* keys
    # randomly initialized.
    encoder = build_htsat_encoder(ckpt_path=str(ckpt_path), freeze_backbone=True, use_csaf=False)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device).eval()


@torch.no_grad()
def extract_task_features(encoder, task_cfg, device, batch_size, num_workers, pool_names):
    ds = LabeledCachedMel(task_cfg["mel_root"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    pools = {name: [] for name in pool_names}
    labels = []
    splits = []
    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        z = encoder(mel)
        mean = z.mean(dim=1)
        if "mean" in pools:
            pools["mean"].append(mean.cpu().numpy())
        if "mean_std" in pools or "mean_std_max" in pools:
            std = z.std(dim=1)
            if "mean_std" in pools:
                pools["mean_std"].append(torch.cat([mean, std], dim=1).cpu().numpy())
            if "mean_std_max" in pools:
                maxv = z.max(dim=1).values
                pools["mean_std_max"].append(torch.cat([mean, std, maxv], dim=1).cpu().numpy())
        labels.append(batch["label"])
        splits.extend(batch["split"])

    return {
        "features": {k: np.concatenate(v, axis=0) for k, v in pools.items()},
        "labels": np.concatenate(labels, axis=0),
        "splits": splits,
    }


def auroc_from_probs(y_true, probs):
    if len(np.unique(y_true)) == 2:
        return float(roc_auc_score(y_true, probs[:, 1]))
    return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))


def fit_eval_probe(features, labels, splits):
    splits = np.array(splits)
    train_mask = np.isin(splits, ["train", "val"])
    test_mask = splits == "test"
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    X_train_all = features[train_mask]
    y_train_all = labels[train_mask]
    X_test = features[test_mask]
    y_test = labels[test_mask]

    if len(np.unique(y_train_all)) < 2 or len(np.unique(y_test)) < 2:
        return None

    class_counts = np.bincount(y_train_all)
    class_counts = class_counts[class_counts > 0]
    n_splits = int(min(5, class_counts.min()))

    best_c = 1.0
    cv_scores = {}
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for c in C_GRID:
            scores = []
            for tr_idx, va_idx in skf.split(X_train_all, y_train_all):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train_all[tr_idx])
                X_va = scaler.transform(X_train_all[va_idx])
                clf = LogisticRegression(
                    max_iter=3000,
                    C=c,
                    solver="lbfgs",
                    class_weight="balanced",
                )
                clf.fit(X_tr, y_train_all[tr_idx])
                probs = clf.predict_proba(X_va)
                try:
                    scores.append(auroc_from_probs(y_train_all[va_idx], probs))
                except ValueError:
                    pass
            if scores:
                cv_scores[c] = float(np.mean(scores))
        if cv_scores:
            best_c = max(cv_scores, key=cv_scores.get)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_all)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(
        max_iter=3000,
        C=best_c,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(X_train, y_train_all)
    probs = clf.predict_proba(X_test)
    return {
        "auroc": auroc_from_probs(y_test, probs),
        "best_c": best_c,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="checkpoints/rq1_mel_linear_probe/results.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pools", nargs="+", default=["mean"],
                   choices=["mean", "mean_std", "mean_std_max"])
    p.add_argument("--methods", nargs="+", default=None,
                   choices=["OPERA", "JEPA w/o SIGReg", "VAST (Ours) LP"])
    p.add_argument("--c-grid", default="0.01,0.1,1,10")
    p.add_argument(
        "--vast-ckpt",
        default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt",
        help="Checkpoint for the VAST LP row. Other methods keep their own fixed checkpoints.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    global C_GRID
    C_GRID = [float(x) for x in args.c_grid.split(",") if x.strip()]

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    methods = {
        "OPERA": lambda: load_opera_ct(device),
        "JEPA w/o SIGReg": lambda: load_vast_like_encoder(
            ROOT / "checkpoints/htsat_jepa_only_d768/htsat_lejepa_best.pt", device, use_csaf=True
        ),
        "VAST (Ours) LP": lambda: load_vast_like_encoder(
            ROOT / args.vast_ckpt, device, use_csaf=True
        ),
    }
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in set(args.methods)}

    results = {
        "_meta": {
            "vast_ckpt": str(ROOT / args.vast_ckpt),
            "c_grid": C_GRID,
            "pools": args.pools,
            "seed": args.seed,
            "protocol": "frozen encoder -> sequence pooling -> standardized logistic regression with train-only C selection",
        }
    }
    for method, loader in methods.items():
        print(f"\n{'=' * 70}\n{method}\n{'=' * 70}")
        encoder = loader()
        method_out = {}
        for task_key, cfg in S_TASKS.items():
            print(f"  Extracting {task_key} ({cfg['name']})")
            data = extract_task_features(
                encoder, cfg, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pool_names=args.pools,
            )
            task_out = {}
            for pool_name in args.pools:
                feats = data["features"][pool_name]
                res = fit_eval_probe(feats, data["labels"], data["splits"])
                task_out[pool_name] = res
                if res:
                    print(
                        f"    {pool_name:<12} AUROC={res['auroc']:.4f} "
                        f"C={res['best_c']} n={res['n_train']}/{res['n_test']}"
                    )
                else:
                    print(f"    {pool_name:<12} AUROC=N/A")
            method_out[task_key] = task_out
        results[method] = method_out
        del encoder
        torch.cuda.empty_cache()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    for pool_name in args.pools:
        print(f"\nTABLE 2 MEL METHODS ({pool_name})")
        header = f"{'Method':<18}" + "".join(f"{cfg['name']:>11}" for cfg in S_TASKS.values()) + f"{'Avg':>8}"
        print(header)
        print("-" * len(header))
        for method, method_out in results.items():
            if method.startswith("_"):
                continue
            vals = []
            for task_key in S_TASKS:
                res = method_out[task_key][pool_name]
                vals.append(None if res is None else res["auroc"])
            avg = float(np.mean([v for v in vals if v is not None]))
            line = f"{method:<18}"
            for v in vals:
                line += f"{v:>11.4f}" if v is not None else f"{'—':>11}"
            line += f"{avg:>8.4f}"
            print(line)


if __name__ == "__main__":
    main()
