#!/usr/bin/env python3
"""
RQ3 audio-only baselines.

Frozen pretrained audio encoders + task-specific logistic heads.
This follows the same source-to-target mapping used by RQ3:
  - ICBHI train -> T1 ICBHI test
  - B2AI pathology train -> T2/T3/T4 B2AI target tasks
  - Coswara cough train -> T5 breathing targets

No DMS text and no LLM are used here.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from respvoice.htsat_encoder import build_htsat_encoder
from scripts.run_rq3_llm import TASKS


EVAL_COLUMNS = {
    "T1_ICBHI": "icbhi_copd",
    "T2_LaryngealCancer": "b2ai_laryngeal_cancer",
    "T3_BenignLesions": "b2ai_benign_lesions",
    "T4_LaryngealDystonia": "b2ai_laryngeal_dystonia",
    "T5a_CoswaraCOVIDBreathing": "coswara_covid_breathing",
    "T5b_CoswaraSmokerBreathing": "coswara_smoker_breathing",
}

SOURCE_FOR_TARGET = {
    "icbhi_copd": "icbhi_copd",
    "b2ai_laryngeal_cancer": "b2ai_voice_pathology",
    "b2ai_benign_lesions": "b2ai_voice_pathology",
    "b2ai_laryngeal_dystonia": "b2ai_voice_pathology",
    "coswara_covid_breathing": "coswara_covid_cough",
    "coswara_smoker_breathing": "coswara_smoker_cough",
}

TARGET_EXCLUDE_DIAGNOSES = {
    "b2ai_laryngeal_cancer": {"laryngeal_cancer", "precancerous_lesions"},
    "b2ai_benign_lesions": {"benign_lesions"},
    "b2ai_laryngeal_dystonia": {"laryngeal_dystonia"},
}

NEEDED_TASKS = sorted(set(EVAL_COLUMNS.values()) | set(SOURCE_FOR_TARGET.values()))


class MelDataset(Dataset):
    def __init__(self, task_key):
        cfg = TASKS[task_key]
        self.task_key = task_key
        self.mel_dir = ROOT / cfg["mel_root"]
        meta_path = self.mel_dir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        self.samples = [s for s in meta.get("samples", []) if "label" in s]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mel = torch.load(self.mel_dir / sample["path"], map_location="cpu", weights_only=True)
        pid = sample.get("participant_id", sample.get("pid", ""))
        diagnosis = sample.get("diagnosis", "")
        return (
            mel.float(),
            int(sample["label"]),
            sample.get("split", "train"),
            sample["path"],
            str(pid),
            str(diagnosis),
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jepa_encoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=True, use_csaf=True)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(f"Loaded JEPA encoder: missing={len(missing)}, unexpected={len(unexpected)}")
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_opera_ct_encoder(ckpt_path, device):
    encoder = build_htsat_encoder(ckpt_path=ckpt_path, freeze_backbone=True, use_csaf=False)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def extract_task_features(task_key, encoder, device, batch_size):
    ds = MelDataset(task_key)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labels, splits, paths, pids, diagnoses = [], [], [], [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for mel, y, split, path, pid, diagnosis in loader:
            mel = mel.to(device, non_blocking=True)
            z = encoder(mel)
            feat = z.float().mean(dim=1).cpu().numpy()
            feats.append(feat)
            labels.extend([int(v) for v in y])
            splits.extend(list(split))
            paths.extend(list(path))
            pids.extend(list(pid))
            diagnoses.extend(list(diagnosis))
    return {
        "X": np.concatenate(feats, axis=0),
        "y": np.asarray(labels, dtype=np.int64),
        "split": np.asarray(splits),
        "path": np.asarray(paths),
        "pid": np.asarray(pids, dtype=object),
        "diagnosis": np.asarray(diagnoses, dtype=object),
    }


def train_head(X, y, seed):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    clf.fit(X, y)
    return clf


def score_binary(clf, X, y):
    probs = clf.predict_proba(X)
    classes = list(clf.named_steps["logisticregression"].classes_)
    if 1 in classes:
        pos_idx = classes.index(1)
    else:
        pos_idx = len(classes) - 1
    scores = probs[:, pos_idx]
    pred = clf.predict(X)
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = float("nan")
    return {
        "auroc": float(auc),
        "accuracy": float(accuracy_score(y, pred)),
        "n": int(len(y)),
    }


def get_train_mask(features, source, target, strict):
    data = features[source]
    if source == "icbhi_copd":
        mask = data["split"] == "train"
    else:
        mask = np.isin(data["split"], ["train", "val"])

    if not strict:
        return mask

    if source == "b2ai_voice_pathology":
        exclude_diag = TARGET_EXCLUDE_DIAGNOSES.get(target, set())
        if exclude_diag:
            mask &= ~np.isin(data["diagnosis"], list(exclude_diag))
        target_pids = set(features[target]["pid"].tolist())
        if target_pids:
            mask &= ~np.isin(data["pid"], list(target_pids))
    return mask


def get_eval_mask(features, target, source, strict):
    data = features[target]
    if target == source:
        return data["split"] == "test"
    if strict and target.startswith("coswara_"):
        return data["split"] == "test"
    return np.ones(len(data["y"]), dtype=bool)


def run_method(method, ckpt_path, device, batch_size, seed, strict):
    if method == "jepa_only":
        encoder = load_jepa_encoder(ckpt_path, device)
    elif method == "opera_ct":
        encoder = load_opera_ct_encoder(ckpt_path, device)
    else:
        raise ValueError(method)

    print(f"\nExtracting features for {method}...")
    features = {}
    for task_key in NEEDED_TASKS:
        print(f"  {task_key}")
        features[task_key] = extract_task_features(task_key, encoder, device, batch_size)

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {}
    for col, target in EVAL_COLUMNS.items():
        source = SOURCE_FOR_TARGET[target]
        source_data = features[source]
        train_mask = get_train_mask(features, source, target, strict)
        X_train, y_train = source_data["X"][train_mask], source_data["y"][train_mask]
        print(
            f"  Training head {source} -> {target}: "
            f"n={len(y_train)}, labels={np.bincount(y_train)}"
        )
        clf = train_head(X_train, y_train, seed)

        target_data = features[target]
        eval_mask = get_eval_mask(features, target, source, strict)
        res = score_binary(clf, target_data["X"][eval_mask], target_data["y"][eval_mask])
        results[col] = res
        print(f"  {col}: AUROC={res['auroc']:.3f}, ACC={res['accuracy']:.3f}, n={res['n']}")

    aucs = [v["auroc"] for v in results.values() if not np.isnan(v["auroc"])]
    results["Avg_T1_T5b"] = {"auroc": float(np.mean(aucs)), "n": int(sum(v["n"] for v in results.values()))}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["jepa_only", "opera_ct"])
    parser.add_argument("--jepa-ckpt", default="checkpoints/htsat_jepa_only_d768/htsat_lejepa_best.pt")
    parser.add_argument("--opera-ckpt", default="checkpoints/opera_cache/encoder-operaCT.ckpt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true",
                        help="Use participant/disease-disjoint target protocol where possible")
    parser.add_argument("--output", default="checkpoints/rq3_baselines/rq3_audio_baselines.json")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    all_results = {}
    for method in args.methods:
        print("\n" + "=" * 80)
        print(f"METHOD: {method}")
        print("=" * 80)
        ckpt = args.jepa_ckpt if method == "jepa_only" else args.opera_ckpt
        all_results[method] = run_method(
            method, ckpt, device, args.batch_size, args.seed, args.strict
        )

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": EVAL_COLUMNS,
        "source_for_target": SOURCE_FOR_TARGET,
        "strict": args.strict,
        "methods": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
