#!/usr/bin/env python3
"""Extra RQ1 baselines for the new S1-S7 table: MARVEL-style EfficientNet and MVP HuBERT."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 16000
WAV_LEN = SR * 8
C_GRID = [0.01, 0.1, 1.0, 10.0]

S_TASKS = {
    "S1_icbhi_copd": {"mel_root": "data/mel_cache/opera_icbhi_disease", "wav_root": "data/wav_cache/opera_icbhi_disease", "name": "ICBHI COPD"},
    "S2_copd_severity": {"mel_root": "data/mel_cache/opera_copd", "wav_root": "data/wav_cache/opera_copd", "name": "COPD Sev."},
    "S3_coswara_covid_exhale": {"mel_root": "data/mel_cache/coswara_covid_exhale", "wav_root": None, "name": "Covid Exhale"},
    "S4_coswara_covid_cough": {"mel_root": "data/mel_cache/coswara_covid_cough", "wav_root": None, "name": "Covid Cough"},
    "S5_coswara_smoker_cough": {"mel_root": "data/mel_cache/coswara_smoker_cough", "wav_root": None, "name": "Smoker Cough"},
    "S6_svd": {"mel_root": "data/mel_cache/svd_full", "wav_root": "data/wav_cache/svd_full", "name": "SVD V+S"},
    "S7_b2ai": {"mel_root": "data/mel_cache/b2ai_voice_pathology", "wav_root": None, "name": "Bridge2AI"},
}


def load_samples(task_cfg):
    raw = json.loads((ROOT / task_cfg["mel_root"] / "metadata.json").read_text(encoding="utf-8"))
    return raw.get("samples", raw if isinstance(raw, list) else [])


def auroc_from_probs(y_true, probs):
    if len(np.unique(y_true)) == 2:
        return float(roc_auc_score(y_true, probs[:, 1]))
    return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))


def fit_probe(features, labels, splits):
    splits = np.array(splits)
    labels = np.array(labels)
    tr = np.isin(splits, ["train", "val"])
    te = splits == "test"
    if tr.sum() == 0 or te.sum() == 0:
        return None
    Xtr, ytr = features[tr], labels[tr]
    Xte, yte = features[te], labels[te]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    counts = np.bincount(ytr)
    counts = counts[counts > 0]
    n_splits = int(min(5, counts.min()))
    best_c = 1.0
    if n_splits >= 2:
        scores = {}
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for c in C_GRID:
            vals = []
            for a, b in skf.split(Xtr, ytr):
                scaler = StandardScaler()
                Xa = scaler.fit_transform(Xtr[a])
                Xb = scaler.transform(Xtr[b])
                clf = LogisticRegression(max_iter=3000, C=c, class_weight="balanced")
                clf.fit(Xa, ytr[a])
                try:
                    vals.append(auroc_from_probs(ytr[b], clf.predict_proba(Xb)))
                except ValueError:
                    pass
            if vals:
                scores[c] = float(np.mean(vals))
        if scores:
            best_c = max(scores, key=scores.get)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=3000, C=best_c, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return {"auroc": auroc_from_probs(yte, clf.predict_proba(Xte)), "best_c": best_c, "n_train": int(tr.sum()), "n_test": int(te.sum())}


@torch.no_grad()
def extract_marvel(model, task_cfg, device):
    samples = load_samples(task_cfg)
    mel_root = ROOT / task_cfg["mel_root"]
    feats, labels, splits = [], [], []
    for s in samples:
        if "label" not in s:
            continue
        path = mel_root / s["path"]
        if not path.exists():
            continue
        mel = torch.load(str(path), map_location="cpu")
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        x = mel.expand(3, -1, -1).unsqueeze(0)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(device)
        feats.append(model(x).squeeze(0).cpu().numpy())
        labels.append(int(s["label"]))
        splits.append(s.get("split", "train"))
    if not feats:
        return None
    return np.stack(feats), np.array(labels), splits


@torch.no_grad()
def extract_hubert(model, processor, task_cfg, device):
    if not task_cfg.get("wav_root"):
        return None
    samples = load_samples(task_cfg)
    wav_root = ROOT / task_cfg["wav_root"]
    feats, labels, splits = [], [], []
    for s in samples:
        if "label" not in s:
            continue
        wav_path = wav_root / s["path"].replace(".pt", ".npy")
        if not wav_path.exists():
            continue
        wav = np.load(str(wav_path)).astype(np.float32)
        wav = (wav - wav.mean()) / (wav.std() + 1e-8)
        if len(wav) > WAV_LEN:
            wav = wav[:WAV_LEN]
        else:
            wav = np.pad(wav, (0, WAV_LEN - len(wav)))
        inputs = processor(wav, sampling_rate=SR, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        hidden = model(**inp).last_hidden_state
        feats.append(hidden.squeeze(0).mean(dim=0).cpu().numpy())
        labels.append(int(s["label"]))
        splits.append(s.get("split", "train"))
    if not feats:
        return None
    return np.stack(feats), np.array(labels), splits


def load_efficientnet(device):
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
    model.classifier = nn.Identity()
    return model.to(device).eval()


def load_audiomae(device):
    import timm
    model = timm.create_model(
        "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k",
        pretrained=True,
        num_classes=0,
    )
    return model.to(device).eval()


@torch.no_grad()
def extract_audiomae(model, task_cfg, device):
    samples = load_samples(task_cfg)
    mel_root = ROOT / task_cfg["mel_root"]
    feats, labels, splits = [], [], []
    for s in samples:
        if "label" not in s:
            continue
        path = mel_root / s["path"]
        if not path.exists():
            continue
        mel = torch.load(str(path), map_location="cpu")
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        # AudioMAE checkpoint expects (B, 1, time=1024, mel=128).
        x = mel.transpose(1, 2).unsqueeze(0)
        x = F.interpolate(x, size=(1024, 128), mode="bilinear", align_corners=False).to(device)
        feats.append(model(x).squeeze(0).cpu().numpy())
        labels.append(int(s["label"]))
        splits.append(s.get("split", "train"))
    if not feats:
        return None
    return np.stack(feats), np.array(labels), splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="checkpoints/rq1_extra_baselines/new_table_results.json")
    p.add_argument("--skip-hubert", action="store_true")
    p.add_argument("--only", nargs="*", default=None,
                   choices=["AudioMAE", "MARVEL / Unified", "MVP (HuBERT)"])
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    only = set(args.only or [])

    if not only or "AudioMAE" in only:
        print("Loading AudioMAE...")
        audiomae = load_audiomae(device)
        row = {}
        for key, cfg in S_TASKS.items():
            data = extract_audiomae(audiomae, cfg, device)
            res = fit_probe(*data) if data else None
            row[key] = res
            print(f"AudioMAE {key}: {res['auroc']:.4f}" if res else f"AudioMAE {key}: —")
        results["AudioMAE"] = row
        del audiomae
        torch.cuda.empty_cache()

    if not only or "MARVEL / Unified" in only:
        print("Loading MARVEL/Unified EfficientNet...")
        marvel = load_efficientnet(device)
        row = {}
        for key, cfg in S_TASKS.items():
            data = extract_marvel(marvel, cfg, device)
            res = fit_probe(*data) if data else None
            row[key] = res
            print(f"MARVEL {key}: {res['auroc']:.4f}" if res else f"MARVEL {key}: —")
        results["MARVEL / Unified"] = row
        del marvel
        torch.cuda.empty_cache()

    if (not only or "MVP (HuBERT)" in only) and not args.skip_hubert:
        print("Loading MVP HuBERT...")
        from transformers import AutoFeatureExtractor, AutoModel
        processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        model = AutoModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
        row = {}
        for key, cfg in S_TASKS.items():
            data = extract_hubert(model, processor, cfg, device)
            res = fit_probe(*data) if data else None
            row[key] = res
            print(f"MVP(HuBERT) {key}: {res['auroc']:.4f}" if res else f"MVP(HuBERT) {key}: —")
        results["MVP (HuBERT)"] = row

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out}")

    header = f"{'Method':<18}" + "".join(f"{cfg['name']:>13}" for cfg in S_TASKS.values()) + f"{'Avg':>8}"
    print(header)
    print("-" * len(header))
    for method, row in results.items():
        vals = [row[k]["auroc"] if row.get(k) else None for k in S_TASKS]
        avg = np.mean([v for v in vals if v is not None])
        line = f"{method:<18}" + "".join(f"{v:>13.4f}" if v is not None else f"{'—':>13}" for v in vals) + f"{avg:>8.4f}"
        print(line)


if __name__ == "__main__":
    main()
