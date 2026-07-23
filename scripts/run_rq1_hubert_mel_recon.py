#!/usr/bin/env python3
"""RQ1 MVP/HuBERT using waveform reconstructed from cached normalized log-mels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


class MelDataset(Dataset):
    def __init__(self, mel_root):
        self.root = ROOT / mel_root
        raw = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        samples = raw.get("samples", raw if isinstance(raw, list) else [])
        self.samples = [s for s in samples if "label" in s and (self.root / s["path"]).exists()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mel = torch.load(str(self.root / s["path"]), map_location="cpu").squeeze(0)
        return mel, int(s["label"]), s.get("split", "train")


def collate(batch):
    return (
        torch.stack([b[0] for b in batch], dim=0),
        np.array([b[1] for b in batch], dtype=np.int64),
        [b[2] for b in batch],
    )


def auc_probs(y, probs):
    if len(np.unique(y)) == 2:
        return float(roc_auc_score(y, probs[:, 1]))
    return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))


def fit_probe(X, y, splits):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    splits = np.array(splits)
    tr = np.isin(splits, ["train", "val"])
    te = splits == "test"
    if tr.sum() == 0 or te.sum() == 0:
        return None
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
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
                    vals.append(auc_probs(ytr[b], clf.predict_proba(Xb)))
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
    return {"auroc": auc_probs(yte, clf.predict_proba(Xte)), "best_c": best_c, "n_train": int(tr.sum()), "n_test": int(te.sum())}


@torch.no_grad()
def extract_task(model, task_cfg, device, batch_size, n_iter):
    ds = MelDataset(task_cfg["mel_root"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)
    inv = torchaudio.transforms.InverseMelScale(
        n_stft=513, n_mels=64, sample_rate=16000, f_min=50, f_max=8000
    ).to(device)
    gl = torchaudio.transforms.GriffinLim(
        n_fft=1024, win_length=1024, hop_length=512, n_iter=n_iter, power=2.0
    ).to(device)
    feats, labels, splits = [], [], []
    for i, (mel, y, sp) in enumerate(loader):
        mel = torch.exp(mel).clamp_min(1e-6).to(device)
        wav = gl(inv(mel))
        wav = wav[:, :128000]
        if wav.size(1) < 128000:
            wav = torch.nn.functional.pad(wav, (0, 128000 - wav.size(1)))
        wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-8)
        hidden = model(input_values=wav).last_hidden_state
        feat = hidden.mean(dim=1).cpu().numpy()
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        feats.append(feat)
        labels.append(y)
        splits.extend(sp)
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(loader)} batches")
    return np.concatenate(feats), np.concatenate(labels), splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="checkpoints/rq1_hubert_mel_recon/results.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-iter", type=int, default=2)
    p.add_argument("--tasks", nargs="*", default=None, choices=list(S_TASKS.keys()))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModel
    print("Loading HuBERT...")
    model = AutoModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    results = {}
    task_items = [(k, S_TASKS[k]) for k in (args.tasks or list(S_TASKS.keys()))]
    for key, cfg in task_items:
        print(f"\n{key} {cfg['name']}")
        data = extract_task(model, cfg, device, args.batch_size, args.n_iter)
        res = fit_probe(*data)
        results[key] = res
        print(f"  AUROC={res['auroc']:.4f} C={res['best_c']}" if res else "  AUROC=—")
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    vals = [r["auroc"] for r in results.values() if r]
    print(f"\nAvg={np.mean(vals):.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
