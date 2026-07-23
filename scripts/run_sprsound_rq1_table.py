#!/usr/bin/env python3
"""Run the SPRSound S-task row with the same RQ1 frozen-LP protocol."""

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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from respvoice.htsat_encoder import build_htsat_encoder

SR = 16000
WAV_LEN = SR * 8
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


class SPRSoundDataset(Dataset):
    def __init__(self, mel_root):
        self.root = ROOT / mel_root
        raw = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.samples = [
            s for s in raw.get("samples", raw if isinstance(raw, list) else [])
            if "label" in s and (self.root / s["path"]).exists()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mel = torch.load(str(self.root / s["path"]), map_location="cpu")
        return {"mel": mel, "label": int(s["label"]), "split": s["split"], "path": s["path"]}


def collate(batch):
    return {
        "mel": torch.stack([b["mel"] for b in batch]),
        "label": np.array([b["label"] for b in batch], dtype=np.int64),
        "split": [b["split"] for b in batch],
        "path": [b["path"] for b in batch],
    }


def auroc(y, probs):
    return float(roc_auc_score(y, probs[:, 1]))


def acc_at_half(y, probs):
    pred = (probs[:, 1] >= 0.5).astype(np.int64)
    return float((pred == y).mean())


def fit_probe(features, labels, splits):
    labels = np.asarray(labels)
    splits = np.asarray(splits)
    tr = splits == "train"
    te = splits == "test"
    Xtr, ytr = features[tr], labels[tr]
    Xte, yte = features[te], labels[te]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None

    counts = np.bincount(ytr)
    n_splits = int(min(5, counts[counts > 0].min()))
    best_c = 1.0
    cv_scores = {}
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for c in C_GRID:
            vals = []
            for a, b in skf.split(Xtr, ytr):
                scaler = StandardScaler()
                Xa = scaler.fit_transform(Xtr[a])
                Xb = scaler.transform(Xtr[b])
                clf = LogisticRegression(max_iter=3000, C=c, class_weight="balanced")
                clf.fit(Xa, ytr[a])
                vals.append(auroc(ytr[b], clf.predict_proba(Xb)))
            cv_scores[str(c)] = float(np.mean(vals))
        best_c = float(max(cv_scores, key=cv_scores.get))

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=5000, C=best_c, class_weight="balanced")
    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)
    return {
        "auroc": auroc(yte, probs),
        "accuracy": acc_at_half(yte, probs),
        "best_c": best_c,
        "cv_scores": cv_scores,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "train_pos": int(ytr.sum()),
        "test_pos": int(yte.sum()),
    }


def load_vast_like(ckpt_path, device, use_csaf=True):
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ck["model_state"].items()
        if k.startswith("encoder.")
    }
    enc = build_htsat_encoder(ckpt_path=None, freeze_backbone=True, use_csaf=use_csaf)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device).eval()


def load_opera(device):
    enc = build_htsat_encoder(
        ckpt_path=str(ROOT / "opera_src/cks/model/encoder-operaCT.ckpt"),
        freeze_backbone=True,
        use_csaf=False,
    )
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device).eval()


@torch.no_grad()
def extract_htsat(encoder, ds, device, batch_size, num_workers, pool):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate, pin_memory=torch.cuda.is_available())
    feats, labels, splits = [], [], []
    for b in loader:
        z = encoder(b["mel"].to(device, non_blocking=True))
        mean = z.mean(dim=1)
        if pool == "mean":
            f = mean
        elif pool == "mean_std":
            f = torch.cat([mean, z.std(dim=1)], dim=1)
        elif pool == "mean_std_max":
            f = torch.cat([mean, z.std(dim=1), z.max(dim=1).values], dim=1)
        else:
            raise ValueError(pool)
        feats.append(f.cpu().numpy())
        labels.append(b["label"])
        splits.extend(b["split"])
    return np.concatenate(feats), np.concatenate(labels), splits


@torch.no_grad()
def extract_marvel(ds, device, batch_size=64, num_workers=4):
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
    model.classifier = nn.Identity()
    model = model.to(device).eval()
    feats, labels, splits = [], [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate, pin_memory=torch.cuda.is_available())
    for b in loader:
        x = b["mel"].expand(-1, 3, -1, -1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(device)
        feats.append(model(x).cpu().numpy())
        labels.append(b["label"])
        splits.extend(b["split"])
    del model
    torch.cuda.empty_cache()
    return np.concatenate(feats), np.concatenate(labels), splits


@torch.no_grad()
def extract_audiomae(ds, device, batch_size=32, num_workers=4):
    import timm
    model = timm.create_model(
        "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k",
        pretrained=True,
        num_classes=0,
    ).to(device).eval()
    feats, labels, splits = [], [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate, pin_memory=torch.cuda.is_available())
    for b in loader:
        x = b["mel"].transpose(2, 3)
        x = F.interpolate(x, size=(1024, 128), mode="bilinear", align_corners=False).to(device)
        feats.append(model(x).cpu().numpy())
        labels.append(b["label"])
        splits.extend(b["split"])
    del model
    torch.cuda.empty_cache()
    return np.concatenate(feats), np.concatenate(labels), splits


@torch.no_grad()
def extract_hf_wav(model_name, ds, wav_root, device, batch_size=16):
    from transformers import AutoFeatureExtractor, AutoModel
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    wav_root = ROOT / wav_root
    feats, labels, splits = [], [], []
    batch_wavs, batch_labels, batch_splits = [], [], []

    def flush():
        if not batch_wavs:
            return
        inputs = processor(batch_wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        out = model(**inp).last_hidden_state
        feats.append(out.mean(dim=1).cpu().numpy())
        labels.append(np.array(batch_labels, dtype=np.int64))
        splits.extend(batch_splits)
        batch_wavs.clear()
        batch_labels.clear()
        batch_splits.clear()

    for s in ds.samples:
        wp = wav_root / s["wav_path"]
        if not wp.exists():
            continue
        wav = np.load(str(wp)).astype(np.float32)
        wav = (wav - wav.mean()) / (wav.std() + 1e-8)
        batch_wavs.append(wav)
        batch_labels.append(int(s["label"]))
        batch_splits.append(s["split"])
        if len(batch_wavs) >= batch_size:
            flush()
    flush()
    del model
    torch.cuda.empty_cache()
    if not feats:
        return None
    return np.concatenate(feats), np.concatenate(labels), splits


def run_method(method, ds, args, device):
    if method == "OPERA":
        enc = load_opera(device)
        data = extract_htsat(enc, ds, device, args.batch_size, args.num_workers, args.pool)
        del enc
        torch.cuda.empty_cache()
        return fit_probe(*data)
    if method == "JEPA w/o SIGReg":
        enc = load_vast_like(ROOT / "checkpoints/htsat_jepa_only_d768/htsat_lejepa_best.pt", device, True)
        data = extract_htsat(enc, ds, device, args.batch_size, args.num_workers, args.pool)
        del enc
        torch.cuda.empty_cache()
        return fit_probe(*data)
    if method == "VAST (Ours) LP":
        enc = load_vast_like(ROOT / args.vast_ckpt, device, True)
        data = extract_htsat(enc, ds, device, args.batch_size, args.num_workers, args.pool)
        del enc
        torch.cuda.empty_cache()
        return fit_probe(*data)
    if method == "MARVEL / Unified":
        return fit_probe(*extract_marvel(ds, device, args.batch_size, args.num_workers))
    if method == "AudioMAE":
        return fit_probe(*extract_audiomae(ds, device, args.batch_size, args.num_workers))
    if method == "MVP (HuBERT)":
        data = extract_hf_wav("facebook/hubert-base-ls960", ds, args.wav_root, device, max(1, args.batch_size // 4))
        return None if data is None else fit_probe(*data)
    if method == "Wav2vec-BERT":
        data = extract_hf_wav("facebook/w2v-bert-2.0", ds, args.wav_root, device, max(1, args.batch_size // 8))
        return None if data is None else fit_probe(*data)
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mel-root", default="data/mel_cache/sprsound_adventitious")
    p.add_argument("--wav-root", default="data/wav_cache/sprsound_adventitious")
    p.add_argument("--output", default="checkpoints/sprsound_rq1/results.json")
    p.add_argument("--pool", default="mean", choices=["mean", "mean_std", "mean_std_max"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--vast-ckpt", default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt")
    p.add_argument("--methods", nargs="+", default=[
        "Wav2vec-BERT", "AudioMAE", "MARVEL / Unified", "MVP (HuBERT)",
        "OPERA", "JEPA w/o SIGReg", "VAST (Ours) LP",
    ])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ds = SPRSoundDataset(args.mel_root)
    labels = np.array([int(s["label"]) for s in ds.samples])
    splits = np.array([s["split"] for s in ds.samples])
    print(
        f"SPRSound n={len(ds)} train={int((splits=='train').sum())} "
        f"test={int((splits=='test').sum())} train_pos={int(labels[splits=='train'].sum())} "
        f"test_pos={int(labels[splits=='test'].sum())}"
    )

    results = {
        "_meta": {
            "task": "SPRSound Adventitious vs Normal",
            "positive": "CAS/DAS/CAS & DAS",
            "negative": "Normal",
            "excluded": "Poor Quality",
            "pool": args.pool,
            "mel_root": args.mel_root,
            "wav_root": args.wav_root,
        }
    }
    for method in args.methods:
        print(f"\n=== {method} ===", flush=True)
        try:
            res = run_method(method, ds, args, device)
            results[method] = res
            print(res if res is None else f"AUROC={res['auroc']:.4f} ACC={res['accuracy']:.4f} C={res['best_c']}")
        except Exception as e:
            results[method] = {"error": repr(e)}
            print(f"ERROR: {e}", flush=True)

        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved: {ROOT / args.output}")
    print("\nSPRSound RQ1")
    print(f"{'Method':<20}{'AUROC':>10}{'ACC':>10}{'n train/test':>16}")
    print("-" * 56)
    for method in args.methods:
        res = results.get(method)
        if not res or "error" in res:
            print(f"{method:<20}{'—':>10}{'—':>10}{'—':>16}")
        else:
            print(f"{method:<20}{res['auroc']:>10.4f}{res['accuracy']:>10.4f}{str(res['n_train'])+'/'+str(res['n_test']):>16}")


if __name__ == "__main__":
    main()
