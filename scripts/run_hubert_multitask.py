"""
HuBERT fine-tuning on all downstream tasks (ICBHI, COPD, KAUH).

Uses raw waveform input, same protocol as our HTS-AT frozen/adapted benchmarks
for fair comparison. 5-seed evaluation per task.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

SR = 16000
MAX_SEC = 8.0
MAX_SAMPLES = int(SR * MAX_SEC)

TASKS = {
    "icbhi_copd": {
        "name": "ICBHI Healthy-vs-COPD",
        "mel_root": "data/mel_cache/opera_icbhi_disease",
        "n_classes": 2,
        "split": "icbhi",
    },
    "copd_severity": {
        "name": "Respiratory@TR COPD Severity",
        "mel_root": "data/mel_cache/opera_copd",
        "n_classes": 5,
        "split": "metadata",
    },
    "kauh_obstructive": {
        "name": "KAUH Obstructive Disease",
        "mel_root": "data/mel_cache/opera_kauh",
        "n_classes": 2,
        "split": "metadata",
    },
}


class WavDataset(Dataset):
    def __init__(self, samples, fe, augment=False):
        self.samples = samples
        self.fe = fe
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            audio, sr = librosa.load(s["wav_path"], sr=SR)
        except Exception:
            audio = np.zeros(MAX_SAMPLES, dtype=np.float32)

        if self.augment and random.random() < 0.25:
            aug = random.choice(["noise", "speed", "pitch"])
            if aug == "noise":
                audio = audio + np.random.normal(0, 0.005, len(audio))
            elif aug == "speed":
                audio = librosa.effects.time_stretch(
                    audio, rate=random.uniform(0.85, 1.15))
            elif aug == "pitch":
                audio = librosa.effects.pitch_shift(
                    audio, sr=SR, n_steps=random.uniform(-3, 3))

        # Pad or trim
        if len(audio) >= MAX_SAMPLES:
            audio = audio[:MAX_SAMPLES]
        else:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

        audio = (audio - audio.mean()) / (audio.std() + 1e-8)
        inputs = self.fe(audio, sampling_rate=SR, return_tensors="pt",
                         max_length=MAX_SAMPLES, truncation=True, padding="max_length")
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }


def load_task_samples(task_key):
    """Load samples with original wav paths."""
    cfg = TASKS[task_key]
    mel_root = Path(cfg["mel_root"])
    meta = json.loads((mel_root / "metadata.json").read_text())
    raw_samples = meta.get("samples", meta if isinstance(meta, list) else [])

    samples = []
    for s in raw_samples:
        wav_path = s.get("original_path", "")
        if not wav_path:
            continue
        # Fix relative paths
        if not os.path.isabs(wav_path):
            wav_path = str(Path("opera_src") / wav_path)
        if not os.path.exists(wav_path):
            continue
        samples.append({
            "wav_path": wav_path,
            "label": int(s["label"]),
            "split": s.get("split", "train"),
        })
    return samples, cfg


def split_samples(samples, mode):
    if mode == "icbhi":
        trainval = [s for s in samples if s["split"] != "test"]
        test = [s for s in samples if s["split"] == "test"]
        labels = [s["label"] for s in trainval]
        train, val = train_test_split(trainval, test_size=0.2,
                                       random_state=1337, stratify=labels)
        return train, val, test

    split_map = {"train": [], "val": [], "test": []}
    for s in samples:
        split_map.setdefault(s["split"], []).append(s)
    if not split_map["val"]:
        labels = [s["label"] for s in split_map["train"]]
        train, val = train_test_split(split_map["train"], test_size=0.15,
                                       random_state=1337, stratify=labels)
        return train, val, split_map["test"]
    return split_map["train"], split_map["val"], split_map["test"]


def train_eval(train_samples, val_samples, test_samples,
               n_classes, fe, args, device, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_ds = WavDataset(train_samples, fe, augment=True)
    val_ds = WavDataset(val_samples, fe, augment=False)
    test_ds = WavDataset(test_samples, fe, augment=False)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)

    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name, num_labels=n_classes,
    ).to(device)

    labels_arr = [s["label"] for s in train_samples]
    counts = torch.bincount(torch.tensor(labels_arr), minlength=n_classes).float()
    weights = (counts.sum() / (counts.clamp_min(1) * n_classes)).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc, best_state, no_improve = -1.0, None, 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            iv = batch["input_values"].to(device)
            lb = batch["labels"].to(device)
            outputs = model(input_values=iv)
            loss = F.cross_entropy(outputs.logits, lb, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(input_values=batch["input_values"].to(device))
                probs = F.softmax(out.logits, dim=1).cpu()
                vp.append(probs)
                vl.append(batch["labels"])
        vp = torch.cat(vp).numpy()
        vl = torch.cat(vl).numpy()
        try:
            if n_classes == 2:
                val_auc = roc_auc_score(vl, vp[:, 1])
            else:
                val_auc = roc_auc_score(vl, vp, multi_class="ovr", average="macro")
        except ValueError:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    tp, tl = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(input_values=batch["input_values"].to(device))
            probs = F.softmax(out.logits, dim=1).cpu()
            tp.append(probs)
            tl.append(batch["labels"])
    tp = torch.cat(tp).numpy()
    tl = torch.cat(tl).numpy()
    try:
        if n_classes == 2:
            test_auc = roc_auc_score(tl, tp[:, 1])
        else:
            test_auc = roc_auc_score(tl, tp, multi_class="ovr", average="macro")
    except ValueError:
        test_auc = 0.5

    del model
    torch.cuda.empty_cache()
    return test_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="facebook/hubert-base-ls960")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")

    fe = AutoFeatureExtractor.from_pretrained(args.model_name)
    results = {"model": args.model_name, "tasks": {}}

    for task_key in args.tasks:
        samples, cfg = load_task_samples(task_key)
        if not samples:
            print(f"\n{cfg['name']}: no wav files found, skipping")
            continue

        train, val, test = split_samples(samples, cfg["split"])
        print(f"\n{cfg['name']}: train={len(train)}, val={len(val)}, "
              f"test={len(test)}, classes={cfg['n_classes']}")

        seed_aucs = []
        for seed in args.seeds:
            auc = train_eval(train, val, test, cfg["n_classes"],
                             fe, args, device, seed)
            print(f"  seed {seed}: AUC={auc:.4f}")
            seed_aucs.append(auc)

        m, s = float(np.mean(seed_aucs)), float(np.std(seed_aucs))
        print(f"  {cfg['name']}: {m:.4f} ± {s:.4f}")
        results["tasks"][task_key] = {
            "auroc_mean": round(m, 4),
            "auroc_std": round(s, 4),
            "per_seed": [round(a, 4) for a in seed_aucs],
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
