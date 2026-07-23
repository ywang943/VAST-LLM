"""
SVD Voice Pathology — HuBERT baseline (fast, using cached waveforms).

Faithful replication of MVP (Koudounas et al., INTERSPEECH 2025):
  - HuBERT-base 94.6M, full fine-tune
  - Raw waveform 16kHz, 5s per source (10s for concat)
  - 10-fold speaker-independent CV
  - AdamW lr=5e-5, weight_decay=0.01
  - Batch 8 × grad_accum 8 = effective batch 64
  - 10 epochs, early stop patience=5
  - Data augmentation: noise/speed/pitch (25% for sentences, 10% for vowels)
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SVD_META = Path("data/mel_cache/svd_full")
WAV_CACHE = Path("data/wav_cache/svd_full")
SR = 16000
MAX_SEC = 5.0


def collect_subjects():
    meta = json.loads((SVD_META / "metadata.json").read_text())
    by_subj = defaultdict(dict)
    labels = {}
    for item in meta["samples"]:
        subj = item["subject_id"]
        labels[subj] = int(item["label"])
        npy_name = item["path"].replace(".pt", ".npy")
        by_subj[subj][item["source"]] = str(WAV_CACHE / npy_name)
    subjects = []
    for subj, parts in by_subj.items():
        if "vowel" in parts and "phrase" in parts:
            subjects.append({"subj": subj, "label": labels[subj],
                             "vowel": parts["vowel"], "phrase": parts["phrase"]})
    return subjects


class SVDWavDataset(Dataset):
    def __init__(self, subjects, indices, fe, fusion="concat", augment=False):
        self.subjects = [subjects[i] for i in indices]
        self.fe = fe
        self.fusion = fusion
        self.augment = augment
        self.max_samples_per_source = int(SR * MAX_SEC)

    def __len__(self):
        return len(self.subjects)

    def _pad_or_trim(self, audio, target_len):
        if len(audio) >= target_len:
            return audio[:target_len]
        return np.pad(audio, (0, target_len - len(audio)))

    def _augment_vowel(self, audio):
        """Vowel augmentation: 10% probability, minimal."""
        if random.random() > 0.10:
            return audio
        aug = random.choice(["noise", "pitch"])
        if aug == "noise":
            audio = audio + np.random.normal(0, 0.003, len(audio))
        elif aug == "pitch":
            audio = librosa.effects.pitch_shift(audio, sr=SR,
                                                 n_steps=random.uniform(-1, 1))
        return audio.astype(np.float32)

    def _augment_phrase(self, audio):
        """Phrase augmentation: 25% probability, stronger."""
        if random.random() > 0.25:
            return audio
        aug = random.choice(["noise", "speed_up", "speed_down", "pitch", "combo"])
        if aug == "noise":
            audio = audio + np.random.normal(0, 0.005, len(audio))
        elif aug == "speed_up":
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(1.05, 1.25))
        elif aug == "speed_down":
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.75, 0.95))
        elif aug == "pitch":
            audio = librosa.effects.pitch_shift(audio, sr=SR,
                                                 n_steps=random.uniform(-4, 4))
        elif aug == "combo":
            audio = audio + np.random.normal(0, 0.005, len(audio))
            audio = librosa.effects.pitch_shift(audio, sr=SR,
                                                 n_steps=random.uniform(-3, 3))
        return audio.astype(np.float32)

    def __getitem__(self, idx):
        s = self.subjects[idx]
        vowel = np.load(s["vowel"]).astype(np.float32)
        phrase = np.load(s["phrase"]).astype(np.float32)

        vowel = self._pad_or_trim(vowel, self.max_samples_per_source)
        phrase = self._pad_or_trim(phrase, self.max_samples_per_source)

        if self.augment:
            vowel = self._augment_vowel(vowel)
            phrase = self._augment_phrase(phrase)

        if self.fusion == "concat":
            audio = np.concatenate([vowel, phrase])
        elif self.fusion == "vowel_only":
            audio = vowel
        else:
            audio = phrase

        # Normalize
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)

        max_len = self.max_samples_per_source * (2 if self.fusion == "concat" else 1)
        inputs = self.fe(
            audio, sampling_rate=SR, return_tensors="pt",
            max_length=max_len, truncation=True, padding="max_length",
        )
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }


def train_one_fold(subjects, train_idx, val_idx, test_idx,
                   args, device, fold_idx):
    random.seed(fold_idx)
    np.random.seed(fold_idx)
    torch.manual_seed(fold_idx)
    torch.cuda.manual_seed_all(fold_idx)

    train_ds = SVDWavDataset(subjects, train_idx, args._fe, args.fusion, augment=True)
    val_ds = SVDWavDataset(subjects, val_idx, args._fe, args.fusion, augment=False)
    test_ds = SVDWavDataset(subjects, test_idx, args._fe, args.fusion, augment=False)

    g = torch.Generator().manual_seed(fold_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4)

    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name, num_labels=2,
    ).to(device)

    if fold_idx == 0:
        print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    labels_arr = [subjects[i]["label"] for i in train_idx]
    counts = torch.bincount(torch.tensor(labels_arr), minlength=2).float()
    weights = (counts.sum() / (counts.clamp_min(1) * 2)).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc, best_state, no_improve = -1.0, None, 0
    grad_accum = args.grad_accum

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss, n_step = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            iv = batch["input_values"].to(device)
            lb = batch["labels"].to(device)
            outputs = model(input_values=iv)
            loss = F.cross_entropy(outputs.logits, lb, weight=weights)
            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(train_loader) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum
            n_step += 1

        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in val_loader:
                iv = batch["input_values"].to(device)
                out = model(input_values=iv)
                vp.append(F.softmax(out.logits, dim=1)[:, 1].cpu())
                vl.append(batch["labels"])
        vp = torch.cat(vp).numpy()
        vl = torch.cat(vl).numpy()
        try:
            val_auc = roc_auc_score(vl, vp)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"    Ep {epoch+1}: loss={epoch_loss/n_step:.4f}, "
              f"val={val_auc:.4f}, best={best_auc:.4f}")

        if no_improve >= args.patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    tp, tl = [], []
    with torch.no_grad():
        for batch in test_loader:
            iv = batch["input_values"].to(device)
            out = model(input_values=iv)
            tp.append(F.softmax(out.logits, dim=1)[:, 1].cpu())
            tl.append(batch["labels"])
    tp = torch.cat(tp).numpy()
    tl = torch.cat(tl).numpy()
    try:
        test_auc = roc_auc_score(tl, tp)
    except ValueError:
        test_auc = 0.5

    print(f"  Fold {fold_idx}: test={test_auc:.4f} (val={best_auc:.4f})")
    del model
    torch.cuda.empty_cache()
    return test_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="facebook/hubert-base-ls960")
    p.add_argument("--fusion", default="concat",
                   choices=["concat", "vowel_only", "phrase_only"])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}, Fusion: {args.fusion}")
    print(f"Batch: {args.batch_size} × {args.grad_accum} = {args.batch_size*args.grad_accum}")

    # Load feature extractor once
    fe = AutoFeatureExtractor.from_pretrained(args.model_name)
    args._fe = fe

    subjects = collect_subjects()
    y = np.array([s["label"] for s in subjects])
    groups = np.array([s["subj"] for s in subjects])
    print(f"Subjects: {len(subjects)} (H={(y==0).sum()}, P={(y==1).sum()})")

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=1337)
    fold_aucs = []

    for fold_idx, (tv_idx, test_idx) in enumerate(skf.split(y, y, groups)):
        n_val = max(1, int(len(tv_idx) * 0.15))
        perm = torch.randperm(len(tv_idx),
                              generator=torch.Generator().manual_seed(fold_idx)).numpy()
        train_idx = tv_idx[perm[n_val:]]
        val_idx = tv_idx[perm[:n_val]]

        print(f"\nFold {fold_idx}: tr={len(train_idx)} val={len(val_idx)} te={len(test_idx)}")
        auc = train_one_fold(subjects, train_idx, val_idx, test_idx,
                             args, device, fold_idx)
        fold_aucs.append(auc)

    m, s = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    print(f"\n{'='*55}")
    print(f"  HuBERT: {m:.4f} ± {s:.4f}")
    print(f"  MVP ref: 0.958")
    print(f"{'='*55}")

    out = {
        "model": args.model_name, "fusion": args.fusion,
        "protocol": "HuBERT fine-tune, raw waveform, 10-fold CV",
        "n_subjects": len(subjects),
        "batch_effective": args.batch_size * args.grad_accum,
        "lr": args.lr, "epochs": args.epochs,
        "auc_mean": round(m, 4), "auc_std": round(s, 4),
        "per_fold": [round(a, 4) for a in fold_aucs],
        "mvp_reference": 0.958,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
