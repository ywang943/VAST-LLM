"""Prepare cached log-Mel datasets for OPERA downloadable downstream tasks.

Supported tasks:
  - copd: RespiratoryDatabase@TR, 5-class COPD severity
  - kauh: KAUH healthy vs obstructive disease (healthy=0, asthma/COPD=1)

The split logic follows OPERA's processing scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def wav_to_mel(wav, sr, n_mels=64, win_ms=64.0, hop_ms=32.0):
    win_length = int(win_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_mels=n_mels,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        fmin=50.0,
        fmax=min(8000.0, sr / 2),
        power=2.0,
    )
    mel = np.log(mel + 1e-6).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return torch.from_numpy(mel).unsqueeze(0)


def pad_or_crop(wav, sr, target_sec):
    target_len = int(sr * target_sec)
    if len(wav) < target_len:
        reps = target_len // max(1, len(wav)) + 1
        wav = np.tile(wav, reps)
    return wav[:target_len]


def load_copd_items():
    data_dir = ROOT / "opera_src" / "datasets" / "copd"
    audio_dir = data_dir / "RespiratoryDatabase@TR"
    labels_csv = data_dir / "Labels.csv"
    label_by_user = {}
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            label_by_user[row[0]] = int(row[1][-1])

    users = list(label_by_user.keys())
    user_labels = [label_by_user[u] for u in users]
    trainval, test, y_trainval, _ = train_test_split(
        users, user_labels, test_size=0.2, random_state=1337, stratify=user_labels
    )
    train, val, _, _ = train_test_split(
        trainval, y_trainval, test_size=0.2, random_state=1337, stratify=y_trainval
    )
    train, val, test = set(train), set(val), set(test)

    items = []
    for wav in sorted(audio_dir.glob("*.wav")):
        user = wav.name[:4]
        split = "train" if user in train else "val" if user in val else "test"
        items.append({"wav": wav, "label": label_by_user[user], "split": split, "user": user})
    return items


def load_kauh_items():
    audio_dir = ROOT / "opera_src" / "datasets" / "KAUH" / "AudioFiles"
    candidates = sorted(audio_dir.glob("*.wav"))
    filtered = []
    for wav in candidates:
        label = wav.name.split(",")[0].split("_")[-1]
        if label == "N":
            mapped = "healthy"
        elif "asthma" in label or "Asthma" in label:
            mapped = "asthma"
        elif "COPD" in label:
            mapped = "COPD"
        else:
            continue
        user = wav.name.split("_")[0][2:]
        filtered.append({"wav": wav, "label_name": mapped, "user": user})

    user_labels = {}
    for item in filtered:
        user_labels.setdefault(item["user"], item["label_name"])
    users = list(user_labels.keys())
    labels = [user_labels[u] for u in users]
    trainval, test, y_trainval, _ = train_test_split(
        users, labels, test_size=0.20, random_state=42, stratify=labels
    )
    train, val, _, _ = train_test_split(
        trainval, y_trainval, test_size=0.1 / (0.1 + 0.7), random_state=42, stratify=y_trainval
    )
    train, val, test = set(train), set(val), set(test)

    items = []
    for item in filtered:
        user = item["user"]
        split = "train" if user in train else "val" if user in val else "test"
        binary = 0 if item["label_name"] == "healthy" else 1
        items.append({**item, "label": binary, "split": split})
    return items


def prepare(args):
    if args.task == "copd":
        items = load_copd_items()
    elif args.task == "kauh":
        items = load_kauh_items()
    else:
        raise ValueError(args.task)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    samples = []
    counts = {}
    failures = 0
    for idx, item in enumerate(items):
        try:
            wav, _ = librosa.load(str(item["wav"]), sr=args.sr, mono=True)
            wav = pad_or_crop(wav.astype(np.float32), args.sr, args.target_sec)
            mel = wav_to_mel(wav, args.sr, n_mels=args.n_mels)
            rel = f"{args.task}_{idx:05d}.pt"
            torch.save(mel, dest / rel)
            sample = {
                "path": rel,
                "original_path": str(item["wav"]),
                "label": int(item["label"]),
                "split": item["split"],
                "user": item["user"],
            }
            if "label_name" in item:
                sample["label_name"] = item["label_name"]
            samples.append(sample)
            counts[(item["split"], int(item["label"]))] = counts.get((item["split"], int(item["label"])), 0) + 1
        except Exception as exc:
            failures += 1
            print(f"skip {item['wav']}: {exc}")

    with open(dest / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2)

    print(f"Done task={args.task} cached={len(samples)} failures={failures} dest={dest}")
    for key, value in sorted(counts.items()):
        print(f"  split={key[0]:5s} label={key[1]} count={value}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["copd", "kauh"], required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--target-sec", type=float, default=8.0)
    p.add_argument("--n-mels", type=int, default=64)
    args = p.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
