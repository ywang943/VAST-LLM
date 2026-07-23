"""
Prepare CoughVID downstream evaluation caches.
Mirrors OPERA's coughvid evaluation tasks:
  1. COVID detection: healthy (0) vs COVID-19 (1)
  2. Sex detection: female (0) vs male (1)

Uses CoughVID Zenodo data already downloaded to data/audio/coughvid_zenodo/

Usage:
    python data/prepare_coughvid_downstream.py
"""

import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from respvoice.preprocessing import AudioPreprocessor

SR = 16000
N_MELS = 64
WIN_MS = 64.0
HOP_MS = 32.0
TARGET_SEC = 8.0
SEGMENTS_PER_FILE = 2

preprocessor = AudioPreprocessor(sr=SR, n_mels=N_MELS, win_ms=WIN_MS,
                                  hop_ms=HOP_MS, target_sec=TARGET_SEC)

SOURCE_DIR = Path("data/audio/coughvid_zenodo")
META_PATH  = SOURCE_DIR / "metadata.json"


def make_cache(task_name: str, label_fn, dest: Path):
    """Create mel cache for a downstream task."""
    dest.mkdir(parents=True, exist_ok=True)
    meta_src = json.loads(META_PATH.read_text())
    samples_src = meta_src.get("samples", [])

    samples_out = []
    cached, skipped = 0, 0

    for s in samples_src:
        label = label_fn(s)
        if label is None:
            skipped += 1
            continue

        wav_path = SOURCE_DIR / s["path"]
        if not wav_path.exists():
            skipped += 1
            continue

        try:
            wav, _ = librosa.load(str(wav_path), sr=SR, mono=True)
            target_len = int(TARGET_SEC * SR)

            for si in range(SEGMENTS_PER_FILE):
                start = si * target_len // 2
                chunk = wav[start:start + target_len]
                if len(chunk) < target_len // 4:
                    break
                mel = preprocessor.to_mel(chunk)
                fname = f"{task_name}_{cached:06d}.pt"
                torch.save(mel, str(dest / fname))
                samples_out.append({
                    "path": fname,
                    "label": label,
                    "label_name": str(label),
                    "split": s.get("split", "train"),  # placeholder
                    "source_path": s["path"],
                })
                cached += 1
        except Exception:
            skipped += 1

    if not samples_out:
        print(f"  No samples for {task_name}")
        return 0

    # Create train/val/test split (stratified)
    by_label = {}
    for s in samples_out:
        by_label.setdefault(s["label"], []).append(s)

    random.seed(1337)
    train, val, test = [], [], []
    for label_id, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * 0.20))
        n_val  = max(1, int(n * 0.10))
        n_train = n - n_test - n_val
        test  += items[:n_test]
        val   += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]

    for split_name, split_items in [("train", train), ("val", val), ("test", test)]:
        for s in split_items:
            s["split"] = split_name

    meta_out = {
        "task": task_name,
        "samples": samples_out,
        "label_counts": {str(k): len(v) for k, v in by_label.items()},
        "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (dest / "metadata.json").write_text(json.dumps(meta_out, indent=2))

    by_label_str = {str(k): len(v) for k, v in by_label.items()}
    print(f"  {task_name}: {cached} windows | labels: {by_label_str}")
    print(f"  split: train={len(train)} val={len(val)} test={len(test)}")
    return cached


def main():
    print("Preparing CoughVID downstream caches...")
    print()

    # Task 1: COVID detection (healthy=0 vs COVID-19=1)
    def covid_label(s):
        ln = s.get("label_name", "")
        if ln == "healthy":     return 0
        if ln == "COVID-19":    return 1
        return None  # skip symptomatic/unknown

    make_cache(
        "coughvid_covid",
        covid_label,
        Path("data/mel_cache/coughvid_covid"),
    )

    # Task 2: Sex detection (female=0 vs male=1)
    def sex_label(s):
        g = str(s.get("gender", "")).lower()
        if g in ("female", "f"):  return 0
        if g in ("male", "m"):    return 1
        return None

    make_cache(
        "coughvid_sex",
        sex_label,
        Path("data/mel_cache/coughvid_sex"),
    )

    print("\nDone. New caches:")
    for d in ["coughvid_covid", "coughvid_sex"]:
        n = len(list(Path(f"data/mel_cache/{d}").glob("*.pt")))
        print(f"  data/mel_cache/{d}: {n} windows")


if __name__ == "__main__":
    main()
