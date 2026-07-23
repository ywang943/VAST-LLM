"""
Create log-Mel caches for OPERA's official ICBHI disease task.

This uses OPERA's bundled files:
  opera_src/feature/icbhidisease_eval/sound_dir_loc.npy
  opera_src/feature/icbhidisease_eval/labels.npy
  opera_src/feature/icbhidisease_eval/split.npy

The downstream binary task matches OPERA's linear_evaluation_icbhidisease:
  Healthy -> 0
  COPD    -> 1

Usage:
  python data/prepare_opera_icbhi_cache.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np
import torch

from data.prepare_mel_cache import make_windows, wav_to_mel


LABEL_MAP = {"Healthy": 0, "COPD": 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", default="./opera_src/feature/icbhidisease_eval")
    p.add_argument("--opera-root", default="./opera_src")
    p.add_argument("--dest", default="./data/mel_cache/opera_icbhi_disease")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--target-sec", type=float, default=8.0)
    args = p.parse_args()

    feature_dir = Path(args.feature_dir)
    opera_root = Path(args.opera_root)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    paths = np.load(feature_dir / "sound_dir_loc.npy", allow_pickle=True)
    labels = np.load(feature_dir / "labels.npy", allow_pickle=True)
    splits = np.load(feature_dir / "split.npy", allow_pickle=True)

    samples = []
    failures = 0
    for i, (path_str, label_name, split) in enumerate(zip(paths, labels, splits)):
        label_name = str(label_name)
        if label_name not in LABEL_MAP:
            continue
        wav_path = opera_root / str(path_str)
        if not wav_path.exists():
            failures += 1
            print(f"missing: {wav_path}")
            continue
        try:
            wav, _ = librosa.load(str(wav_path), sr=args.sr, mono=True)
            windows = make_windows(wav.astype(np.float32), args.sr, args.target_sec, segments_per_file=1)
            if not windows:
                failures += 1
                continue
            mel = wav_to_mel(windows[0], args.sr)
            rel = f"opera_icbhi_{len(samples):04d}.pt"
            torch.save(mel, dest / rel)
            samples.append(
                {
                    "path": rel,
                    "source": "opera_icbhi_disease",
                    "original_path": str(path_str),
                    "label": LABEL_MAP[label_name],
                    "label_name": label_name,
                    "split": str(split),
                }
            )
        except Exception as exc:
            failures += 1
            print(f"skip {wav_path}: {exc}")

    with open(dest / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2)

    counts = {}
    for sample in samples:
        key = (sample["split"], sample["label_name"])
        counts[key] = counts.get(key, 0) + 1
    print(f"Done: cached={len(samples)}, failures={failures}, metadata={dest / 'metadata.json'}")
    for key, value in sorted(counts.items()):
        print(key, value)


if __name__ == "__main__":
    main()
