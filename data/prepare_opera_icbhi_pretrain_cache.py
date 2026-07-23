"""
Create an unlabeled pretraining cache from OPERA's official ICBHI train split only.

This avoids using the official ICBHI test split during self-supervised pretraining.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", default="./opera_src/feature/icbhidisease_eval")
    p.add_argument("--opera-root", default="./opera_src")
    p.add_argument("--dest", default="./data/mel_cache/opera_icbhi_train_unlabeled")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--target-sec", type=float, default=8.0)
    args = p.parse_args()

    feature_dir = Path(args.feature_dir)
    opera_root = Path(args.opera_root)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    paths = np.load(feature_dir / "sound_dir_loc.npy", allow_pickle=True)
    splits = np.load(feature_dir / "split.npy", allow_pickle=True)

    samples = []
    failures = 0
    for path_str, split in zip(paths, splits):
        if str(split) != "train":
            continue
        wav_path = opera_root / str(path_str)
        try:
            wav, _ = librosa.load(str(wav_path), sr=args.sr, mono=True)
            windows = make_windows(wav.astype(np.float32), args.sr, args.target_sec, segments_per_file=1)
            if not windows:
                failures += 1
                continue
            mel = wav_to_mel(windows[0], args.sr)
            rel = f"opera_icbhi_train_{len(samples):04d}.pt"
            torch.save(mel, dest / rel)
            samples.append({"path": rel, "source": "opera_icbhi_train", "original_path": str(path_str)})
        except Exception as exc:
            failures += 1
            print(f"skip {wav_path}: {exc}")

    with open(dest / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2)
    print(f"Done: cached={len(samples)}, failures={failures}, metadata={dest / 'metadata.json'}")


if __name__ == "__main__":
    main()
