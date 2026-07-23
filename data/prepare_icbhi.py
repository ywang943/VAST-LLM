"""
Download ICBHI 2017 respiratory sound samples from HuggingFace
and save to data/audio/icbhi/ as wav files + metadata JSON.

Usage:
  python data/prepare_icbhi.py
  python data/prepare_icbhi.py --max-samples 50
"""

import os
import io
import json
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio

LABEL_MAP = {
    "No potential disease detected": 0,
    "URTI": 1,
    "LRTI": 2,
    "Bronchiolitis": 3,
    "Pneumonia": 4,
    "COPD": 5,
    "Bronchiectasis": 6,
    "Asthma": 7,
}


def prepare(dest: str = "./data/audio/icbhi", max_samples: int = None):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print("Loading ICBHI dataset from HuggingFace (DynamicSuperb)...")
    ds = load_dataset(
        "DynamicSuperb/RespiratorySoundClassification_ICBHI2017",
        split="test",
    )
    ds = ds.cast_column("audio", Audio(decode=False))  # get raw bytes, no torchcodec needed
    print(f"Total samples: {len(ds)}")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples = []
    label_counts = {}

    for i, item in enumerate(ds):
        audio_bytes = item["audio"]["bytes"]
        label_str = item["label"]
        label_int = LABEL_MAP.get(label_str, 0)

        # Decode wav bytes with soundfile
        with io.BytesIO(audio_bytes) as buf:
            wav, sr = sf.read(buf)

        # Save to disk
        fname = f"icbhi_{i:04d}.wav"
        fpath = dest / fname
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # to mono
        sf.write(str(fpath), wav.astype(np.float32), sr)

        samples.append({
            "path": fname,
            "label": label_int,
            "label_name": label_str,
            "sr": sr,
            "duration": len(wav) / sr,
        })
        label_counts[label_str] = label_counts.get(label_str, 0) + 1

        if (i + 1) % 20 == 0 or (i + 1) == len(ds):
            print(f"  [{i+1}/{len(ds)}] saved {fname}  label={label_str}  dur={len(wav)/sr:.1f}s")

    # Write metadata JSON
    meta_path = dest / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"samples": samples}, f, indent=2)

    print(f"\nDone. {len(samples)} samples saved to {dest}/")
    print("Label distribution:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl:40s}: {cnt}")
    print(f"\nMetadata: {meta_path}")
    return str(dest), str(meta_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="./data/audio/icbhi")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()
    prepare(args.dest, args.max_samples)
