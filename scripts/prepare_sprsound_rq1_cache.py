#!/usr/bin/env python3
"""Build SPRSound caches for the RQ1 same-source linear-probe table.

Task definition:
  SPRSound Adventitious Sound Detection
  label 0: Normal
  label 1: CAS / DAS / CAS & DAS
  excluded: Poor Quality

The cache mirrors the rest of the project: 8-second mono audio at 16 kHz and
per-sample standardized log-mel tensors shaped (1, 64, 251).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent

SR = 16000
WAV_LEN = SR * 8
LABELS = {
    "Normal": 0,
    "CAS": 1,
    "DAS": 1,
    "CAS & DAS": 1,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_audio(path):
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.numel() >= WAV_LEN:
        wav = wav[:WAV_LEN]
    else:
        wav = torch.nn.functional.pad(wav, (0, WAV_LEN - wav.numel()))
    return wav.float()


def make_mel_transform():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=64,
        f_min=50,
        f_max=8000,
        power=2.0,
        center=True,
        norm=None,
    )


def wav_to_logmel(wav, mel_tf):
    mel = mel_tf(wav)
    mel = torch.log(mel + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0).to(torch.float32)


def iter_split(json_dir, wav_dir, split):
    json_dir = ROOT / json_dir
    wav_dir = ROOT / wav_dir
    for jp in sorted(json_dir.glob("*.json")):
        meta = read_json(jp)
        record = meta.get("record_annotation")
        if record not in LABELS:
            continue
        wp = wav_dir / f"{jp.stem}.wav"
        if not wp.exists():
            continue
        yield jp.stem, wp, int(LABELS[record]), split, record, meta.get("event_annotation", [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mel-root", default="data/mel_cache/sprsound_adventitious")
    p.add_argument("--wav-root", default="data/wav_cache/sprsound_adventitious")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    mel_root = ROOT / args.mel_root
    wav_root = ROOT / args.wav_root
    mel_root.mkdir(parents=True, exist_ok=True)
    wav_root.mkdir(parents=True, exist_ok=True)

    mel_tf = make_mel_transform()
    samples = []
    counts = {"train": {0: 0, 1: 0}, "test": {0: 0, 1: 0}}

    sources = [
        (
            "data/SPRSound/Classification/train_classification_json",
            "data/SPRSound/Classification/train_classification_wav",
            "train",
        ),
        (
            "data/SPRSound/BioCAS2023/test2023_json",
            "data/SPRSound/BioCAS2023/test2023_wav",
            "test",
        ),
    ]

    idx = 0
    for json_dir, wav_dir, split in sources:
        for stem, wav_path, label, split, record, events in iter_split(json_dir, wav_dir, split):
            out_name = f"sprsound_{idx:06d}.pt"
            wav_name = f"sprsound_{idx:06d}.npy"
            mel_path = mel_root / out_name
            wav_out = wav_root / wav_name
            if args.overwrite or not mel_path.exists() or not wav_out.exists():
                wav = load_audio(wav_path)
                mel = wav_to_logmel(wav, mel_tf)
                torch.save(mel, mel_path)
                np.save(wav_out, wav.numpy().astype(np.float32))
            samples.append(
                {
                    "path": out_name,
                    "wav_path": wav_name,
                    "label": label,
                    "split": split,
                    "dataset": "SPRSound",
                    "task": "adventitious_vs_normal",
                    "record_annotation": record,
                    "source_wav": str(wav_path.relative_to(ROOT)),
                    "source_json": str((ROOT / json_dir / f"{stem}.json").relative_to(ROOT)),
                    "event_annotation": events,
                }
            )
            counts[split][label] += 1
            idx += 1

    if not samples:
        raise RuntimeError("No SPRSound samples were cached.")

    (mel_root / "metadata.json").write_text(
        json.dumps({"samples": samples}, indent=2), encoding="utf-8"
    )
    (wav_root / "metadata.json").write_text(
        json.dumps({"samples": samples}, indent=2), encoding="utf-8"
    )
    print(f"Saved mel cache: {mel_root} ({len(samples)} samples)")
    print(f"Saved wav cache: {wav_root} ({len(samples)} samples)")
    print("Counts:", counts)


if __name__ == "__main__":
    sys.exit(main())
