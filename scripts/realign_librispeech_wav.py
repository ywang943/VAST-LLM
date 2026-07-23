"""Rebuild LibriSpeech waveform windows to match cached mel windows exactly."""

import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset


SR = 16000
WINDOW_SAMPLES = 8 * SR
HOP_SAMPLES = 4 * SR
MEL_ROOT = Path("data/mel_cache/librispeech_100h")
WAV_ROOT = Path("data/wav_cache/librispeech_100h_aligned")


def pad_or_crop(wav):
    if len(wav) >= WINDOW_SAMPLES:
        return wav[:WINDOW_SAMPLES]
    if len(wav) == 0:
        raise ValueError("Cannot pad an empty waveform window")
    repeats = WINDOW_SAMPLES // len(wav) + 1
    return np.tile(wav, repeats)[:WINDOW_SAMPLES]


def main():
    metadata = json.loads((MEL_ROOT / "metadata.json").read_text())
    samples = metadata.get("samples", metadata)
    required = {sample["path"].replace(".pt", ".npy") for sample in samples}
    max_source_idx = max(int(name.split("_")[1]) for name in required)

    WAV_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "openslr/librispeech_asr", "clean", split="train.100", streaming=True
    )

    written = 0
    for source_idx, item in enumerate(dataset):
        if source_idx > max_source_idx:
            break

        wav = np.asarray(item["audio"]["array"], dtype=np.float32)
        sample_rate = item["audio"]["sampling_rate"]
        if sample_rate != SR:
            raise ValueError(f"Unexpected sample rate {sample_rate} at {source_idx}")

        for window_idx in (0, 1):
            name = f"librispeech_{source_idx:07d}_{window_idx}.npy"
            if name not in required:
                continue

            start = window_idx * HOP_SAMPLES
            segment = pad_or_crop(wav[start : start + WINDOW_SAMPLES])
            output = WAV_ROOT / name
            temporary = output.with_suffix(".tmp.npy")
            np.save(temporary, segment.astype(np.float32))
            os.replace(temporary, output)
            written += 1

        if (source_idx + 1) % 1000 == 0:
            print(f"Processed {source_idx + 1}/{max_source_idx + 1}; wrote {written}")

    missing = [name for name in required if not (WAV_ROOT / name).exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} aligned windows; first={missing[0]}")
    print(f"Complete: {len(required)} aligned waveform windows")


if __name__ == "__main__":
    main()
