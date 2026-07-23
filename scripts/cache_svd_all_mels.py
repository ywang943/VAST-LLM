"""Persist mel spectrograms for every unique SVD waveform used in pretraining."""

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from respvoice.preprocessing import AudioPreprocessor


WAV_ROOT = Path("data/wav_cache/svd_all")
MEL_ROOT = Path("data/mel_cache/svd_all")


def cache_one(name):
    output = MEL_ROOT / name.replace(".npy", ".pt")
    if output.exists():
        return "existing"

    wav = np.load(WAV_ROOT / name).astype(np.float32)
    mel = AudioPreprocessor(sr=16000, target_sec=8.0).to_mel(wav)
    temporary = output.with_suffix(".tmp")
    torch.save(mel, temporary)
    os.replace(temporary, output)
    return "written"


def main():
    raw = json.loads((WAV_ROOT / "metadata.json").read_text())
    samples = raw.get("samples", raw)
    unique_names = sorted({sample["path"] for sample in samples})
    MEL_ROOT.mkdir(parents=True, exist_ok=True)

    counts = {"written": 0, "existing": 0}
    with Pool(processes=8) as pool:
        for index, status in enumerate(
            pool.imap_unordered(cache_one, unique_names, chunksize=16), start=1
        ):
            counts[status] += 1
            if index % 1000 == 0:
                print(f"Processed {index}/{len(unique_names)}: {counts}", flush=True)

    mel_samples = []
    for sample in samples:
        converted = dict(sample)
        converted["path"] = sample["path"].replace(".npy", ".pt")
        mel_samples.append(converted)
    (MEL_ROOT / "metadata.json").write_text(
        json.dumps({"samples": mel_samples}, ensure_ascii=True)
    )
    print(
        f"Complete: {len(unique_names)} unique files, {len(mel_samples)} records, {counts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
