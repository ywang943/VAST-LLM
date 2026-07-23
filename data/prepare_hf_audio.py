"""
Prepare additional HuggingFace audio datasets for RespVoice.

The script stores audio as local wav files so the rest of the project can use
the same librosa-based preprocessing path on Windows.

Examples:
  python data/prepare_hf_audio.py --dataset coughvid --max-samples 300
  python data/prepare_hf_audio.py --dataset coswara --max-samples 300
"""

import argparse
import io
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


DATASETS = {
    "coughvid": {
        "repo": "DynamicSuperb/Covid19CoughAudioClassification_CoughVid",
        "config": None,
        "split": "test",
        "dest": "./data/audio/coughvid_hf",
        "label_field": "label",
        "name_field": "file",
    },
    "coswara": {
        "repo": "szzs1693/coswara-data",
        "config": "audio",
        "split": "train",
        "dest": "./data/audio/coswara_hf",
        "label_field": "covid_status",
        "name_field": "participant_id",
    },
}


def decode_audio_bytes(audio_bytes: bytes, target_sr: int = 16000):
    """Decode arbitrary HF audio bytes into mono float32 wav."""
    try:
        with io.BytesIO(audio_bytes) as buf:
            wav, sr = sf.read(buf, always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav, target_sr
    except Exception:
        with io.BytesIO(audio_bytes) as buf:
            wav, sr = librosa.load(buf, sr=target_sr, mono=True)
        return wav.astype(np.float32), target_sr


def prepare(dataset: str, max_samples: int, target_sr: int, dest=None):
    info = DATASETS[dataset]
    dest_path = Path(dest or info["dest"])
    dest_path.mkdir(parents=True, exist_ok=True)

    meta_path = dest_path / "metadata.json"
    samples = []
    label_map = {}
    existing = 0
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            raw = json.load(f)
        samples = raw.get("samples", [])
        label_map = raw.get("label_map", {})
        existing = len(samples)
        print(f"Resuming {dataset}: found {existing} existing samples in {meta_path}")
        if existing >= max_samples:
            print(f"Already have {existing} samples >= requested {max_samples}; nothing to do.")
            return

    print(f"Loading {dataset}: {info['repo']}")
    kwargs = {"split": info["split"], "streaming": True}
    if info["config"]:
        ds = load_dataset(info["repo"], info["config"], **kwargs)
    else:
        ds = load_dataset(info["repo"], **kwargs)
    ds = ds.cast_column("audio", Audio(decode=False))

    saved = existing
    skipped = 0
    seen = 0

    for item in ds:
        if seen < existing:
            seen += 1
            continue
        if saved >= max_samples:
            break
        seen += 1
        audio = item.get("audio") or {}
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            skipped += 1
            continue

        try:
            wav, sr = decode_audio_bytes(audio_bytes, target_sr=target_sr)
        except Exception as exc:
            skipped += 1
            print(f"  skip decode failure #{saved + skipped}: {exc}")
            continue

        label_name = str(item.get(info["label_field"], "unknown"))
        if label_name not in label_map:
            label_map[label_name] = len(label_map)

        stem = str(item.get(info["name_field"], saved)).replace("/", "_").replace("\\", "_")
        fname = f"{dataset}_{saved:05d}_{stem}.wav"
        fpath = dest_path / fname
        sf.write(str(fpath), wav, sr)

        meta = {
            "path": fname,
            "label": label_map[label_name],
            "label_name": label_name,
            "sr": sr,
            "duration": float(len(wav) / sr),
            "source": dataset,
        }
        for key in ("audio_type", "instruction", "covid_status", "quality_score", "gender", "age", "country"):
            if key in item and item[key] is not None:
                meta[key] = item[key]
        samples.append(meta)
        saved += 1

        if saved % 25 == 0:
            print(f"  saved {saved}/{max_samples}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples, "label_map": label_map}, f, indent=2)

    print(f"Done: saved={saved}, skipped={skipped}, dest={dest_path}")
    print(f"Labels: {label_map}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--dest", default=None)
    args = p.parse_args()
    prepare(args.dataset, args.max_samples, args.target_sr, args.dest)


if __name__ == "__main__":
    main()
