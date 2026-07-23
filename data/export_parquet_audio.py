"""
Export local HuggingFace parquet audio snapshots to wav files.

This avoids fragile streaming downloads. It expects parquet files already
downloaded under data/hf_snapshots.
"""

import argparse
import io
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def decode_audio(audio_obj, target_sr):
    audio_bytes = None
    path = None
    if isinstance(audio_obj, dict):
        audio_bytes = audio_obj.get("bytes")
        path = audio_obj.get("path")
    if audio_bytes is None:
        raise ValueError("missing audio bytes")

    try:
        with io.BytesIO(audio_bytes) as buf:
            wav, sr = sf.read(buf, always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav.astype(np.float32), target_sr, path
    except Exception:
        with io.BytesIO(audio_bytes) as buf:
            wav, sr = librosa.load(buf, sr=target_sr, mono=True)
        return wav.astype(np.float32), target_sr, path


def load_existing(dest):
    meta_path = dest / "metadata.json"
    if not meta_path.exists():
        return [], {}, set()
    with open(meta_path, encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw.get("samples", [])
    label_map = raw.get("label_map", {})
    keys = {s.get("key") for s in samples if s.get("key")}
    return samples, label_map, keys


def save_meta(dest, samples, label_map):
    with open(dest / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples, "label_map": label_map}, f, indent=2)


def export_coughvid(snapshot, dest, max_samples, target_sr):
    parquet_files = sorted((snapshot / "data").glob("*.parquet"))
    samples, label_map, existing_keys = load_existing(dest)
    saved = len(samples)
    failures = 0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        for row_idx, row in df.iterrows():
            key = f"{pq_path.name}:{row_idx}"
            if key in existing_keys:
                continue
            if max_samples and saved >= max_samples:
                save_meta(dest, samples, label_map)
                print(f"Reached max_samples={max_samples}")
                return
            try:
                wav, sr, audio_path = decode_audio(row["audio"], target_sr)
                label = str(row.get("label", "unknown"))
                if label not in label_map:
                    label_map[label] = len(label_map)
                stem = str(row.get("file", saved)).replace("/", "_").replace("\\", "_")
                fname = f"coughvid_{saved:06d}_{stem}.wav"
                sf.write(str(dest / fname), wav, sr)
                samples.append({
                    "path": fname,
                    "key": key,
                    "source": "coughvid",
                    "label": label_map[label],
                    "label_name": label,
                    "sr": sr,
                    "duration": float(len(wav) / sr),
                    "original_path": audio_path,
                })
                existing_keys.add(key)
                saved += 1
            except Exception as exc:
                failures += 1
                print(f"skip {key}: {exc}")
            if saved and saved % 500 == 0:
                print(f"  coughvid saved {saved}")
                save_meta(dest, samples, label_map)

    save_meta(dest, samples, label_map)
    print(f"CoughVID done: saved={saved}, failures={failures}")


def export_coswara(snapshot, dest, max_samples, target_sr, min_quality):
    parquet_files = sorted((snapshot / "audio").rglob("*.parquet"))
    samples, label_map, existing_keys = load_existing(dest)
    saved = len(samples)
    failures = 0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        for row_idx, row in df.iterrows():
            key = f"{pq_path.relative_to(snapshot)}:{row_idx}"
            if key in existing_keys:
                continue
            if max_samples and saved >= max_samples:
                save_meta(dest, samples, label_map)
                print(f"Reached max_samples={max_samples}")
                return
            quality = row.get("quality_score")
            quality_is_nan = quality is None or (isinstance(quality, float) and np.isnan(quality))
            if min_quality is not None and not quality_is_nan and int(quality) < min_quality:
                continue
            try:
                wav, sr, audio_path = decode_audio(row["audio"], target_sr)
                label = str(row.get("covid_status", "unknown"))
                if label not in label_map:
                    label_map[label] = len(label_map)
                audio_type = str(row.get("audio_type", "audio"))
                pid = str(row.get("participant_id", saved)).replace("/", "_").replace("\\", "_")
                fname = f"coswara_{saved:06d}_{audio_type}_{pid}.wav"
                sf.write(str(dest / fname), wav, sr)
                samples.append({
                    "path": fname,
                    "key": key,
                    "source": "coswara",
                    "label": label_map[label],
                    "label_name": label,
                    "audio_type": audio_type,
                    "quality_score": None if quality_is_nan else int(quality),
                    "sr": sr,
                    "duration": float(len(wav) / sr),
                    "original_path": audio_path,
                })
                existing_keys.add(key)
                saved += 1
            except Exception as exc:
                failures += 1
                print(f"skip {key}: {exc}")
            if saved and saved % 500 == 0:
                print(f"  coswara saved {saved}")
                save_meta(dest, samples, label_map)

    save_meta(dest, samples, label_map)
    print(f"Coswara done: saved={saved}, failures={failures}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["coughvid", "coswara"], required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--max-samples", type=int, default=0, help="0 means all")
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--min-quality", type=int, default=None)
    args = p.parse_args()

    snapshot = Path(args.snapshot)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    max_samples = args.max_samples or None

    if args.dataset == "coughvid":
        export_coughvid(snapshot, dest, max_samples, args.target_sr)
    else:
        export_coswara(snapshot, dest, max_samples, args.target_sr, args.min_quality)


if __name__ == "__main__":
    main()
