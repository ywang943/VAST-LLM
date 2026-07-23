"""
Export the full Zenodo CoughVID public dataset to local 16 kHz wav files.

The raw archive stores audio as webm/ogg. This script uses the ffmpeg binary
bundled by imageio-ffmpeg, so it does not require a system ffmpeg install.

Usage:
  python data/export_coughvid_zenodo.py --raw-dir data/raw/coughvid_public/public_dataset --dest data/audio/coughvid_zenodo --min-cough 0.8
"""

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def load_existing(dest: Path):
    meta_path = dest / "metadata.json"
    if not meta_path.exists():
        return [], set()
    with open(meta_path, encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw.get("samples", [])
    done = {s.get("uuid") for s in samples if s.get("uuid")}
    return samples, done


def decode_to_wav(ffmpeg: str, src: Path, out: Path, sr: int):
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-vn",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def prepare(args):
    raw_dir = Path(args.raw_dir)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "metadata_compiled.csv"
    df = pd.read_csv(csv_path)
    df["cough_detected"] = pd.to_numeric(df["cough_detected"], errors="coerce").fillna(0.0)
    df = df[df["cough_detected"] >= args.min_cough].copy()
    if args.require_status:
        df = df[df["status"].notna()].copy()
    if args.max_samples:
        df = df.head(args.max_samples).copy()

    samples, done = load_existing(dest)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    failures = 0

    print(f"Candidates: {len(df)} from {csv_path}")
    print(f"Existing exported: {len(done)}")
    print(f"ffmpeg: {ffmpeg}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CoughVID export"):
        uuid = str(row["uuid"])
        if uuid in done:
            continue
        src = raw_dir / f"{uuid}.webm"
        if not src.exists():
            src = raw_dir / f"{uuid}.ogg"
        if not src.exists():
            failures += 1
            continue

        out_name = f"coughvid_zenodo_{len(samples):06d}_{uuid}.wav"
        out_path = dest / out_name
        try:
            decode_to_wav(ffmpeg, src, out_path, args.sr)
            info = sf.info(str(out_path))
            status = row.get("status")
            label_name = "unknown" if pd.isna(status) else str(status)
            samples.append(
                {
                    "path": out_name,
                    "uuid": uuid,
                    "source": "coughvid_zenodo",
                    "label_name": label_name,
                    "sr": args.sr,
                    "duration": float(info.frames / info.samplerate),
                    "cough_detected": float(row["cough_detected"]),
                    "SNR": None if pd.isna(row.get("SNR")) else float(row.get("SNR")),
                    "status": label_name,
                    "age": None if pd.isna(row.get("age")) else str(row.get("age")),
                    "gender": None if pd.isna(row.get("gender")) else str(row.get("gender")),
                    "respiratory_condition": None
                    if pd.isna(row.get("respiratory_condition"))
                    else str(row.get("respiratory_condition")),
                }
            )
            done.add(uuid)
        except Exception as exc:
            failures += 1
            if out_path.exists():
                out_path.unlink()
            print(f"skip {uuid}: {exc}")

        if len(samples) % 1000 == 0:
            with open(dest / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({"samples": samples}, f, indent=2)

    with open(dest / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2)

    print(f"Done: exported={len(samples)}, failures={failures}, dest={dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="./data/raw/coughvid_public/public_dataset")
    p.add_argument("--dest", default="./data/audio/coughvid_zenodo")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--min-cough", type=float, default=0.8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--require-status", action="store_true")
    args = p.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
