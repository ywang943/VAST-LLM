"""
Download LibriSpeech train-clean-100 (100 hours, clean English speech)
and prepare mel cache for voice pretraining.

This adds ~42K mel windows of clean speech to the pretraining pool,
dramatically increasing voice modality coverage.

Usage:
    python data/download_librispeech.py
    python data/download_librispeech.py --max-samples 5000  # quick subset
"""

import argparse
import io
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from respvoice.preprocessing import AudioPreprocessor

SR = 16000
N_MELS = 64
WIN_MS = 64.0
HOP_MS = 32.0
TARGET_SEC = 8.0
SEGMENTS_PER_FILE = 2   # 2 × 8s = 16s used per 10s LibriSpeech clip


def wav_bytes_to_mel(audio_array, orig_sr: int, preprocessor: AudioPreprocessor):
    """Convert audio array to mel tensor."""
    # Resample if needed
    if orig_sr != SR:
        wav = librosa.resample(audio_array.astype(np.float32), orig_sr=orig_sr, target_sr=SR)
    else:
        wav = audio_array.astype(np.float32)

    # Extract multiple 8s segments
    target_len = int(TARGET_SEC * SR)
    segments = []
    for start in range(0, max(1, len(wav) - target_len + 1), target_len // 2):
        chunk = wav[start: start + target_len]
        if len(chunk) < target_len // 2:
            break
        mel = preprocessor.to_mel(chunk)   # (1, 64, T)
        segments.append(mel)
        if len(segments) >= SEGMENTS_PER_FILE:
            break

    if not segments:
        # pad and use full clip
        segments.append(preprocessor.to_mel(wav))
    return segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="./data/mel_cache/librispeech_100h")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap number of audio clips (for quick test runs)")
    p.add_argument("--split", default="train.100",
                   choices=["train.100", "train.360", "validation"],
                   help="train.100=100h, train.360=360h, validation=5h")
    args = p.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    preprocessor = AudioPreprocessor(
        sr=SR, n_mels=N_MELS, win_ms=WIN_MS, hop_ms=HOP_MS, target_sec=TARGET_SEC
    )

    print(f"Loading LibriSpeech {args.split} from HuggingFace (streaming)...")
    from datasets import load_dataset, Audio
    ds = load_dataset(
        "openslr/librispeech_asr",
        name="clean",
        split=args.split,
        streaming=True,
    )
    # Decode audio manually with soundfile (avoid torchcodec DLL issues on Windows)
    from datasets import Audio as HFAudio
    ds = ds.cast_column("audio", HFAudio(decode=False))

    # Check existing files to support resume
    existing = set(p.name for p in dest.glob("*.pt"))
    print(f"  Resuming: {len(existing)} existing windows, continuing download...")

    samples = []
    cached = 0
    failures = 0
    idx = 0

    for item in ds:
        if args.max_samples and idx >= args.max_samples:
            break

        try:
            # Skip already-cached clips
            fname_check = f"librispeech_{idx:07d}_0.pt"
            if fname_check in existing:
                idx += 1
                continue

            audio = item["audio"]
            import io as _io
            audio_bytes = audio.get("bytes") or b""
            if not audio_bytes:
                idx += 1; continue
            with _io.BytesIO(audio_bytes) as buf:
                arr, orig_sr = sf.read(buf)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            speaker = item.get("speaker_id", idx)

            segments = wav_bytes_to_mel(arr, orig_sr, preprocessor)
            for si, mel in enumerate(segments):
                fname = f"librispeech_{idx:07d}_{si}.pt"
                torch.save(mel, str(dest / fname))
                samples.append({"path": fname, "label_name": "speech",
                                 "speaker": str(speaker)})
                cached += 1
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"  Warning [{idx}]: {e}")

        idx += 1
        if idx % 1000 == 0:
            print(f"  Processed {idx} clips → {cached} windows saved "
                  f"(failures={failures})")

    # Save metadata
    meta = {"samples": samples, "source": "librispeech", "split": args.split}
    with open(dest / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {cached} mel windows from {idx} clips → {dest}")
    print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
