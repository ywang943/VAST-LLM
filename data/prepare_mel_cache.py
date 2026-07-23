"""
Precompute log-Mel tensors from local wav files.

This turns a few hundred long recordings into a few thousand fixed-length
training windows and avoids repeated librosa decoding during training.

Examples:
  python data/prepare_mel_cache.py --sources data/audio/icbhi data/audio/coughvid_hf data/audio/coswara_hf --dest data/mel_cache/pretrain --segments-per-file 4
  python data/prepare_mel_cache.py --sources data/audio/icbhi --dest data/mel_cache/icbhi_binary --segments-per-file 3 --binary-icbhi
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch


def load_source_metadata(source: Path):
    meta_path = source / "metadata.json"
    if not meta_path.exists():
        files = sorted(source.rglob("*.wav"))
        return [{"path": str(p.relative_to(source)), "label_name": "unknown"} for p in files]
    with open(meta_path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("samples", raw if isinstance(raw, list) else [])


def wav_to_mel(wav, sr, n_mels=64, win_ms=64.0, hop_ms=32.0):
    win_length = int(win_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_mels=n_mels,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        fmin=50.0,
        fmax=min(8000.0, sr / 2),
        power=2.0,
    )
    mel = np.log(mel + 1e-6).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return torch.from_numpy(mel).unsqueeze(0)


def make_windows(wav, sr, target_sec, segments_per_file):
    target_len = int(sr * target_sec)
    if len(wav) < 1:
        return []
    if len(wav) < target_len:
        reps = target_len // len(wav) + 1
        wav = np.tile(wav, reps)
    max_start = max(0, len(wav) - target_len)
    if segments_per_file <= 1 or max_start == 0:
        starts = [0]
    else:
        starts = np.linspace(0, max_start, segments_per_file, dtype=np.int64).tolist()
    return [wav[s : s + target_len] for s in starts]


def prepare(args):
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    samples = []
    failures = 0
    index = 0

    for source_str in args.sources:
        source = Path(source_str)
        source_name = source.name
        print(f"Source: {source}")
        for item in load_source_metadata(source):
            wav_path = source / item["path"]
            if not wav_path.exists():
                failures += 1
                continue
            try:
                wav, sr = librosa.load(str(wav_path), sr=args.sr, mono=True)
                wav = wav.astype(np.float32)
                windows = make_windows(wav, args.sr, args.target_sec, args.segments_per_file)
                for seg_idx, win in enumerate(windows):
                    meta = {
                        "source": source_name,
                        "original_path": item["path"],
                        "segment": seg_idx,
                        "label_name": item.get("label_name", "unknown"),
                    }
                    if args.icbhi_copd:
                        label_name = item.get("label_name")
                        if label_name not in {"No potential disease detected", "COPD"}:
                            continue
                        normal = label_name == "No potential disease detected"
                        meta["label"] = 0 if normal else 1
                        meta["label_name_binary"] = "healthy" if normal else "copd"
                    elif args.binary_icbhi:
                        normal = item.get("label_name") == "No potential disease detected"
                        meta["label"] = 0 if normal else 1
                        meta["label_name_binary"] = "normal" if normal else "disease"
                    elif "label" in item:
                        meta["label"] = item["label"]
                    mel = wav_to_mel(win, args.sr, n_mels=args.n_mels)
                    rel = f"{source_name}_{index:06d}.pt"
                    torch.save(mel, dest / rel)
                    meta["path"] = rel
                    samples.append(meta)
                    index += 1
            except Exception as exc:
                failures += 1
                print(f"  skip {wav_path}: {exc}")

            if index and index % 500 == 0:
                print(f"  cached {index} windows")

    meta_path = dest / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2)
    print(f"Done: cached={len(samples)}, failures={failures}, metadata={meta_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--segments-per-file", type=int, default=4)
    p.add_argument("--target-sec", type=float, default=8.0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--binary-icbhi", action="store_true")
    p.add_argument("--icbhi-copd", action="store_true",
                   help="Keep only ICBHI healthy and COPD samples; label healthy=0, COPD=1")
    args = p.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
