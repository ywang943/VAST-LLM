"""
Download respiratory and voice datasets from HuggingFace.
Each dataset is saved as mel windows for pretraining.

Targets:
  1. CaReSound (cardiac + respiratory auscultation)
  2. Lung cancer audio datasets
  3. Any accessible respiratory HF datasets

Usage:
    python data/download_hf_respiratory.py
"""
import io, json, sys
from pathlib import Path

import librosa, numpy as np, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets import load_dataset, Audio as HFAudio
from respvoice.preprocessing import AudioPreprocessor

SR, N_MELS, WIN_MS, HOP_MS, TARGET_SEC = 16000, 64, 64.0, 32.0, 8.0
preprocessor = AudioPreprocessor(sr=SR, n_mels=N_MELS, win_ms=WIN_MS,
                                  hop_ms=HOP_MS, target_sec=TARGET_SEC)


def get_audio_bytes(item):
    """Extract (bytes, sr) from HuggingFace dataset item."""
    audio = item.get("audio", {})
    if isinstance(audio, dict):
        b = audio.get("bytes") or audio.get("array")
        if isinstance(b, bytes) and len(b) > 0:
            return b, audio.get("sampling_rate", SR)
        elif hasattr(b, "__len__") and not isinstance(b, bytes):
            # numpy array
            return b, audio.get("sampling_rate", SR)
    return None, None


def process_audio(raw, orig_sr, n_segments=2):
    if isinstance(raw, bytes):
        with io.BytesIO(raw) as buf:
            arr, sr = sf.read(buf)
    else:
        arr, sr = np.array(raw), orig_sr
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != SR:
        arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=SR)
    target_len = int(TARGET_SEC * SR)
    segs = []
    for start in range(0, max(1, len(arr) - target_len + 1), target_len // 2):
        chunk = arr[start:start + target_len]
        if len(chunk) < target_len // 4:
            break
        segs.append(preprocessor.to_mel(chunk.astype(np.float32)))
        if len(segs) >= n_segments:
            break
    if not segs:
        segs.append(preprocessor.to_mel(arr.astype(np.float32)))
    return segs


def download_dataset(ds_id, ds_kwargs, dest_name, label_key=None, max_samples=None):
    dest = Path(f"data/mel_cache/{dest_name}")
    dest.mkdir(parents=True, exist_ok=True)
    meta_path = dest / "metadata.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        print(f"  {dest_name}: already has {len(existing['samples'])} windows, skipping")
        return len(existing['samples'])

    print(f"\nDownloading {dest_name} from {ds_id}...")
    try:
        ds = load_dataset(ds_id, **{k: v for k, v in ds_kwargs.items()
                                     if k != "audio_col"})
        audio_col = ds_kwargs.get("audio_col", "audio")
        ds = ds.cast_column(audio_col, HFAudio(decode=False))
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return 0

    samples, cached, failures, idx = [], 0, 0, 0
    for item in ds:
        if max_samples and idx >= max_samples:
            break
        try:
            raw, sr = get_audio_bytes(item if "audio" in item
                                       else {"audio": item.get(audio_col, {})})
            if raw is None:
                idx += 1; continue
            segs = process_audio(raw, sr)
            label = str(item.get(label_key, "unknown")) if label_key else "unlabeled"
            for si, mel in enumerate(segs):
                fname = f"{dest_name}_{idx:07d}_{si}.pt"
                torch.save(mel, str(dest / fname))
                samples.append({"path": fname, "label_name": label})
                cached += 1
        except Exception as e:
            failures += 1
            if failures <= 2:
                print(f"  Warning [{idx}]: {e}")
        idx += 1
        if idx % 500 == 0:
            print(f"  {idx} items → {cached} windows saved")

    meta = {"samples": samples, "source": ds_id, "dest": dest_name}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Done: {cached} windows from {idx} items (failures={failures})")
    return cached


def main():
    total = 0

    # 1. CaReSound - cardiac + respiratory auscultation
    total += download_dataset(
        "tsnngw/CaReSound",
        {"split": "train", "streaming": True},
        dest_name="caresound",
        label_key="label",
        max_samples=5000,
    )

    # 2. Pulmonary disease dataset
    total += download_dataset(
        "ericyxy98/pulmonary-disease-airway-lung-function-dataset",
        {"split": "train", "streaming": True},
        dest_name="pulmonary_disease",
        label_key="label",
        max_samples=3000,
    )

    # 3. Lung cancer audio
    total += download_dataset(
        "nateraw/lung-cancer",
        {"split": "train", "streaming": True},
        dest_name="lung_cancer",
        label_key="label",
        max_samples=2000,
    )

    print(f"\nTotal new windows: {total}")
    print("Run data/prepare_mel_cache.py to merge with existing caches if needed.")


if __name__ == "__main__":
    main()
