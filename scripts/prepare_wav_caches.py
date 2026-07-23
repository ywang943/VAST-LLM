"""
Prepare raw waveform caches (.npy) for datasets that have local wav files.

Creates wav_cache/<dataset>/ directories with .npy files aligned to mel_cache.
For datasets without local wav (LibriSpeech), training falls back to mel-only.
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 16000
TARGET_SEC = 8.0
TARGET_LEN = int(SR * TARGET_SEC)


def pad_or_crop(wav, target_len):
    if len(wav) >= target_len:
        return wav[:target_len]
    reps = target_len // len(wav) + 1
    return np.tile(wav, reps)[:target_len]


def cache_coswara():
    """Coswara: sorted wav files → sequential mel indices."""
    wav_dir = Path("data/audio/coswara_hf")
    mel_dir = Path("data/mel_cache/coswara_hf")
    out_dir = Path("data/wav_cache/coswara_hf")
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_dir.glob("*.wav"))
    mel_meta = json.loads((mel_dir / "metadata.json").read_text())
    mel_samples = mel_meta.get("samples", [])

    print(f"Coswara: {len(wav_files)} wav, {len(mel_samples)} mel")

    # mel cache was built with segments_per_file, need to reconstruct mapping
    # Each wav → multiple mel windows. We just cache the full wav per mel index.
    # The mel cache script generates windows from each wav, so we need the
    # wav-to-mel mapping. Since metadata doesn't store it, we'll cache each
    # wav as a single file and let the dataset handle windowing.

    done, fail = 0, 0
    for i, wf in enumerate(wav_files):
        out_path = out_dir / f"coswara_{i:05d}.npy"
        if out_path.exists():
            done += 1
            continue
        try:
            wav, _ = librosa.load(str(wf), sr=SR, mono=True)
            wav = pad_or_crop(wav, TARGET_LEN)
            np.save(str(out_path), wav.astype(np.float32))
            done += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print(f"  FAIL {wf.name}: {e}")
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(wav_files)}")
    print(f"  Done: {done}, Failed: {fail}")


def cache_coughvid():
    """CoughVID: webm files → decode and cache as npy."""
    raw_dir = Path("data/raw/coughvid_public/public_dataset")
    mel_dir = Path("data/mel_cache/coughvid_zenodo")
    out_dir = Path("data/wav_cache/coughvid_zenodo")
    out_dir.mkdir(parents=True, exist_ok=True)

    mel_meta = json.loads((mel_dir / "metadata.json").read_text())
    mel_samples = mel_meta.get("samples", [])
    print(f"CoughVID: {len(mel_samples)} mel files")

    # CoughVID mel files are named coughvid_zenodo_XXXXXXX_Y.pt
    # We need the original webm file mapping - not stored in metadata
    # Cache all webm files by name for lookup
    webm_files = sorted(raw_dir.glob("*.webm"))
    print(f"  Found {len(webm_files)} webm files")

    done, fail = 0, 0
    for i, wf in enumerate(webm_files):
        out_path = out_dir / f"{wf.stem}.npy"
        if out_path.exists():
            done += 1
            continue
        try:
            wav, _ = librosa.load(str(wf), sr=SR, mono=True)
            wav = pad_or_crop(wav, TARGET_LEN)
            np.save(str(out_path), wav.astype(np.float32))
            done += 1
        except Exception as e:
            fail += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(webm_files)} done={done} fail={fail}")
    print(f"  Done: {done}, Failed: {fail}")


def cache_downstream():
    """Cache wav for ICBHI, COPD, KAUH downstream tasks."""
    tasks = {
        "opera_icbhi_disease": "opera_src/datasets/icbhi/ICBHI_final_database",
        "opera_copd": None,  # paths are absolute in metadata
        "opera_kauh": None,
    }

    for mel_name, base_dir in tasks.items():
        mel_dir = Path(f"data/mel_cache/{mel_name}")
        out_dir = Path(f"data/wav_cache/{mel_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = json.loads((mel_dir / "metadata.json").read_text())
        samples = meta.get("samples", [])
        print(f"\n{mel_name}: {len(samples)} samples")

        done, fail = 0, 0
        for i, s in enumerate(samples):
            out_path = out_dir / s["path"].replace(".pt", ".npy")
            if out_path.exists():
                done += 1
                continue

            wav_path = s.get("original_path", "")
            if not wav_path:
                fail += 1
                continue
            if base_dir and not Path(wav_path).is_absolute():
                wav_path = str(Path(base_dir) / wav_path)
            if not Path(wav_path).exists():
                fail += 1
                continue

            try:
                wav, _ = librosa.load(wav_path, sr=SR, mono=True)
                wav = pad_or_crop(wav, TARGET_LEN)
                np.save(str(out_path), wav.astype(np.float32))
                done += 1
            except Exception as e:
                fail += 1
                if fail <= 3:
                    print(f"  FAIL: {e}")
        print(f"  Done: {done}, Failed: {fail}")


if __name__ == "__main__":
    print("=== Caching raw waveforms ===\n")

    print("[1/3] Coswara")
    cache_coswara()

    print("\n[2/3] Downstream (ICBHI, COPD, KAUH)")
    cache_downstream()

    # CoughVID is large and slow (webm decoding), do it last
    print("\n[3/3] CoughVID (this takes a while)")
    cache_coughvid()

    print("\nAll done!")
