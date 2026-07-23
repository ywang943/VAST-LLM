"""
Download additional datasets from HuggingFace for downstream evaluation.
Targets:
  1. CovidUK (Cambridge COVID-19 sounds)
  2. ICBHI 4-class (respiratory sounds classification)
  3. UrbanSound or other audio (for out-of-domain test)

Usage: python data/download_more_datasets.py
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


def decode_audio(item, audio_col="audio"):
    audio = item.get(audio_col, {})
    if not isinstance(audio, dict): return None, None
    raw = audio.get("bytes")
    if not raw: return None, None
    try:
        arr, sr = sf.read(io.BytesIO(raw))
        if arr.ndim > 1: arr = arr.mean(axis=1)
        return arr.astype(np.float32), sr
    except Exception:
        return None, None


def try_download(ds_id, kwargs, dest_name, label_key, max_samples=5000):
    dest = Path(f"data/mel_cache/{dest_name}")
    if (dest / "metadata.json").exists():
        n = len(json.loads((dest / "metadata.json").read_text()).get("samples", []))
        print(f"  {dest_name}: already {n} windows")
        return n
    dest.mkdir(parents=True, exist_ok=True)

    print(f"  Trying {ds_id}...")
    try:
        ds = load_dataset(ds_id, **{k: v for k, v in kwargs.items() if k != "audio_col"})
        audio_col = kwargs.get("audio_col", "audio")
        ds = ds.cast_column(audio_col, HFAudio(decode=False))
    except Exception as e:
        print(f"    FAILED to load: {str(e)[:80]}")
        return 0

    samples_out, cached, idx = [], 0, 0
    for item in ds:
        if max_samples and idx >= max_samples: break
        arr, sr = decode_audio(item, audio_col)
        if arr is None: idx += 1; continue
        try:
            if sr != SR:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
            label = item.get(label_key)
            if label is None: idx += 1; continue
            mel = preprocessor.to_mel(arr)
            fname = f"{dest_name}_{cached:06d}.pt"
            torch.save(mel, str(dest / fname))
            samples_out.append({"path": fname, "label": int(label) if isinstance(label, (int, float)) else str(label),
                                 "label_name": str(label), "split": "train"})
            cached += 1
        except Exception: pass
        idx += 1
        if idx % 500 == 0: print(f"    {idx} processed, {cached} cached")

    if not samples_out:
        print(f"    No data cached"); return 0

    meta = {"task": dest_name, "samples": samples_out}
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"    {cached} windows from {idx} samples")
    return cached


def main():
    print("Trying to download more datasets from HuggingFace...")
    print()

    # ICBHI on HuggingFace (binary, already have cache but try 4-class)
    # ICBHI 2017 from DynamicSuperb
    try_download(
        "DynamicSuperb/RespiratorySoundClassification_ICBHI2017",
        {"split": "test", "streaming": False, "audio_col": "audio"},
        "icbhi_hf_4class",
        label_key="label",
        max_samples=None,
    )

    # Try finding CovidUK or similar
    for ds_id, name in [
        ("speech-commands", "google_speech_commands"),
    ]:
        try_download(ds_id, {"split": "train", "streaming": True}, name, "label", 3000)

    print("\nDone.")


if __name__ == "__main__":
    main()
