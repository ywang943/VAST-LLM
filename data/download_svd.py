"""
Download Saarbruecken Voice Database (SVD) from Zenodo.
Record: https://zenodo.org/records/16874898
License: CC BY 4.0

Contents: 73 zip files, organized by voice pathology type.
Each zip contains .nsp files (phonetogram recordings).
We convert them to wav and prepare for downstream evaluation.

Usage:
    python data/download_svd.py
    python data/download_svd.py --max-files 5  # quick test with 5 categories
"""

import argparse, io, json, os, sys, zipfile
from pathlib import Path

import librosa, numpy as np, requests, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from respvoice.preprocessing import AudioPreprocessor

ZENODO_ID  = "16874898"
DEST_WAV   = Path("data/audio/svd")
DEST_CACHE = Path("data/mel_cache/svd")
SR = 16000

preprocessor = AudioPreprocessor(sr=SR, n_mels=64, win_ms=64.0, hop_ms=32.0, target_sec=8.0)

# SVD pathology categories: healthy starts with "Normal"
HEALTHY_KEYS = {"Normal", "Normalstimmen"}


def get_file_list():
    """Fetch list of zip files from Zenodo API."""
    r = requests.get(f"https://zenodo.org/api/records/{ZENODO_ID}", timeout=15)
    r.raise_for_status()
    data = r.json()
    files = data.get("files", [])
    total_mb = sum(f.get("size", 0) for f in files) // 1024 // 1024
    print(f"SVD: {len(files)} zip files, ~{total_mb}MB total")
    return files


def convert_nsp_to_wav(nsp_bytes: bytes) -> tuple:
    """
    Try to decode NSP file as raw audio.
    NSP format: 16-bit PCM, 16kHz or 25kHz. Try both.
    """
    for sr_try in [25000, 16000, 22050]:
        try:
            arr = np.frombuffer(nsp_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(arr) < 100: continue
            if sr_try != SR:
                arr = librosa.resample(arr, orig_sr=sr_try, target_sr=SR)
            return arr, True
        except Exception:
            continue
    return None, False


def download_and_process_zip(file_info: dict, dest_wav: Path, label: int, category: str) -> list:
    """Download one zip, extract NSP files, convert to wav + mel."""
    url = file_info["links"]["self"]
    key = file_info["key"]
    name = Path(key).stem  # e.g. "Normal" or "Cyste"

    print(f"  Downloading {key} ({file_info['size']//1024//1024}MB) ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        zdata = io.BytesIO(r.content)
    except Exception as e:
        print(f"FAILED: {e}")
        return []

    samples = []
    try:
        with zipfile.ZipFile(zdata) as zf:
            nsp_files = [n for n in zf.namelist() if n.lower().endswith(".nsp")]
            print(f"{len(nsp_files)} NSP files")
            for nsp_name in nsp_files[:200]:  # limit per category
                try:
                    nsp_bytes = zf.read(nsp_name)
                    arr, ok = convert_nsp_to_wav(nsp_bytes)
                    if not ok or arr is None: continue

                    # Save wav
                    speaker = Path(nsp_name).stem.replace("/", "_")
                    wav_path = dest_wav / f"{name}_{speaker}.wav"
                    dest_wav.mkdir(parents=True, exist_ok=True)
                    sf.write(str(wav_path), arr, SR)

                    # Save mel cache
                    DEST_CACHE.mkdir(parents=True, exist_ok=True)
                    mel = preprocessor.to_mel(arr)
                    fname = f"svd_{len(samples):06d}.pt"
                    torch.save(mel, str(DEST_CACHE / fname))
                    samples.append({
                        "path": fname,
                        "label": label,
                        "label_name": "healthy" if label == 0 else "pathological",
                        "category": name,
                        "split": "train",
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"  Zip error: {e}")

    return samples


def stratified_split(samples):
    """Assign train/val/test splits stratified by label."""
    import random
    random.seed(1337)
    by_label = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)
    for label_id, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * 0.20))
        n_val  = max(1, int(n * 0.10))
        for i, s in enumerate(items):
            if i < n_test:        s["split"] = "test"
            elif i < n_test+n_val: s["split"] = "val"
            else:                  s["split"] = "train"
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-files", type=int, default=None, help="Max zip files to download")
    p.add_argument("--skip-existing", action="store_true", default=True)
    args = p.parse_args()

    meta_path = DEST_CACHE / "metadata.json"
    if args.skip_existing and meta_path.exists():
        existing = json.loads(meta_path.read_text())
        n = len(existing.get("samples", []))
        print(f"SVD cache already exists ({n} windows). Delete to re-download.")
        return

    print("Fetching SVD file list from Zenodo...")
    files = get_file_list()

    if args.max_files:
        files = files[:args.max_files]

    all_samples = []
    for file_info in files:
        key = file_info["key"]
        name = Path(key).stem
        # Determine label: Normal/Normalstimmen -> healthy (0), else pathological (1)
        label = 0 if any(h in name for h in HEALTHY_KEYS) else 1
        category = name

        samples = download_and_process_zip(file_info, DEST_WAV, label, category)
        all_samples.extend(samples)
        print(f"    -> {len(samples)} windows (total: {len(all_samples)})")

    if not all_samples:
        print("No samples collected!")
        return

    all_samples = stratified_split(all_samples)

    # Count labels
    label_counts = {}
    for s in all_samples:
        label_counts[s["label_name"]] = label_counts.get(s["label_name"], 0) + 1
    split_counts = {}
    for s in all_samples:
        split_counts[s["split"]] = split_counts.get(s["split"], 0) + 1

    meta = {
        "task": "svd_voice_pathology",
        "description": "SVD: healthy vs pathological voice (Saarbruecken Voice Database)",
        "samples": all_samples,
        "label_counts": label_counts,
        "split_counts": split_counts,
    }
    DEST_CACHE.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nDone: {len(all_samples)} mel windows")
    print(f"Labels: {label_counts}")
    print(f"Splits: {split_counts}")
    print(f"Cache: {DEST_CACHE}")


if __name__ == "__main__":
    main()
