"""
Download COVID-19 Sounds dataset from Zenodo.

This is the dataset used by RespLLM (Cambridge, Zenodo record 10043978).
Contains: exhalation, cough, breathing recordings with COVID labels.

RespLLM tasks from this dataset:
  S1: UK COVID exhalation  (1500 train / 1000 test)
  S2: UK COVID cough       (1500 train / 1000 test)

The Zenodo record contains the processed dataset.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEST = ROOT / "data/downloads/covid19_sounds"
DEST.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD = "10043978"
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

FALLBACK_FILES = [
    ("audio_metadata.csv", 23736633),
    ("DataDictionary_UKCOVID19VocalAudioDataset_OpenAccess.xlsx", 22291),
    ("train_test_splits.csv", 7270746),
    ("covid_data.zip", 2092112787),
    *[(f"covid_data.z{i:02d}", 2147483648) for i in range(1, 25)],
    ("README.md", 0),
    ("participant_metadata.csv", 0),
]


def download_with_wget(url, dest_path):
    cmd = [
        "wget",
        "-c",
        "--progress=dot:giga",
        "--tries=50",
        "--waitretry=10",
        "--retry-connrefused",
        "--read-timeout=60",
        "--timeout=30",
        "--no-check-certificate",
        "-O", str(dest_path),
        url,
    ]
    print(f"  Downloading: {url}")
    print(f"  To: {dest_path}")
    subprocess.run(cmd, check=True)


def main():
    print("COVID-19 Sounds Dataset Downloader")
    print("=" * 60)

    # First, fetch the record metadata to get file URLs
    import json
    import urllib.request

    print(f"\nFetching Zenodo record {ZENODO_RECORD} metadata...")
    try:
        with urllib.request.urlopen(ZENODO_URL, timeout=60) as resp:
            record = json.loads(resp.read())
        files = record.get("files", [])
    except Exception as e:
        print(f"  Failed to fetch metadata: {e}")
        print("  Falling back to the cached file manifest.")
        files = [
            {
                "key": fname,
                "size": size,
                "links": {
                    "self": f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{fname}/content"
                },
            }
            for fname, size in FALLBACK_FILES
        ]

    print(f"  Found {len(files)} files:")
    total_size = 0
    for f in files:
        size_mb = f.get("size", 0) / 1e6
        total_size += size_mb
        print(f"    {f['key']:50s} {size_mb:8.1f} MB")
    print(f"  Total: {total_size:.0f} MB")

    # Download all files
    for f in files:
        fname = f["key"]
        url = f["links"]["self"]
        dest_path = DEST / fname

        if dest_path.exists():
            existing_size = dest_path.stat().st_size
            expected_size = f.get("size", 0)
            if expected_size <= 0 and existing_size > 0:
                print(f"\n  {fname}: already downloaded ({existing_size / 1e6:.1f} MB)")
                continue
            if abs(existing_size - expected_size) < 1024:
                print(f"\n  {fname}: already downloaded ({existing_size / 1e6:.1f} MB)")
                continue
            else:
                print(f"\n  {fname}: partial ({existing_size / 1e6:.1f} MB / {expected_size / 1e6:.1f} MB), resuming...")

        print(f"\n  Downloading {fname}...")
        try:
            download_with_wget(url, dest_path)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: failed to download {fname}: {e}")
            print("  Continuing with the next file; rerun this script to resume.")
            continue

    incomplete = []
    for f in files:
        fname = f["key"]
        expected_size = f.get("size", 0)
        fpath = DEST / fname
        if expected_size and (not fpath.exists() or abs(fpath.stat().st_size - expected_size) >= 1024):
            incomplete.append(fname)

    if incomplete:
        print("\nSome files are still incomplete; skipping extraction.")
        print("Incomplete files:")
        for fname in incomplete:
            print(f"  {fname}")
        return

    # Extract only after all split zip parts are complete.
    for f in files:
        fname = f["key"]
        fpath = DEST / fname
        if fname.endswith(".zip") and fpath.exists():
            extract_dir = DEST / fname.replace(".zip", "")
            if extract_dir.exists():
                print(f"\n  {fname}: already extracted")
                continue
            print(f"\n  Extracting {fname}...")
            subprocess.run(["unzip", "-o", "-q", str(fpath), "-d", str(DEST)], check=True)
        elif fname.endswith(".tar.gz") and fpath.exists():
            print(f"\n  Extracting {fname}...")
            subprocess.run(["tar", "xzf", str(fpath), "-C", str(DEST)], check=True)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Files in: {DEST}")
    print("\nNext: prepare mel caches for RQ1/RQ3 integration")


if __name__ == "__main__":
    main()
