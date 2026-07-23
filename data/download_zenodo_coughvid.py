"""
Download the full CoughVID public_dataset.zip from Zenodo with resume support.

Usage:
  python data/download_zenodo_coughvid.py
  python data/download_zenodo_coughvid.py --extract
"""

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


URL = "https://zenodo.org/api/records/4498364/files/public_dataset.zip/content"


def download(dest: Path, chunk_size: int = 1024 * 1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(URL, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        if existing and r.status_code == 206:
            total += existing
            mode = "ab"
        else:
            existing = 0
            mode = "wb"

        with open(tmp, mode + "") as f, tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc="CoughVID Zenodo",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    tmp.replace(dest)
    print(f"Downloaded: {dest}")


def extract(zip_path: Path, extract_dir: Path):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, extract_dir)
    print(f"Extracted to: {extract_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="./data/downloads/coughvid_public_dataset.zip")
    p.add_argument("--extract-dir", default="./data/raw/coughvid_public")
    p.add_argument("--extract", action="store_true")
    args = p.parse_args()

    zip_path = Path(args.dest)
    if not zip_path.exists():
        download(zip_path)
    else:
        print(f"Already downloaded: {zip_path}")

    if args.extract:
        extract(zip_path, Path(args.extract_dir))


if __name__ == "__main__":
    main()
