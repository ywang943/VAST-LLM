"""
Download and extract the official ICBHI 2017 database.

The OPERA benchmark expects the directory layout:
  opera_src/datasets/icbhi/ICBHI_final_database/*.wav

Usage:
  python data/download_icbhi_official.py
"""

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(url, headers=headers, stream=True, timeout=60, verify=False) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        mode = "ab" if existing and r.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        elif total:
            total += existing
        with open(tmp, mode) as f, tqdm(total=total, initial=existing, unit="B", unit_scale=True, desc="ICBHI") as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    tmp.replace(dest)
    print(f"Downloaded: {dest}")


def extract(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for item in tqdm(zf.infolist(), desc="Extracting ICBHI"):
            zf.extract(item, dest)
    print(f"Extracted to: {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="./data/downloads/ICBHI_final_database.zip")
    p.add_argument("--dest", default="./opera_src/datasets/icbhi")
    args = p.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        download(URL, zip_path)
    else:
        print(f"Already downloaded: {zip_path}")
    extract(zip_path, Path(args.dest))


if __name__ == "__main__":
    main()
