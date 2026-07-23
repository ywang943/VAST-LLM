"""
Data download helpers for RespVoice.

Usage:
  python data/download_data.py --dataset icbhi --dest ./data/audio/icbhi
  python data/download_data.py --dataset coswara --dest ./data/audio/coswara
  python data/download_data.py --list

Datasets are ALL open-source / CC-licensed (from audio.md Section 8):

Pretraining sources:
  - ICBHI 2017 (respiratory)       [CC BY 4.0]
  - Coswara (voice + breath)       [CC BY 4.0]
  - CoughVID (cough)               [CC BY 4.0]
  - HF Lung (lung sounds)          [custom license, research use]
  - Saarbrücken Voice DB (voice)   [free for research]

Downstream:
  - SVD  (voice pathology)
  - AVFAD (voice pathology, Arabic)
  - VOICED (voice disorders)
"""

import os
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path


DATASET_INFO = {
    "icbhi": {
        "description": "ICBHI 2017 Respiratory Sound Database",
        "url": "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database.zip",
        "type": "zip",
        "note": "ICBHI requires manual download from the challenge website. "
                "Visit: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge",
    },
    "coughvid": {
        "description": "CoughVID — crowdsourced cough recordings",
        "url": "https://zenodo.org/record/4498364/files/public_dataset.zip",
        "type": "zip",
        "note": "Zenodo public dataset. ~1.2 GB.",
    },
    "coswara": {
        "description": "Coswara — COVID-19 sound dataset",
        "url": "https://github.com/iiscleap/Coswara-Data",
        "type": "git",
        "note": "Clone from GitHub: git clone https://github.com/iiscleap/Coswara-Data",
    },
    "svd": {
        "description": "Saarbrücken Voice Database",
        "url": "http://www.stimmdatenbank.coli.uni-saarland.de/",
        "type": "manual",
        "note": "Requires registration. Download from: https://www.stimmdatenbank.coli.uni-saarland.de/",
    },
}


def list_datasets():
    print("\nAvailable datasets for RespVoice:")
    print("-" * 60)
    for name, info in DATASET_INFO.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        print(f"    {info['note']}")
    print()


def download_coughvid(dest: str):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    url = DATASET_INFO["coughvid"]["url"]
    zip_path = dest / "coughvid.zip"

    print(f"Downloading CoughVID to {dest}...")
    print(f"  URL: {url}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=progress)
        print()
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        print(f"Done. Files in {dest}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from: https://zenodo.org/record/4498364")


def check_opera_checkpoint():
    """Download OPERA-CT checkpoint from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        print("Downloading OPERA-CT checkpoint from HuggingFace...")
        path = hf_hub_download(
            repo_id="evelyn0414/OPERA",
            filename="operaCT.pth",
            cache_dir="./checkpoints/opera_cache",
        )
        print(f"OPERA-CT checkpoint saved to: {path}")
        return path
    except Exception as e:
        print(f"Failed to download OPERA-CT: {e}")
        print("Install huggingface_hub: pip install huggingface_hub")
        return None


def print_manual_instructions():
    print("""
=== Manual Download Instructions for RespVoice Datasets ===

1. ICBHI 2017 (respiratory sounds, 4-class):
   https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
   → Extract to: data/audio/icbhi/

2. Coswara (COVID-19 sounds, voice+breath):
   git clone https://github.com/iiscleap/Coswara-Data data/audio/coswara

3. CoughVID (cough recordings):
   https://zenodo.org/record/4498364
   → Extract public_dataset.zip to: data/audio/coughvid/

4. SVD — Saarbrücken Voice Database (voice pathology):
   http://www.stimmdatenbank.coli.uni-saarland.de/
   → Requires registration → Extract to: data/audio/svd/

5. AVFAD (Arabic voice pathology):
   https://archive.ics.uci.edu/dataset/555/avfad
   → Extract to: data/audio/avfad/

After downloading, run the preprocessing check:
   python main.py --check-data

=============================================================
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RespVoice datasets")
    parser.add_argument("--dataset", choices=list(DATASET_INFO.keys()),
                        help="Dataset to download")
    parser.add_argument("--dest", default="./data/audio",
                        help="Destination directory")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")
    parser.add_argument("--opera-ckpt", action="store_true",
                        help="Download OPERA-CT pretrained checkpoint")
    parser.add_argument("--instructions", action="store_true",
                        help="Print manual download instructions")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.opera_ckpt:
        check_opera_checkpoint()
    elif args.instructions:
        print_manual_instructions()
    elif args.dataset == "coughvid":
        download_coughvid(os.path.join(args.dest, "coughvid"))
    elif args.dataset:
        info = DATASET_INFO[args.dataset]
        print(f"\n{info['description']}")
        print(f"Note: {info['note']}")
        if info["type"] == "git":
            print(f"  Run: git clone {info['url']} {os.path.join(args.dest, args.dataset)}")
        elif info["type"] == "manual":
            print(f"  Manual download required. See: {info['url']}")
    else:
        parser.print_help()
        print()
        print_manual_instructions()
