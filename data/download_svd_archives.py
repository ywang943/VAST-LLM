"""Download the complete SVD category archives from the official Zenodo record.

The record also contains ``data.zip``, an aggregate archive that duplicates the
category archives. By default this script downloads all 72 category archives
(including ``healthy.zip``) and skips that duplicate aggregate.

Downloads are resumable through ``wget --continue`` and every completed file is
checked against the size and MD5 checksum reported by Zenodo.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests


ZENODO_RECORD = "16874898"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
DEFAULT_DEST = Path("data/downloads/svd_archives")


def fetch_manifest(include_aggregate: bool) -> list[dict]:
    response = requests.get(ZENODO_API, timeout=60)
    response.raise_for_status()
    files = response.json().get("files", [])
    if not include_aggregate:
        files = [item for item in files if item["key"] != "data.zip"]

    # Healthy data is needed most urgently for the balanced downstream task.
    return sorted(
        files,
        key=lambda item: (item["key"] != "healthy.zip", item["size"]),
    )


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_md5(item: dict) -> str:
    checksum = item.get("checksum", "")
    return checksum.split(":", 1)[-1] if checksum else ""


def is_complete(path: Path, item: dict, verify_md5: bool = True) -> bool:
    if not path.exists() or path.stat().st_size != item["size"]:
        return False
    return not verify_md5 or md5sum(path) == expected_md5(item)


def write_status(dest: Path, files: list[dict], active: Optional[str] = None) -> None:
    rows = []
    for item in files:
        path = dest / item["key"]
        size = path.stat().st_size if path.exists() else 0
        rows.append(
            {
                "name": item["key"],
                "expected_bytes": item["size"],
                "downloaded_bytes": size,
                "size_complete": size == item["size"],
                "md5": expected_md5(item),
            }
        )
    payload = {
        "zenodo_record": ZENODO_RECORD,
        "active": active,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_bytes": sum(item["size"] for item in files),
        "downloaded_bytes": sum(row["downloaded_bytes"] for row in rows),
        "files": rows,
    }
    (dest / "download_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )


def download_file(dest: Path, item: dict, retry_delay: int = 15) -> None:
    path = dest / item["key"]
    url = item["links"].get("content") or item["links"]["self"]

    if is_complete(path, item):
        print(f"[skip] {item['key']} already complete and verified", flush=True)
        return

    if path.exists() and path.stat().st_size == item["size"]:
        corrupt = path.with_suffix(path.suffix + ".bad-md5")
        path.replace(corrupt)
        print(f"[warn] checksum mismatch; moved existing file to {corrupt}", flush=True)

    print(
        f"[download] {item['key']} "
        f"({item['size'] / 1024**3:.2f} GiB)",
        flush=True,
    )
    command = [
        "wget",
        "--continue",
        "--tries=0",
        "--timeout=30",
        "--read-timeout=30",
        "--retry-connrefused",
        "--waitretry=5",
        "--progress=dot:giga",
        "--output-document",
        str(path),
        url,
    ]
    attempt = 0
    while True:
        attempt += 1
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            break
        current_size = path.stat().st_size if path.exists() else 0
        print(
            f"[retry] {item['key']} wget exit={result.returncode}, "
            f"downloaded={current_size / 1024**3:.2f} GiB; "
            f"retrying in {retry_delay}s (attempt {attempt})",
            flush=True,
        )
        time.sleep(retry_delay)

    if path.stat().st_size != item["size"]:
        raise RuntimeError(
            f"size mismatch for {item['key']}: "
            f"{path.stat().st_size} != {item['size']}"
        )
    actual_md5 = md5sum(path)
    if actual_md5 != expected_md5(item):
        raise RuntimeError(
            f"MD5 mismatch for {item['key']}: "
            f"{actual_md5} != {expected_md5(item)}"
        )
    print(f"[verified] {item['key']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--include-aggregate",
        action="store_true",
        help="Also download data.zip (duplicates the category archives).",
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    files = fetch_manifest(args.include_aggregate)
    expected = sum(item["size"] for item in files)
    print(
        f"SVD: {len(files)} archives, {expected / 1024**3:.2f} GiB total; "
        f"destination={args.dest}",
        flush=True,
    )

    write_status(args.dest, files)
    for item in files:
        write_status(args.dest, files, active=item["key"])
        download_file(args.dest, item)
        write_status(args.dest, files)
    print("All requested SVD archives downloaded and verified.", flush=True)


if __name__ == "__main__":
    main()
