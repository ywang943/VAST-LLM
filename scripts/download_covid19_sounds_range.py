#!/usr/bin/env python3
"""Concurrent range downloader for incomplete Zenodo COVID-19 Sounds split zips."""

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data/downloads/covid19_sounds"
RECORD_URL = "https://zenodo.org/api/records/10043978"
CHUNK_SIZE = 64 * 1024 * 1024


def fetch_files():
    with urllib.request.urlopen(RECORD_URL, timeout=60) as resp:
        record = json.loads(resp.read())
    out = []
    for f in record["files"]:
        key = f["key"]
        if key.startswith("covid_data"):
            checksum = f.get("checksum", "")
            if checksum.startswith("md5:"):
                checksum = checksum.split(":", 1)[1]
            out.append({
                "key": key,
                "size": int(f["size"]),
                "url": f["links"]["self"],
                "md5": checksum,
            })
    return sorted(out, key=lambda x: x["key"])


def md5_file(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_state(path):
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("done", []))
    except Exception:
        return set()


def save_state(path, done):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"done": sorted(done)}, indent=2), encoding="utf-8")
    tmp.replace(path)


def request_range(url, start, end, timeout):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_chunk(url, fd, idx, start, end, timeout, retries):
    expected = end - start + 1
    for attempt in range(1, retries + 1):
        try:
            data = request_range(url, start, end, timeout)
            if len(data) != expected:
                raise IOError(f"short read chunk={idx} got={len(data)} expected={expected}")
            os.pwrite(fd, data, start)
            return idx, expected
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(30, 2 * attempt))
    raise RuntimeError("unreachable")


def prepare_part_file(dest, part, state_path, size):
    if part.exists():
        if part.stat().st_size > size:
            part.unlink()
            state_path.unlink(missing_ok=True)
            part.touch()
    elif dest.exists() and dest.stat().st_size < size:
        dest.replace(part)
    else:
        part.touch()


def download_one(item, workers, timeout, retries, only_missing_tail):
    dest = DEST / item["key"]
    size = item["size"]
    expected_md5 = item["md5"]

    if dest.exists() and dest.stat().st_size == size:
        got = md5_file(dest)
        if got == expected_md5:
            print(f"{item['key']}: complete md5 ok")
            return True
        print(f"{item['key']}: size ok but md5 mismatch ({got}); redownloading")
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".partdl")
    state_path = dest.with_suffix(dest.suffix + ".partdl.json")
    prepare_part_file(dest, part, state_path, size)

    chunks = []
    for idx, start in enumerate(range(0, size, CHUNK_SIZE)):
        end = min(size - 1, start + CHUNK_SIZE - 1)
        chunks.append((idx, start, end))

    done = load_state(state_path)
    if not done and part.stat().st_size > 0:
        covered = min(part.stat().st_size, size)
        for idx, start, end in chunks:
            if end < covered:
                done.add(idx)
        save_state(state_path, done)

    if only_missing_tail:
        covered = min(part.stat().st_size, size)
        todo = [(idx, start, end) for idx, start, end in chunks if end >= covered and idx not in done]
    else:
        todo = [(idx, start, end) for idx, start, end in chunks if idx not in done]

    print(f"{item['key']}: size={size} done={len(done)}/{len(chunks)} todo={len(todo)} workers={workers}")
    if todo:
        lock = threading.Lock()
        downloaded = 0
        fd = os.open(part, os.O_RDWR)
        try:
            with futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [
                    ex.submit(download_chunk, item["url"], fd, idx, start, end, timeout, retries)
                    for idx, start, end in todo
                ]
                for fut in futures.as_completed(futs):
                    idx, nbytes = fut.result()
                    with lock:
                        done.add(idx)
                        downloaded += nbytes
                        save_state(state_path, done)
                        print(
                            f"  {item['key']}: chunks {len(done)}/{len(chunks)} "
                            f"new={downloaded / 1e9:.2f} GB",
                            flush=True,
                        )
        finally:
            os.close(fd)

    got = md5_file(part)
    if got != expected_md5:
        print(f"{item['key']}: md5 mismatch got={got} expected={expected_md5}")
        return False
    part.replace(dest)
    state_path.unlink(missing_ok=True)
    print(f"{item['key']}: complete md5 ok")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--only-missing-tail", action="store_true")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    wanted = set(args.files or [])
    files = fetch_files()
    if wanted:
        files = [f for f in files if f["key"] in wanted]

    failed = []
    for item in files:
        dest = DEST / item["key"]
        if dest.exists() and dest.stat().st_size == item["size"] and md5_file(dest) == item["md5"]:
            print(f"{item['key']}: already complete")
            continue
        ok = download_one(item, args.workers, args.timeout, args.retries, args.only_missing_tail)
        if not ok:
            failed.append(item["key"])

    if failed:
        raise SystemExit("Failed files: " + ", ".join(failed))


if __name__ == "__main__":
    main()
