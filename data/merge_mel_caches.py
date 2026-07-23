"""Merge multiple mel-cache directories into one cache directory."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--dest", required=True)
    args = p.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    samples = []
    for source_str in args.sources:
        source = Path(source_str)
        with open(source / "metadata.json", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw.get("samples", []):
            old = source / item["path"]
            rel = f"{source.name}_{len(samples):06d}.pt"
            shutil.copy2(old, dest / rel)
            new_item = dict(item)
            new_item["path"] = rel
            new_item["cache_source"] = source.name
            samples.append(new_item)
            if len(samples) % 1000 == 0:
                print(f"merged {len(samples)}")

    with open(dest / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2)
    print(f"Done: merged={len(samples)}, metadata={dest / 'metadata.json'}")


if __name__ == "__main__":
    main()
