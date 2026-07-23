#!/usr/bin/env python3
"""Prepare derived mel-cache task folders for the paper table IDs.

This only creates lightweight hardlinks/copies from already-computed mel caches;
it does not recompute spectrograms. It is used for task definitions that exist
inside a broader cache but do not yet have a standalone metadata.json.
"""

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def subset_coswara_audio_type(src_name, dst_name, audio_type, prefix):
    src_root = ROOT / "data/mel_cache" / src_name
    dst_root = ROOT / "data/mel_cache" / dst_name
    raw = json.loads((src_root / "metadata.json").read_text(encoding="utf-8"))
    samples = raw.get("samples", raw if isinstance(raw, list) else [])
    out = []
    for s in samples:
        if s.get("audio_type") != audio_type:
            continue
        idx = len(out)
        dst_file = f"{prefix}_{idx:06d}.pt"
        link_or_copy(src_root / s["path"], dst_root / dst_file)
        item = dict(s)
        item["path"] = dst_file
        item["source_cache"] = src_name
        item["source_path"] = s["path"]
        out.append(item)
    if not out:
        raise RuntimeError(f"No samples with audio_type={audio_type} in {src_name}")
    (dst_root / "metadata.json").write_text(
        json.dumps({"samples": out}, indent=2),
        encoding="utf-8",
    )
    print(f"{dst_name}: {len(out)} samples from {src_name} audio_type={audio_type}")


def all_test_clone(src_name, dst_name, prefix):
    src_root = ROOT / "data/mel_cache" / src_name
    dst_root = ROOT / "data/mel_cache" / dst_name
    raw = json.loads((src_root / "metadata.json").read_text(encoding="utf-8"))
    samples = raw.get("samples", raw if isinstance(raw, list) else [])
    out = []
    for s in samples:
        if "label" not in s:
            continue
        idx = len(out)
        dst_file = f"{prefix}_{idx:06d}.pt"
        link_or_copy(src_root / s["path"], dst_root / dst_file)
        item = dict(s)
        item["path"] = dst_file
        item["source_cache"] = src_name
        item["source_path"] = s["path"]
        item["original_split"] = s.get("split")
        item["split"] = "test"
        out.append(item)
    if not out:
        raise RuntimeError(f"No labeled samples in {src_name}")
    (dst_root / "metadata.json").write_text(
        json.dumps({"samples": out}, indent=2),
        encoding="utf-8",
    )
    print(f"{dst_name}: {len(out)} all-test samples cloned from {src_name}")


def main():
    subset_coswara_audio_type(
        "coswara_covid_all",
        "coswara_covid_exhale",
        "breathing-shallow",
        "coswara_covid_exhale",
    )
    # Current SVD cache does not expose a separate unseen protocol for Table T7.
    # This clone is for table plumbing/prototype experiments only; analysis must
    # flag it as non-independent if S6 also trains on svd_full.
    all_test_clone("svd_full", "svd_full_target_alltest", "svd_target")


if __name__ == "__main__":
    main()
