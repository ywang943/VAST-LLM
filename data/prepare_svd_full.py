"""Prepare the complete Saarbruecken Voice Database for downstream tasks.

The official archives store 16-bit PCM in a small FORM/DS16 container.  This
script reads the selected recordings directly from the verified ZIP archives,
resamples them from 50 kHz to 16 kHz, and writes the project's 8-second Mel
cache.  Splits are stratified by label and grouped by subject.
"""

import argparse
import json
import math
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from respvoice.preprocessing import AudioPreprocessor


ARCHIVE_ROOT = Path("data/downloads/svd_archives")
OUTPUT_ROOT = Path("data/mel_cache/svd_full")
SOURCES = {
    "vowel": ("vowels", "-a_n.nsp"),
    "phrase": ("sentences", "-phrase.nsp"),
}


def decode_ds16(payload: bytes) -> tuple[np.ndarray, int]:
    """Decode a FORM/DS16 recording into mono float32 PCM and sample rate."""
    if payload[:8] != b"FORMDS16":
        raise ValueError("not a FORM/DS16 recording")

    sample_rate = None
    samples = None
    offset = 12
    while offset + 8 <= len(payload):
        tag = payload[offset:offset + 4]
        size = struct.unpack_from("<I", payload, offset + 4)[0]
        start = offset + 8
        end = start + size
        if end > len(payload):
            raise ValueError(f"truncated {tag!r} chunk")
        if tag == b"HEDR" and size >= 24:
            sample_rate = struct.unpack_from("<I", payload, start + 20)[0]
        elif tag == b"SDA_":
            samples = np.frombuffer(payload[start:end], dtype="<i2").copy()
        offset = end + (size % 2)

    if sample_rate is None or samples is None:
        raise ValueError("missing HEDR or SDA_ chunk")
    if not 8_000 <= sample_rate <= 192_000:
        raise ValueError(f"invalid sample rate: {sample_rate}")
    return samples.astype(np.float32) / 32768.0, sample_rate


def archive_manifest(archive_root: Path):
    """Collect one vowel and phrase per subject plus all diagnosis labels."""
    archives = sorted(archive_root.glob("*.zip"), key=lambda p: p.name != "healthy.zip")
    if len(archives) != 72:
        raise RuntimeError(f"expected 72 SVD category archives, found {len(archives)}")

    recordings = defaultdict(dict)
    diagnoses = defaultdict(set)
    labels = {}
    for archive in archives:
        healthy = archive.name == "healthy.zip"
        diagnosis = "healthy" if healthy else archive.stem
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                parts = Path(member).parts
                if len(parts) != 3 or not parts[0].isdigit():
                    continue
                subject_id = parts[0]
                source = None
                for candidate, (folder, suffix) in SOURCES.items():
                    if parts[1] == folder and parts[2].endswith(suffix):
                        source = candidate
                        break
                if source is None:
                    continue
                label = 0 if healthy else 1
                previous = labels.setdefault(subject_id, label)
                if previous != label:
                    raise RuntimeError(f"subject {subject_id} appears in both classes")
                diagnoses[subject_id].add(diagnosis)
                recordings[subject_id].setdefault(source, (archive, member))
    return recordings, diagnoses, labels


def subject_splits(subject_ids, labels, seed):
    y = [labels[subject_id] for subject_id in subject_ids]
    trainval, test = train_test_split(
        subject_ids, test_size=0.20, random_state=seed, stratify=y,
    )
    trainval_y = [labels[subject_id] for subject_id in trainval]
    train, val = train_test_split(
        trainval, test_size=0.125, random_state=seed, stratify=trainval_y,
    )
    return {
        **{subject_id: "train" for subject_id in train},
        **{subject_id: "val" for subject_id in val},
        **{subject_id: "test" for subject_id in test},
    }


def resample(wav: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return wav
    divisor = math.gcd(source_sr, target_sr)
    return resample_poly(wav, target_sr // divisor, source_sr // divisor).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Smoke-test limit; do not use for final evaluation.")
    args = parser.parse_args()

    recordings, diagnoses, labels = archive_manifest(args.archive_root)
    subject_ids = sorted(recordings, key=int)
    if args.max_subjects:
        healthy = [s for s in subject_ids if labels[s] == 0][:args.max_subjects]
        pathological = [s for s in subject_ids if labels[s] == 1][:args.max_subjects]
        subject_ids = sorted(healthy + pathological, key=int)
    splits = subject_splits(subject_ids, labels, args.seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    preprocessor = AudioPreprocessor(
        sr=16_000, n_mels=64, win_ms=64.0, hop_ms=32.0, target_sec=8.0,
    )
    samples = []
    failures = []
    by_archive = defaultdict(list)
    for subject_id in subject_ids:
        for source, location in recordings[subject_id].items():
            archive, member = location
            by_archive[archive].append((subject_id, source, member))

    completed = 0
    total = sum(len(items) for items in by_archive.values())
    for archive in sorted(by_archive):
        with zipfile.ZipFile(archive) as handle:
            for subject_id, source, member in by_archive[archive]:
                filename = f"svd_{int(subject_id):04d}_{source}.pt"
                output = args.output_root / filename
                try:
                    if not output.exists():
                        wav, source_sr = decode_ds16(handle.read(member))
                        wav = resample(wav, source_sr, preprocessor.sr)
                        torch.save(preprocessor.to_mel(wav), output)
                    samples.append({
                        "path": filename,
                        "label": labels[subject_id],
                        "label_name": "healthy" if labels[subject_id] == 0 else "pathological",
                        "subject_id": subject_id,
                        "source": source,
                        "split": splits[subject_id],
                        "diagnoses": sorted(diagnoses[subject_id]),
                        "archive": archive.name,
                        "member": member,
                    })
                except Exception as error:
                    failures.append({"subject_id": subject_id, "source": source,
                                     "member": member, "error": repr(error)})
                completed += 1
                if completed % 100 == 0 or completed == total:
                    print(f"processed {completed}/{total}, failures={len(failures)}", flush=True)

    subject_counts = {
        name: sum(splits[s] == name for s in subject_ids)
        for name in ("train", "val", "test")
    }
    label_counts = {
        "healthy": sum(labels[s] == 0 for s in subject_ids),
        "pathological": sum(labels[s] == 1 for s in subject_ids),
    }
    paired_subjects = sum(
        {"vowel", "phrase"}.issubset(recordings[s]) for s in subject_ids
    )
    metadata = {
        "task": "svd_voice_pathology_full",
        "description": "Complete SVD, subject-independent healthy-vs-pathological split",
        "archive_count": 72,
        "subject_count": len(subject_ids),
        "paired_subject_count": paired_subjects,
        "label_counts": label_counts,
        "subject_split_counts": subject_counts,
        "sample_count": len(samples),
        "failure_count": len(failures),
        "failures": failures,
        "samples": sorted(samples, key=lambda x: (int(x["subject_id"]), x["source"])),
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("samples", "failures")},
                     indent=2, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"SVD preparation completed with {len(failures)} failures")


if __name__ == "__main__":
    main()
