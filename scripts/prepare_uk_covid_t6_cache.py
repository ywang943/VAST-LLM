#!/usr/bin/env python3
"""Create T6 UK COVID cough mel cache from the split Zenodo zip.

The dataset is distributed as a multi-disk zip. Python's zipfile cannot extract
spanned zip64 archives, so this script parses the central directory and reads
only the cough WAV files needed for T6 directly from the concatenated archive.
"""

import argparse
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from respvoice.preprocessing import AudioPreprocessor  # noqa: E402


DOWNLOAD = ROOT / "data/downloads/covid19_sounds"
COMBINED = DOWNLOAD / "covid_data_combined.zip"


def disk_offsets():
    parts = [DOWNLOAD / f"covid_data.z{i:02d}" for i in range(1, 25)] + [DOWNLOAD / "covid_data.zip"]
    offsets = [0]
    total = 0
    for p in parts[:-1]:
        total += p.stat().st_size
        offsets.append(total)
    return offsets


def read_eocd(fp):
    size = COMBINED.stat().st_size
    fp.seek(max(0, size - 200000))
    tail = fp.read()
    base = size - len(tail)
    eocd_rel = tail.rfind(b"PK\x05\x06")
    if eocd_rel < 0:
        raise RuntimeError("EOCD not found")
    eocd_abs = base + eocd_rel
    eocd = struct.unpack("<4s4H2LH", tail[eocd_rel:eocd_rel + 22])
    loc_rel = tail.rfind(b"PK\x06\x07", 0, eocd_rel)
    if loc_rel < 0:
        raise RuntimeError("ZIP64 locator not found")
    locator = struct.unpack("<4sLQL", tail[loc_rel:loc_rel + 20])
    return eocd_abs, eocd, locator


def parse_central_directory(wanted_names):
    offsets = disk_offsets()
    wanted = {f"audio/{name}" for name in wanted_names}
    found = {}
    with COMBINED.open("rb") as fp:
        _, eocd, _ = read_eocd(fp)
        cd_disk = eocd[2]
        cd_size = eocd[5]
        cd_offset = eocd[6]
        cd_abs = offsets[cd_disk] + cd_offset
        fp.seek(cd_abs)
        end = cd_abs + cd_size
        while fp.tell() < end:
            pos = fp.tell()
            hdr = fp.read(46)
            if hdr[:4] != b"PK\x01\x02":
                raise RuntimeError(f"Bad central-directory signature at {pos}")
            vals = struct.unpack("<4s6H3L5H2L", hdr)
            method = vals[4]
            flags = vals[3]
            comp_size = vals[8]
            uncomp_size = vals[9]
            fn_len, extra_len, comment_len = vals[10], vals[11], vals[12]
            disk_start = vals[13]
            rel_offset = vals[16]
            name = fp.read(fn_len).decode("utf-8", errors="replace")
            fp.seek(extra_len + comment_len, 1)
            if name in wanted:
                found[name] = {
                    "method": method,
                    "flags": flags,
                    "comp_size": comp_size,
                    "uncomp_size": uncomp_size,
                    "disk_start": disk_start,
                    "central_local_abs": offsets[disk_start] + rel_offset,
                }
                if len(found) == len(wanted):
                    break
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} files in zip, first={missing[:5]}")
    fill_real_local_offsets(found)
    return found


def fill_real_local_offsets(found):
    """Scan local headers once and fill real offsets for wanted filenames.

    Multi-disk zip central-directory offsets can be logical offsets that do not
    directly map to the concatenated file after entries span disks. The local
    headers themselves are unambiguous, so scan for them and match filenames.
    """
    wanted = set(found)
    remaining = set(wanted)
    sig = b"PK\x03\x04"
    chunk_size = 64 * 1024 * 1024
    overlap = 4096
    pos = 0
    carry = b""
    with COMBINED.open("rb") as fp:
        while remaining:
            block = fp.read(chunk_size)
            if not block:
                break
            buf = carry + block
            buf_base = pos - len(carry)
            i = 0
            while True:
                j = buf.find(sig, i)
                if j < 0:
                    break
                if j + 30 <= len(buf):
                    try:
                        vals = struct.unpack("<4s5H3L2H", buf[j:j + 30])
                        method = vals[3]
                        fn_len, extra_len = vals[9], vals[10]
                        name_start = j + 30
                        name_end = name_start + fn_len
                        if (
                            0 < fn_len < 512
                            and name_end <= len(buf)
                            and method in (0, 8)
                        ):
                            name = buf[name_start:name_end].decode("utf-8", errors="replace")
                            if name in remaining:
                                found[name]["local_abs"] = buf_base + j
                                remaining.remove(name)
                                print(f"  local header found {len(wanted) - len(remaining)}/{len(wanted)}", flush=True)
                    except Exception:
                        pass
                i = j + 4
            pos += len(block)
            carry = buf[-overlap:]
    if remaining:
        raise RuntimeError(f"Could not locate {len(remaining)} local headers, first={sorted(remaining)[:5]}")


def extract_entry(fp, info):
    fp.seek(info["local_abs"])
    local = fp.read(30)
    if local[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local header at {info['local_abs']}")
    vals = struct.unpack("<4s5H3L2H", local)
    fn_len, extra_len = vals[9], vals[10]
    fp.seek(fn_len + extra_len, 1)
    data = fp.read(info["comp_size"])
    if info["method"] == 0:
        out = data
    elif info["method"] == 8:
        out = zlib.decompress(data, -15)
    else:
        raise RuntimeError(f"Unsupported compression method {info['method']}")
    if len(out) != info["uncomp_size"]:
        raise RuntimeError(f"Bad uncompressed size got={len(out)} expected={info['uncomp_size']}")
    return out


def dms_text(row):
    symptoms = []
    for col, label in [
        ("symptom_none", "no symptoms"),
        ("symptom_cough_any", "cough"),
        ("symptom_shortness_of_breath", "shortness of breath"),
        ("symptom_sore_throat", "sore throat"),
        ("symptom_fatigue", "fatigue"),
        ("symptom_fever_high_temperature", "fever"),
        ("symptom_headache", "headache"),
        ("symptom_change_to_sense_of_smell_or_taste", "smell/taste change"),
    ]:
        if str(row.get(col, "0")) in ("1", "1.0", "True", "true"):
            symptoms.append(label)
    symptom_text = ", ".join(symptoms) if symptoms else "not reported"
    return (
        f"Age: {row.get('age', 'unknown')}; Gender: {row.get('gender', 'unknown')}; "
        f"Region: {row.get('region_name', 'unknown')}; Symptoms: {symptom_text}; "
        f"Smoker status: {row.get('smoker_status', 'unknown')}; "
        f"Asthma: {row.get('respiratory_condition_asthma', 'unknown')}; "
        f"Other respiratory condition: {row.get('respiratory_condition_other', 'unknown')}"
    )


def build_samples(split_name, max_samples=None):
    audio = pd.read_csv(DOWNLOAD / "audio_metadata.csv")
    part = pd.read_csv(DOWNLOAD / "participant_metadata.csv", low_memory=False)
    splits = pd.read_csv(DOWNLOAD / "train_test_splits.csv", low_memory=False)
    keep_cols = [
        "participant_identifier", "covid_test_result", "age", "gender", "region_name",
        "symptom_none", "symptom_cough_any", "symptom_shortness_of_breath",
        "symptom_sore_throat", "symptom_fatigue", "symptom_fever_high_temperature",
        "symptom_headache", "symptom_change_to_sense_of_smell_or_taste",
        "smoker_status", "respiratory_condition_asthma", "respiratory_condition_other",
    ]
    df = (
        audio[["participant_identifier", "cough_file_name"]]
        .merge(part[keep_cols], on="participant_identifier")
        .merge(splits[["participant_identifier", "splits"]], on="participant_identifier")
    )
    df = df[
        df["cough_file_name"].notna()
        & df["covid_test_result"].isin(["Positive", "Negative"])
        & df["splits"].eq(split_name)
    ].copy()
    df["label"] = (df["covid_test_result"] == "Positive").astype(int)
    if max_samples:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_samples // 2), random_state=0)
        )
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["test", "train", "val", "long"])
    parser.add_argument("--out", default="data/mel_cache/uk_covid_cough")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_samples(args.split, args.max_samples)
    print(f"T6 samples split={args.split}: n={len(df)} labels={df['label'].value_counts().to_dict()}")

    entries = parse_central_directory(df["cough_file_name"].tolist())
    pre = AudioPreprocessor(sr=16000, n_mels=64, win_ms=64.0, hop_ms=32.0, target_sec=8.0)
    meta = []
    with COMBINED.open("rb") as fp:
        for i, row in df.iterrows():
            out_name = f"uk_covid_cough_{i:06d}.pt"
            out_path = out_dir / out_name
            if not out_path.exists():
                wav_bytes = extract_entry(fp, entries[f"audio/{row['cough_file_name']}"])
                wav, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    import librosa
                    wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000)
                mel = pre.to_mel(np.asarray(wav, dtype=np.float32))
                torch.save(mel, out_path)
            meta.append({
                "path": out_name,
                "label": int(row["label"]),
                "split": "test",
                "participant_identifier": row["participant_identifier"],
                "audio_file": row["cough_file_name"],
                "covid_test_result": row["covid_test_result"],
                "dms_text": dms_text(row),
            })
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(df)}", flush=True)
    (out_dir / "metadata.json").write_text(json.dumps({"samples": meta}, indent=2), encoding="utf-8")
    print(f"Saved {len(meta)} samples to {out_dir}")


if __name__ == "__main__":
    main()
