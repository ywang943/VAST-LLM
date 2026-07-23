#!/usr/bin/env python3
"""
Rebuild DMS templates for SVD and KAUH datasets.

SVD: Extract gender and age from overview.csv files inside zip archives.
     Maps AufnahmeID (= subject_id in metadata) to demographics.
KAUH: Extract gender, age, and recording location from original filenames.

Neither includes diagnosis in the DMS text (avoids label leakage).
"""

import json
import os
import csv
import io
import zipfile
from datetime import datetime
from typing import Optional

BASE_DIR = "/hpc2hdd/home/ywang943/KDD_Audio"
SVD_ARCHIVES_DIR = os.path.join(BASE_DIR, "data/downloads/svd_archives")
SVD_META_PATH = os.path.join(BASE_DIR, "data/mel_cache/svd_full/metadata.json")
SVD_DMS_PATH = os.path.join(BASE_DIR, "data/dms_templates/svd_dms.json")

KAUH_META_PATH = os.path.join(BASE_DIR, "data/mel_cache/opera_kauh/metadata.json")
KAUH_DMS_PATH = os.path.join(BASE_DIR, "data/dms_templates/kauh_dms.json")

# --- Location mapping for KAUH ---
KAUH_LOCATION_MAP = {
    "P R U": "posterior right upper",
    "P R M": "posterior right middle",
    "P R L": "posterior right lower",
    "P L U": "posterior left upper",
    "P L M": "posterior left middle",
    "P L L": "posterior left lower",
    "A R": "anterior right",
    "A L": "anterior left",
    # Extended locations found in actual data
    "A R U": "anterior right upper",
    "A R M": "anterior right middle",
    "A R L": "anterior right lower",
    "A L U": "anterior left upper",
    "A L M": "anterior left middle",
    "A L L": "anterior left lower",
    "P L R": "posterior left lower",      # Likely abbreviation variant
    "P L L R": "posterior left lower",    # Likely abbreviation variant
}


def compute_age(birthdate_str: str, recording_date_str: str) -> Optional[int]:
    """Compute age from birthdate and recording date strings (YYYY-MM-DD)."""
    try:
        bdate = datetime.strptime(birthdate_str, "%Y-%m-%d")
        rdate = datetime.strptime(recording_date_str, "%Y-%m-%d")
        age = rdate.year - bdate.year
        if (rdate.month, rdate.day) < (bdate.month, bdate.day):
            age -= 1
        if age < 0 or age > 120:
            return None
        return age
    except (ValueError, TypeError):
        return None


def build_svd_aufnahme_map() -> dict:
    """
    Parse all overview.csv files from SVD zip archives.
    Returns dict: AufnahmeID -> {gender, birthdate, recording_date}
    """
    aufnahme_map = {}
    zips = [f for f in os.listdir(SVD_ARCHIVES_DIR) if f.endswith(".zip")]

    for zname in sorted(zips):
        zpath = os.path.join(SVD_ARCHIVES_DIR, zname)
        try:
            with zipfile.ZipFile(zpath) as z:
                content = z.read("overview.csv").decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(content))
                for row in reader:
                    aid = row["AufnahmeID"].strip()
                    gender = row["Geschlecht"].strip()
                    bdate = row["Geburtsdatum"].strip()
                    rdate = row["AufnahmeDatum"].strip()
                    if aid not in aufnahme_map:
                        aufnahme_map[aid] = {
                            "gender": gender,
                            "birthdate": bdate,
                            "recording_date": rdate,
                        }
        except Exception as e:
            print(f"  Warning: could not read {zname}: {e}")

    return aufnahme_map


def rebuild_svd_dms():
    """Rebuild SVD DMS template with gender and age from overview.csv."""
    print("=== Rebuilding SVD DMS ===")

    # Load existing DMS to preserve label/split/source structure
    with open(SVD_DMS_PATH) as f:
        existing_dms = json.load(f)
    print(f"  Existing DMS entries: {len(existing_dms)}")

    # Load mel cache metadata for subject_id (= AufnahmeID) mapping
    with open(SVD_META_PATH) as f:
        meta = json.load(f)
    samples = meta["samples"]
    # Build path -> sample lookup
    path_to_sample = {s["path"]: s for s in samples}
    print(f"  Metadata samples: {len(samples)}")

    # Build AufnahmeID -> demographics from zip archives
    aufnahme_map = build_svd_aufnahme_map()
    print(f"  AufnahmeIDs from archives: {len(aufnahme_map)}")

    # Rebuild each DMS entry
    new_dms = []
    stats = {"matched": 0, "no_meta": 0, "no_csv": 0, "no_age": 0}

    for entry in existing_dms:
        mel_path = entry["mel_path"]
        new_entry = {
            "mel_path": mel_path,
            "label": entry["label"],
            "label_name": entry["label_name"],
            "split": entry["split"],
            "dms_text": "",  # Will be filled below
            "source": "svd",
        }

        # Look up in metadata
        sample = path_to_sample.get(mel_path)
        if sample is None:
            stats["no_meta"] += 1
            new_entry["dms_text"] = "No clinical information available."
            new_dms.append(new_entry)
            continue

        subject_id = sample["subject_id"]  # This is AufnahmeID

        # Look up in CSV data
        csv_info = aufnahme_map.get(subject_id)
        if csv_info is None:
            stats["no_csv"] += 1
            new_entry["dms_text"] = "No clinical information available."
            new_dms.append(new_entry)
            continue

        # Gender
        raw_gender = csv_info["gender"]
        if raw_gender == "m":
            gender = "Male"
        elif raw_gender == "w":
            gender = "Female"
        else:
            gender = "Unknown"

        # Age
        age = compute_age(csv_info["birthdate"], csv_info["recording_date"])

        if age is not None:
            new_entry["dms_text"] = f"Gender: {gender}. Age: {age}."
        else:
            stats["no_age"] += 1
            new_entry["dms_text"] = f"Gender: {gender}."

        stats["matched"] += 1
        new_dms.append(new_entry)

    # Write output
    with open(SVD_DMS_PATH, "w") as f:
        json.dump(new_dms, f, indent=2, ensure_ascii=False)

    print(f"  Stats: {stats}")
    print(f"  Written {len(new_dms)} entries to {SVD_DMS_PATH}")

    # Show sample entries
    print("  Sample entries:")
    for e in new_dms[:3]:
        print(f"    {e['mel_path']}: {e['dms_text']}")

    # Check unique DMS texts
    unique = set(e["dms_text"] for e in new_dms)
    print(f"  Unique DMS texts: {len(unique)}")
    # Verify no label leakage
    for text in unique:
        lower = text.lower()
        if "healthy" in lower or "patholog" in lower or "diagnos" in lower:
            print(f"  WARNING: Possible label leakage in: {text}")


def parse_kauh_filename(filepath: str) -> Optional[dict]:
    """
    Parse KAUH filename to extract demographics.
    Format: BP{id}_{Disease},{AuscultationType},{Location},{Age},{Gender}.wav
    """
    fname = os.path.basename(filepath)
    if not fname.endswith(".wav"):
        return None

    fname_no_ext = fname[:-4]  # Remove .wav

    # Split by comma - the format is: BP{id}_{Disease},{AuscType},{Location},{Age},{Gender}
    parts = fname_no_ext.split(",")
    if len(parts) < 5:
        return None

    # Gender is the last part
    gender_raw = parts[-1].strip()

    # Age is the second-to-last part
    age_raw = parts[-2].strip()

    # Location is the third-to-last part
    location_raw = parts[-3].strip()

    # Parse gender
    if gender_raw == "M":
        gender = "Male"
    elif gender_raw == "F":
        gender = "Female"
    else:
        gender = "Unknown"

    # Parse age
    try:
        age = int(age_raw)
    except ValueError:
        age = None

    # Map location
    location = KAUH_LOCATION_MAP.get(location_raw)
    if location is None:
        loc_parts = location_raw.split()
        desc_parts = []
        position_map = {"P": "posterior", "A": "anterior"}
        side_map = {"R": "right", "L": "left"}
        level_map = {"U": "upper", "M": "middle", "L": "lower"}
        maps = [position_map, side_map, level_map]
        for i, p in enumerate(loc_parts):
            mapping = maps[i] if i < len(maps) else {}
            desc_parts.append(mapping.get(p, p))
        location = " ".join(desc_parts)

    return {"gender": gender, "age": age, "location": location}


def rebuild_kauh_dms():
    """Rebuild KAUH DMS template with gender, age, and recording location from filenames."""
    print("\n=== Rebuilding KAUH DMS ===")

    # Load existing DMS
    with open(KAUH_DMS_PATH) as f:
        existing_dms = json.load(f)
    print(f"  Existing DMS entries: {len(existing_dms)}")

    # Load mel cache metadata
    with open(KAUH_META_PATH) as f:
        meta = json.load(f)
    samples = meta["samples"]
    path_to_sample = {s["path"]: s for s in samples}
    print(f"  Metadata samples: {len(samples)}")

    new_dms = []
    stats = {"matched": 0, "no_meta": 0, "parse_fail": 0}

    for entry in existing_dms:
        mel_path = entry["mel_path"]
        new_entry = {
            "mel_path": mel_path,
            "label": entry["label"],
            "label_name": entry["label_name"],
            "split": entry["split"],
            "dms_text": "",
            "source": "kauh",
        }

        sample = path_to_sample.get(mel_path)
        if sample is None:
            stats["no_meta"] += 1
            new_entry["dms_text"] = "No clinical information available."
            new_dms.append(new_entry)
            continue

        parsed = parse_kauh_filename(sample["original_path"])
        if parsed is None:
            stats["parse_fail"] += 1
            new_entry["dms_text"] = "No clinical information available."
            new_dms.append(new_entry)
            continue

        # Build DMS text
        parts = [f"Gender: {parsed['gender']}."]
        if parsed["age"] is not None:
            parts.append(f"Age: {parsed['age']}.")
        if parsed["location"]:
            parts.append(f"Recording location: {parsed['location']}.")

        new_entry["dms_text"] = " ".join(parts)
        stats["matched"] += 1
        new_dms.append(new_entry)

    # Write output
    with open(KAUH_DMS_PATH, "w") as f:
        json.dump(new_dms, f, indent=2, ensure_ascii=False)

    print(f"  Stats: {stats}")
    print(f"  Written {len(new_dms)} entries to {KAUH_DMS_PATH}")

    # Show sample entries
    print("  Sample entries:")
    for e in new_dms[:5]:
        print(f"    {e['mel_path']}: {e['dms_text']}")

    # Check for label leakage
    unique = set(e["dms_text"] for e in new_dms)
    print(f"  Unique DMS texts: {len(unique)}")
    for text in unique:
        lower = text.lower()
        if any(kw in lower for kw in ["healthy", "normal", "asthma", "copd",
                                        "heart failure", "pneumonia", "disease",
                                        "patholog", "diagnos"]):
            print(f"  WARNING: Possible label leakage in: {text}")


if __name__ == "__main__":
    rebuild_svd_dms()
    rebuild_kauh_dms()
    print("\nDone.")
