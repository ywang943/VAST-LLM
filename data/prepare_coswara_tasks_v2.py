"""
Prepare Coswara downstream mel caches from full extracted data.

Source: data/audio/coswara_full/iiscleap-Coswara-Data-bf300ae/Extracted_data/
Metadata: data/coswara_full/metadata.json

Tasks created:
  coswara_covid_cough     - COVID detection from cough-heavy
  coswara_covid_breathing - COVID detection from breathing-deep
  coswara_smoker_cough    - Smoker detection from cough-heavy
  coswara_smoker_breathing - Smoker detection from breathing-deep

Usage: python data/prepare_coswara_tasks_v2.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from respvoice.preprocessing import AudioPreprocessor

SR, N_MELS, WIN_MS, HOP_MS, TARGET_SEC = 16000, 64, 64.0, 32.0, 8.0
preprocessor = AudioPreprocessor(sr=SR, n_mels=N_MELS, win_ms=WIN_MS,
                                 hop_ms=HOP_MS, target_sec=TARGET_SEC)

ROOT = Path(__file__).parent.parent
EXTRACTED = ROOT / "data/audio/coswara_full/iiscleap-Coswara-Data-bf300ae/Extracted_data"
META_PATH = ROOT / "data/coswara_full/metadata.json"
DMS_PATH = ROOT / "data/dms_templates/coswara_dms.json"


def load_metadata():
    """Load metadata from both consolidated JSON and per-user metadata files."""
    meta = json.loads(META_PATH.read_text())
    participants = {}

    # First pass: consolidated metadata for COVID status, demographics
    for s in meta["samples"]:
        pid = s["id"]
        covid_status = s.get("covid_status", "")
        gender = s.get("g", "")
        age = s.get("a", "")
        symptoms = []
        for sym in ["cold", "cough", "fever", "diarrhoea", "loss_of_smell", "ftg", "st"]:
            if s.get(sym):
                symptoms.append(sym.replace("ftg", "fatigue").replace("st", "sore_throat"))
        conditions = []
        for cond in ["asthma", "diabetes", "ht", "ihd", "cld", "pneumonia"]:
            if s.get(cond):
                conditions.append(cond.replace("ht", "hypertension").replace("ihd", "heart_disease").replace("cld", "chronic_lung_disease"))
        dms = f"Gender: {gender.capitalize()}. Age: {age}."
        if symptoms:
            dms += f" Symptoms: {', '.join(symptoms)}."
        else:
            dms += " No reported symptoms."
        if conditions:
            dms += f" Pre-existing conditions: {', '.join(conditions)}."
        else:
            dms += " No pre-existing conditions reported."
        participants[pid] = {
            "covid_status": covid_status,
            "smoker": None,
            "dms_text": dms,
            "split": s.get("_split", "train"),
        }

    # Second pass: per-user metadata.json for smoker status (not in consolidated)
    for date_dir in EXTRACTED.iterdir():
        if not date_dir.is_dir():
            continue
        for user_dir in date_dir.iterdir():
            if not user_dir.is_dir():
                continue
            pid = user_dir.name
            mj = user_dir / "metadata.json"
            if pid in participants and mj.exists():
                try:
                    d = json.loads(mj.read_text())
                    sm = d.get("smoker")
                    if sm in (True, "True", "y"):
                        participants[pid]["smoker"] = True
                    elif sm in (False, "False", "n"):
                        participants[pid]["smoker"] = False
                except Exception:
                    pass

    n_smoker = sum(1 for p in participants.values() if p["smoker"] is True)
    n_nonsmoker = sum(1 for p in participants.values() if p["smoker"] is False)
    print(f"  Smoker labels: {n_smoker} smokers, {n_nonsmoker} non-smokers, "
          f"{len(participants) - n_smoker - n_nonsmoker} unknown")
    return participants


def find_wav_files(participants):
    """Walk Extracted_data and collect (pid, audio_type, wav_path) tuples."""
    records = []
    for date_dir in sorted(EXTRACTED.iterdir()):
        if not date_dir.is_dir():
            continue
        for user_dir in date_dir.iterdir():
            if not user_dir.is_dir():
                continue
            pid = user_dir.name
            if pid not in participants:
                continue
            for wav_file in user_dir.glob("*.wav"):
                audio_type = wav_file.stem
                records.append((pid, audio_type, wav_file))
    return records


def make_task_cache(task_name, records, label_fn, dest, participants,
                    min_class_samples=20):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    meta_out_path = dest / "metadata.json"
    if meta_out_path.exists():
        existing = json.loads(meta_out_path.read_text())
        n = len(existing.get("samples", []))
        print(f"  {task_name}: already exists ({n} samples)")
        return n

    samples_out = []
    cached = 0
    skipped = 0

    for pid, audio_type, wav_path in records:
        info = participants[pid]
        label = label_fn(info, audio_type)
        if label is None:
            continue
        try:
            wav, _ = librosa.load(str(wav_path), sr=SR, mono=True)
            if len(wav) < SR:
                skipped += 1
                continue
            chunk = wav[:int(TARGET_SEC * SR)]
            if len(chunk) < int(TARGET_SEC * SR):
                chunk = np.pad(chunk, (0, int(TARGET_SEC * SR) - len(chunk)))
            mel = preprocessor.to_mel(chunk.astype(np.float32))
            fname = f"{task_name}_{cached:06d}.pt"
            torch.save(mel, str(dest / fname))
            samples_out.append({
                "path": fname,
                "label": label,
                "label_name": str(label),
                "split": info.get("split", "train"),
                "pid": pid,
                "audio_type": audio_type,
                "dms_text": info.get("dms_text", ""),
            })
            cached += 1
        except Exception as e:
            skipped += 1

    if not samples_out:
        print(f"  {task_name}: no samples!")
        return 0

    by_label = defaultdict(list)
    for s in samples_out:
        by_label[s["label"]].append(s)

    if any(len(v) < min_class_samples for v in by_label.values()):
        print(f"  {task_name}: too few samples per class ({dict((k,len(v)) for k,v in by_label.items())})")
        return 0

    random.seed(42)
    by_pid_split = defaultdict(str)
    for s in samples_out:
        if s["split"] in ("train", "val", "test"):
            by_pid_split[s["pid"]] = s["split"]

    pids_by_label = defaultdict(set)
    for s in samples_out:
        pids_by_label[s["label"]].add(s["pid"])

    for lbl, pids in pids_by_label.items():
        pid_list = sorted(pids)
        random.shuffle(pid_list)
        n = len(pid_list)
        n_test = max(2, int(n * 0.20))
        n_val = max(1, int(n * 0.10))
        test_pids = set(pid_list[:n_test])
        val_pids = set(pid_list[n_test:n_test + n_val])
        train_pids = set(pid_list[n_test + n_val:])
        for s in samples_out:
            if s["label"] != lbl:
                continue
            if s["pid"] in test_pids:
                s["split"] = "test"
            elif s["pid"] in val_pids:
                s["split"] = "val"
            else:
                s["split"] = "train"

    lc = {str(k): len(v) for k, v in by_label.items()}
    split_counts = defaultdict(int)
    for s in samples_out:
        split_counts[s["split"]] += 1

    meta_out = {
        "task": task_name,
        "samples": samples_out,
        "label_counts": lc,
        "split_counts": dict(split_counts),
    }
    meta_out_path.write_text(json.dumps(meta_out, indent=2))
    print(f"  {task_name}: {cached} samples | labels={lc} | splits={dict(split_counts)}")
    return cached


def main():
    print("Loading Coswara metadata...")
    participants = load_metadata()
    print(f"  {len(participants)} participants")

    print("Scanning wav files from Extracted_data...")
    records = find_wav_files(participants)
    print(f"  {len(records)} wav files found")

    covid_positive = {"positive_mild", "positive_moderate", "positive_asymp"}

    print("\nPreparing task caches...")

    # COVID from cough-heavy (matches RespLLM S4: Coswara COVID cough)
    make_task_cache(
        "coswara_covid_cough",
        records,
        lambda info, at: (
            0 if info["covid_status"] == "healthy" and at == "cough-heavy"
            else 1 if info["covid_status"] in covid_positive and at == "cough-heavy"
            else None
        ),
        ROOT / "data/mel_cache/coswara_covid_cough",
        participants,
    )

    # COVID from breathing-deep (matches RespLLM S3: Coswara COVID breath)
    make_task_cache(
        "coswara_covid_breathing",
        records,
        lambda info, at: (
            0 if info["covid_status"] == "healthy" and at == "breathing-deep"
            else 1 if info["covid_status"] in covid_positive and at == "breathing-deep"
            else None
        ),
        ROOT / "data/mel_cache/coswara_covid_breathing",
        participants,
    )

    # Smoker from cough-heavy (matches RespLLM S6: Coswara Smoker cough)
    make_task_cache(
        "coswara_smoker_cough",
        records,
        lambda info, at: (
            1 if info["smoker"] is True and at == "cough-heavy"
            else 0 if info["smoker"] is False and at == "cough-heavy"
            else None
        ),
        ROOT / "data/mel_cache/coswara_smoker_cough",
        participants,
    )

    # Smoker from breathing-deep (matches RespLLM S5: Coswara Smoker breath)
    make_task_cache(
        "coswara_smoker_breathing",
        records,
        lambda info, at: (
            1 if info["smoker"] is True and at == "breathing-deep"
            else 0 if info["smoker"] is False and at == "breathing-deep"
            else None
        ),
        ROOT / "data/mel_cache/coswara_smoker_breathing",
        participants,
    )

    # COVID from all modalities (for generalization test)
    make_task_cache(
        "coswara_covid_all",
        records,
        lambda info, at: (
            0 if info["covid_status"] == "healthy"
            else 1 if info["covid_status"] in covid_positive
            else None
        ),
        ROOT / "data/mel_cache/coswara_covid_all",
        participants,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
