#!/usr/bin/env python3
"""
Prepare Bridge2AI Voice v3.1 mel caches and DMS template for the RQ3 LLM pipeline.

Creates:
  1. data/mel_cache/b2ai_voice_pathology/  — binary control vs any pathology (SEEN, S7)
  2. data/mel_cache/b2ai_laryngeal_cancer/ — zero-shot target (T2)
  3. data/mel_cache/b2ai_benign_lesions/   — zero-shot target (T3)
  4. data/mel_cache/b2ai_laryngeal_dystonia/ — zero-shot target (T4)
  5. data/dms_templates/b2ai_dms.json      — DMS template for the main pathology task
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# ──────────────────── Config ────────────────────
BASE_DIR = "/hpc2hdd/home/ywang943/KDD_Audio"
B2AI_DIR = os.path.join(BASE_DIR, "data/b2ai-voice_3.1")
MEL_CACHE_DIR = os.path.join(BASE_DIR, "data/mel_cache")
DMS_DIR = os.path.join(BASE_DIR, "data/dms_templates")

TARGET_MEL_BINS = 64   # our model expects 64
TARGET_FRAMES = 251    # 8s @ 16kHz / 512 hop + 1
SEED = 42

# ──────────────────── Load data ────────────────────
print("Loading mel spectrogram parquet...")
mel_df = pd.read_parquet(
    os.path.join(B2AI_DIR, "features/torchaudio_mel_spectrogram.parquet")
)
# Filter to prolonged-vowel only
mel_df = mel_df[mel_df["task_name"] == "prolonged-vowel"].reset_index(drop=True)
print(f"  prolonged-vowel recordings: {len(mel_df)}")

# Convert participant_id from zero-padded string to int for matching
mel_df["pid_int"] = mel_df["participant_id"].apply(int)

# ──────────────────── Load diagnoses ────────────────────
diag_dir = os.path.join(B2AI_DIR, "phenotype/diagnosis")

# Pathology categories we use
PATHOLOGY_CATS = [
    "airway_stenosis", "parkinsons_disease", "laryngeal_dystonia",
    "unilateral_vocal_fold_paralysis", "benign_lesions",
    "muscle_tension_dysphonia", "laryngeal_cancer", "precancerous_lesions",
]

# Load control IDs
ctrl_df = pd.read_csv(os.path.join(diag_dir, "control.tsv"), sep="\t")
control_pids = set(ctrl_df["participant_id"].astype(int))

# Load all pathology IDs with their category
pid_to_diag = {}  # pid -> diagnosis category name
pathology_all_pids = set()
for cat in PATHOLOGY_CATS:
    df = pd.read_csv(os.path.join(diag_dir, f"{cat}.tsv"), sep="\t")
    pids = set(df["participant_id"].astype(int))
    pathology_all_pids |= pids
    for pid in pids:
        if pid not in pid_to_diag:
            pid_to_diag[pid] = cat

# Remove controls that also appear in pathology (2 participants)
overlap = control_pids & pathology_all_pids
if overlap:
    print(f"  Removing {len(overlap)} controls that overlap with pathology: {overlap}")
    control_pids -= overlap

# Mark controls
for pid in control_pids:
    pid_to_diag[pid] = "control"

print(f"  Controls: {len(control_pids)}, Pathology: {len(pathology_all_pids)}")

# ──────────────────── Build DMS text ────────────────────
def clean_value(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "no answer"}:
        return None
    return value


def load_table_lookup(rel_path):
    path = os.path.join(B2AI_DIR, rel_path)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    df["participant_id"] = df["participant_id"].astype(int)
    return {int(row["participant_id"]): row for _, row in df.iterrows()}


def add_field(parts, row, col, label):
    if row is None or col not in row:
        return
    value = clean_value(row.get(col))
    if value is not None:
        parts.append(f"{label}: {value}.")


demo_df = pd.read_csv(
    os.path.join(B2AI_DIR, "phenotype/demographics/demographics.tsv"), sep="\t"
)
demo_df["participant_id"] = demo_df["participant_id"].astype(int)
demo_rows = {int(row["participant_id"]): row for _, row in demo_df.iterrows()}
conf_rows = load_table_lookup("phenotype/confounders/confounders.tsv")
vhi_rows = load_table_lookup("phenotype/questionnaire/vhi10.tsv")
voice_rows = load_table_lookup("phenotype/questionnaire/voice_perception.tsv")
dyspnea_rows = load_table_lookup("phenotype/questionnaire/dyspnea_index.tsv")
cough_rows = load_table_lookup("phenotype/questionnaire/leicester_cough_questionnaire.tsv")


def build_dms_text(pid):
    parts = []

    demo = demo_rows.get(pid)
    if demo is not None:
        sex = clean_value(demo.get("sex_at_birth"))
        age = demo.get("age")
        if sex:
            parts.append(f"Gender: {sex}.")
        if pd.notna(age):
            try:
                parts.append(f"Age: {int(float(age))}.")
            except (ValueError, TypeError):
                pass

    voice = voice_rows.get(pid)
    add_field(parts, voice, "voice_quality_perception", "Self-rated voice quality")

    vhi = vhi_rows.get(pid)
    add_field(parts, vhi, "vhi_10_calc_score", "Voice handicap index score")
    add_field(parts, vhi, "strain_voice", "Voice strain")
    add_field(parts, vhi, "tough_to_understand", "Difficulty being understood")
    add_field(parts, vhi, "voice_difficult_hear", "Difficulty being heard")

    conf = conf_rows.get(pid)
    add_field(parts, conf, "coughing_clearing_throat", "Coughing or throat clearing")
    add_field(parts, conf, "shortness_breath", "Shortness of breath")
    add_field(parts, conf, "scratchy_sore_throat", "Scratchy or sore throat")
    add_field(parts, conf, "hydration", "Hydration")
    add_field(parts, conf, "hours_voice_activity", "Daily voice activity hours")
    add_field(parts, conf, "current_use_nicotine_products", "Current nicotine use")
    add_field(parts, conf, "alcohol_yn", "Alcohol use")
    add_field(parts, conf, "reflux_medications", "Reflux medication use")
    add_field(parts, conf, "seasonal_allergies", "Seasonal allergies")

    dyspnea = dyspnea_rows.get(pid)
    add_field(parts, dyspnea, "di_effort_breathe", "Effort to breathe")
    add_field(parts, dyspnea, "di_tightness_throat", "Throat tightness")
    add_field(parts, dyspnea, "di_sound_breathing_in", "Noisy breathing")

    cough = cough_rows.get(pid)
    add_field(parts, cough, "lcq_hoarse_voice", "Hoarse voice")
    add_field(parts, cough, "lcq_sputum_phlegm", "Sputum or phlegm")
    add_field(parts, cough, "lcq_sleep", "Cough affects sleep")

    return " ".join(parts) if parts else "No clinical information available."


demo_lookup = {}
all_dms_pids = (
    set(demo_rows) | set(conf_rows) | set(vhi_rows) | set(voice_rows)
    | set(dyspnea_rows) | set(cough_rows)
)
for pid in all_dms_pids:
    demo_lookup[pid] = build_dms_text(pid)

DEFAULT_DMS = "No clinical information available."


# ──────────────────── Helper: process mel ────────────────────
def process_mel(mel_raw, n_frames):
    """
    Convert raw mel (60, T) to tensor (1, 64, 251).
    - Interpolate mel bins 60 -> 64
    - Crop/pad frames to 251
    """
    # mel_raw is a numpy array of shape (60,) where each element is an array of length T
    mel_2d = np.stack(mel_raw)  # (60, T)
    mel_tensor = torch.tensor(mel_2d, dtype=torch.float32)  # (60, T)
    mel_tensor = torch.log1p(torch.clamp(mel_tensor, min=0.0))

    # Interpolate mel bins: 60 -> 64
    # F.interpolate expects (N, C, L) for 1D or (N, C, H, W) for 2D
    # We treat this as 1D interpolation along the mel dimension
    # Reshape to (1, 1, 60, T) and interpolate to (1, 1, 64, T)
    mel_4d = mel_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 60, T)
    mel_resized = F.interpolate(mel_4d, size=(TARGET_MEL_BINS, mel_tensor.shape[1]),
                                mode="bilinear", align_corners=False)
    mel_resized = mel_resized.squeeze(0)  # (1, 64, T)

    T = mel_resized.shape[2]

    if T >= TARGET_FRAMES:
        # Center crop
        start = (T - TARGET_FRAMES) // 2
        mel_out = mel_resized[:, :, start:start + TARGET_FRAMES]
    else:
        # Zero-pad symmetrically
        pad_total = TARGET_FRAMES - T
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        mel_out = F.pad(mel_resized, (pad_left, pad_right), mode="constant", value=0.0)

    mel_out = (mel_out - mel_out.mean()) / (mel_out.std() + 1e-8)
    return mel_out  # (1, 64, 251)


# ──────────────────── Helper: create mel cache ────────────────────
def create_mel_cache(task_name, samples_info, output_dir, n_classes=2):
    """
    Save mel .pt files and metadata.json.

    samples_info: list of dicts with keys:
        mel_row_idx, label, label_name, split, dms_text, diagnosis, participant_id
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata_samples = []
    for i, info in enumerate(samples_info):
        idx = info["mel_row_idx"]
        row = mel_df_labeled.iloc[idx]
        mel_raw = row["mel_spectrogram"]
        n_frames = row["n_frames"]

        mel_tensor = process_mel(mel_raw, n_frames)
        fname = f"b2ai_{i:05d}.pt"
        torch.save(mel_tensor, os.path.join(output_dir, fname))

        metadata_samples.append({
            "path": fname,
            "label": info["label"],
            "label_name": info["label_name"],
            "split": info["split"],
            "dms_text": info["dms_text"],
            "diagnosis": info["diagnosis"],
            "participant_id": int(info["participant_id"]),
            "mel_row_idx": int(info["mel_row_idx"]),
        })

    metadata = {
        "task": task_name,
        "n_classes": n_classes,
        "samples": metadata_samples,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {len(metadata_samples)} samples to {output_dir}")
    return metadata_samples


# ──────────────────── Filter mel_df to known diagnosis participants ────────────────────
# Only keep rows where participant has a known diagnosis
mel_df["diagnosis"] = mel_df["pid_int"].map(pid_to_diag)
mel_df_labeled = mel_df[mel_df["diagnosis"].notna()].reset_index(drop=True)
print(f"  Labeled prolonged-vowel recordings: {len(mel_df_labeled)}")
print(f"  Diagnosis distribution:")
print(mel_df_labeled["diagnosis"].value_counts().to_string())
print()


# ======================================================================
# TASK 1: b2ai_voice_pathology — binary control(0) vs any_pathology(1)
# ======================================================================
print("=" * 60)
print("Creating b2ai_voice_pathology mel cache (S7 - SEEN task)")
print("=" * 60)

# Build sample list
pathology_samples = []
for idx, row in mel_df_labeled.iterrows():
    pid = row["pid_int"]
    diag = row["diagnosis"]
    label = 0 if diag == "control" else 1
    label_name = "control" if label == 0 else "pathology"
    dms_text = demo_lookup.get(pid, DEFAULT_DMS)

    pathology_samples.append({
        "mel_row_idx": idx,
        "label": label,
        "label_name": label_name,
        "split": None,  # will be assigned below
        "dms_text": dms_text,
        "diagnosis": diag,
        "participant_id": pid,
    })

# Stratified split: 70% train, 10% val, 20% test
labels = [s["label"] for s in pathology_samples]
np.random.seed(SEED)

indices = list(range(len(pathology_samples)))
# First split: 80% train+val, 20% test
train_val_idx, test_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=SEED
)
# Second split: from the 80%, take 10/80 = 12.5% as val
train_val_labels = [labels[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.125, stratify=train_val_labels, random_state=SEED
)

for i in train_idx:
    pathology_samples[i]["split"] = "train"
for i in val_idx:
    pathology_samples[i]["split"] = "val"
for i in test_idx:
    pathology_samples[i]["split"] = "test"

# Print stats
split_counts = {}
for s in pathology_samples:
    key = (s["split"], s["label_name"])
    split_counts[key] = split_counts.get(key, 0) + 1
print("  Split distribution:")
for k in sorted(split_counts.keys()):
    print(f"    {k}: {split_counts[k]}")

output_dir = os.path.join(MEL_CACHE_DIR, "b2ai_voice_pathology")
main_meta = create_mel_cache("b2ai_voice_pathology", pathology_samples, output_dir)


# ======================================================================
# TASK 2: b2ai_laryngeal_cancer — zero-shot (T2)
# control (~20) vs laryngeal_cancer + precancerous_lesions (17)
# ======================================================================
print()
print("=" * 60)
print("Creating b2ai_laryngeal_cancer mel cache (T2 - zero-shot)")
print("=" * 60)

cancer_cats = {"laryngeal_cancer", "precancerous_lesions"}
cancer_rows = mel_df_labeled[mel_df_labeled["diagnosis"].isin(cancer_cats)].copy()
control_rows = mel_df_labeled[mel_df_labeled["diagnosis"] == "control"].copy()

n_cancer = len(cancer_rows)
print(f"  Cancer/precancerous samples: {n_cancer}")

# Sample ~20 controls to roughly match (17 pathology -> ~20 controls)
np.random.seed(SEED)
control_subset = control_rows.sample(n=min(20, len(control_rows)), random_state=SEED)

cancer_samples = []
for idx, row in control_subset.iterrows():
    pid = row["pid_int"]
    dms_text = demo_lookup.get(pid, DEFAULT_DMS)
    cancer_samples.append({
        "mel_row_idx": idx,
        "label": 0,
        "label_name": "control",
        "split": "test",
        "dms_text": dms_text,
        "diagnosis": "control",
        "participant_id": pid,
    })

for idx, row in cancer_rows.iterrows():
    pid = row["pid_int"]
    dms_text = demo_lookup.get(pid, DEFAULT_DMS)
    cancer_samples.append({
        "mel_row_idx": idx,
        "label": 1,
        "label_name": "laryngeal_cancer",
        "split": "test",
        "dms_text": dms_text,
        "diagnosis": row["diagnosis"],
        "participant_id": pid,
    })

print(f"  Total samples (control + cancer): {len(cancer_samples)}")
output_dir = os.path.join(MEL_CACHE_DIR, "b2ai_laryngeal_cancer")
create_mel_cache("b2ai_laryngeal_cancer", cancer_samples, output_dir)


# ======================================================================
# TASK 3: b2ai_benign_lesions — zero-shot (T3)
# control (~55) vs benign_lesions (55)
# ======================================================================
print()
print("=" * 60)
print("Creating b2ai_benign_lesions mel cache (T3 - zero-shot)")
print("=" * 60)

lesion_rows = mel_df_labeled[mel_df_labeled["diagnosis"] == "benign_lesions"].copy()
n_lesions = len(lesion_rows)
print(f"  Benign lesion samples: {n_lesions}")

np.random.seed(SEED)
control_subset_bl = control_rows.sample(n=min(n_lesions, len(control_rows)), random_state=SEED)

lesion_samples = []
for idx, row in control_subset_bl.iterrows():
    pid = row["pid_int"]
    dms_text = demo_lookup.get(pid, DEFAULT_DMS)
    lesion_samples.append({
        "mel_row_idx": idx,
        "label": 0,
        "label_name": "control",
        "split": "test",
        "dms_text": dms_text,
        "diagnosis": "control",
        "participant_id": pid,
    })

for idx, row in lesion_rows.iterrows():
    pid = row["pid_int"]
    dms_text = demo_lookup.get(pid, DEFAULT_DMS)
    lesion_samples.append({
        "mel_row_idx": idx,
        "label": 1,
        "label_name": "benign_lesions",
        "split": "test",
        "dms_text": dms_text,
        "diagnosis": "benign_lesions",
        "participant_id": pid,
    })

print(f"  Total samples (control + lesions): {len(lesion_samples)}")
output_dir = os.path.join(MEL_CACHE_DIR, "b2ai_benign_lesions")
create_mel_cache("b2ai_benign_lesions", lesion_samples, output_dir)


# ======================================================================
# TASK 4: b2ai_laryngeal_dystonia — zero-shot (T4)
# control (~78) vs laryngeal_dystonia (78)
# ======================================================================
print()
print("=" * 60)
print("Creating b2ai_laryngeal_dystonia mel cache (T4 - zero-shot)")
print("=" * 60)

dystonia_rows = mel_df_labeled[mel_df_labeled["diagnosis"] == "laryngeal_dystonia"].copy()
n_dystonia = len(dystonia_rows)
print(f"  Laryngeal dystonia samples: {n_dystonia}")

np.random.seed(SEED)
control_subset_ld = control_rows.sample(n=min(n_dystonia, len(control_rows)), random_state=SEED)

dystonia_samples = []
for idx, row in control_subset_ld.iterrows():
    pid = row["pid_int"]
    dms_text = demo_lookup.get(pid, DEFAULT_DMS)
    dystonia_samples.append({
        "mel_row_idx": idx,
        "label": 0,
        "label_name": "control",
        "split": "test",
        "dms_text": dms_text,
        "diagnosis": "control",
        "participant_id": pid,
    })

for idx, row in dystonia_rows.iterrows():
    pid = row["pid_int"]
    dms_text = demo_lookup.get(pid, DEFAULT_DMS)
    dystonia_samples.append({
        "mel_row_idx": idx,
        "label": 1,
        "label_name": "laryngeal_dystonia",
        "split": "test",
        "dms_text": dms_text,
        "diagnosis": "laryngeal_dystonia",
        "participant_id": pid,
    })

print(f"  Total samples (control + dystonia): {len(dystonia_samples)}")
output_dir = os.path.join(MEL_CACHE_DIR, "b2ai_laryngeal_dystonia")
create_mel_cache("b2ai_laryngeal_dystonia", dystonia_samples, output_dir)


# ======================================================================
# TASK 5: DMS template — data/dms_templates/b2ai_dms.json
# ======================================================================
print()
print("=" * 60)
print("Creating DMS template: b2ai_dms.json")
print("=" * 60)

dms_entries = []
for sample in main_meta:
    dms_entries.append({
        "mel_path": sample["path"],
        "label": sample["label"],
        "label_name": sample["label_name"],
        "split": sample["split"],
        "dms_text": sample["dms_text"],
        "source": "b2ai",
    })

os.makedirs(DMS_DIR, exist_ok=True)
dms_path = os.path.join(DMS_DIR, "b2ai_dms.json")
with open(dms_path, "w") as f:
    json.dump(dms_entries, f, indent=2)

print(f"  Saved {len(dms_entries)} DMS entries to {dms_path}")


# ──────────────────── Summary ────────────────────
print()
print("=" * 60)
print("DONE. Summary:")
print("=" * 60)
for task_dir in ["b2ai_voice_pathology", "b2ai_laryngeal_cancer",
                 "b2ai_benign_lesions", "b2ai_laryngeal_dystonia"]:
    full_path = os.path.join(MEL_CACHE_DIR, task_dir)
    n_pt = len([f for f in os.listdir(full_path) if f.endswith(".pt")])
    print(f"  {task_dir}: {n_pt} .pt files")

print(f"  DMS template: {len(dms_entries)} entries")
print()

# Verify a random .pt file
sample_pt = os.path.join(MEL_CACHE_DIR, "b2ai_voice_pathology", "b2ai_00000.pt")
t = torch.load(sample_pt, weights_only=True)
print(f"  Verification: {sample_pt}")
print(f"    Shape: {t.shape} (expected: [1, 64, 251])")
print(f"    dtype: {t.dtype}")
print(f"    min: {t.min():.4f}, max: {t.max():.4f}")
