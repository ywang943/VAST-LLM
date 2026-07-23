"""
Build DMS (Demographics-Medical history-Symptoms) text templates for each dataset.
Following RespLLM's approach: metadata fields → natural language description.

Outputs a unified JSON file with all samples' DMS text and task prompts.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_icbhi_dms(metadata_path, annotations_dir=None):
    """Build DMS text for ICBHI.
    ICBHI has: patient_id (from filename), recording location, device.
    Disease label comes from patient diagnosis file.
    """
    meta = json.loads(Path(metadata_path).read_text())
    samples = meta.get("samples", [])

    # ICBHI patient demographics from the official demographic_info.txt
    demo_path = ROOT / "opera_src/datasets/icbhi/ICBHI_Challenge_demographic_information.txt"
    demographics = {}
    if demo_path.exists():
        for line in demo_path.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 4:
                pid = parts[0]
                age = parts[1]
                sex = parts[2]
                bmi_adult = parts[3] if len(parts) > 3 else "unknown"
                demographics[pid] = {"age": age, "sex": sex, "bmi": bmi_adult}

    results = []
    for sample in samples:
        orig = sample.get("original_path", "")
        rec_name = Path(orig).stem if orig else sample["path"].rsplit(".", 1)[0]

        # Extract patient ID and recording location from ICBHI filename
        # Format: PatientID_RecordingIndex_ChestLocation_AcquisitionMode_Device
        parts = rec_name.split("_")
        patient_id = parts[0] if parts else "unknown"
        chest_loc_map = {
            "Tc": "trachea", "Al": "anterior left", "Ar": "anterior right",
            "Pl": "posterior left", "Pr": "posterior right",
            "Ll": "lateral left", "Lr": "lateral right",
        }
        chest_loc = "unknown"
        if len(parts) >= 3:
            chest_loc = chest_loc_map.get(parts[2], parts[2])

        demo = demographics.get(patient_id, {})
        age = demo.get("age", "unknown")
        sex_map = {"M": "Male", "F": "Female"}
        sex = sex_map.get(demo.get("sex", ""), "unknown")

        dms_parts = []
        if sex != "unknown":
            dms_parts.append(f"Gender: {sex}.")
        if age != "unknown":
            dms_parts.append(f"Age: {age}.")
        if demo.get("bmi", "unknown") != "unknown":
            dms_parts.append(f"BMI: {demo['bmi']}.")
        dms_parts.append(f"Recording location: {chest_loc}.")

        dms_text = " ".join(dms_parts)
        results.append({
            "mel_path": sample["path"],
            "label": sample.get("label"),
            "label_name": sample.get("label_name"),
            "split": sample.get("split"),
            "dms_text": dms_text,
            "patient_id": patient_id,
            "source": "icbhi",
        })

    return results


def build_coswara_dms(metadata_path):
    """Build DMS text for Coswara.
    Coswara has: age (a), gender (g), covid_status, smoker, symptoms, etc.
    """
    meta = json.loads(Path(metadata_path).read_text())
    samples = meta.get("samples", [])

    results = []
    for sample in samples:
        sex_map = {"male": "Male", "female": "Female"}
        sex = sex_map.get(str(sample.get("g", "")).lower(), "unknown")
        age = sample.get("a")

        dms_parts = []
        if sex != "unknown":
            dms_parts.append(f"Gender: {sex}.")
        if age is not None:
            dms_parts.append(f"Age: {age}.")

        # Symptoms
        symptoms = []
        if sample.get("cough"):
            symptoms.append("cough")
        if sample.get("fever"):
            symptoms.append("fever")
        if sample.get("cold"):
            symptoms.append("cold")
        if sample.get("diarrhoea"):
            symptoms.append("diarrhoea")
        if sample.get("loss_of_smell"):
            symptoms.append("loss of smell")
        if sample.get("ftg"):
            symptoms.append("fatigue")
        if sample.get("st"):
            symptoms.append("sore throat")

        if symptoms:
            dms_parts.append(f"Symptoms: {', '.join(symptoms)}.")
        else:
            dms_parts.append("No reported symptoms.")

        # Pre-existing conditions
        conditions = []
        if sample.get("asthma"):
            conditions.append("asthma")
        if sample.get("diabetes"):
            conditions.append("diabetes")
        if sample.get("ht"):
            conditions.append("hypertension")
        if sample.get("ihd"):
            conditions.append("ischemic heart disease")
        if sample.get("cld"):
            conditions.append("chronic lung disease")

        if conditions:
            dms_parts.append(f"Pre-existing conditions: {', '.join(conditions)}.")
        else:
            dms_parts.append("No pre-existing conditions reported.")

        smoker = sample.get("smoker")
        if smoker in (True, "y", "True"):
            dms_parts.append("Patient is a smoker.")
        elif smoker in (False, "n", "False"):
            dms_parts.append("Patient is a non-smoker.")

        dms_text = " ".join(dms_parts)

        # Determine COVID label
        covid_status = sample.get("covid_status", "unknown")
        if covid_status == "healthy":
            covid_label = 0
        elif covid_status.startswith("positive"):
            covid_label = 1
        else:
            covid_label = -1  # skip for COVID task

        # Determine Smoker label
        if smoker in (True, "y", "True"):
            smoker_label = 1
        elif smoker in (False, "n", "False"):
            smoker_label = 0
        else:
            smoker_label = -1

        results.append({
            "sample_id": sample.get("id", ""),
            "covid_status": covid_status,
            "covid_label": covid_label,
            "smoker_label": smoker_label,
            "dms_text": dms_text,
            "split": sample.get("_split", "train"),
            "source": "coswara",
        })

    return results


def build_svd_dms(metadata_path):
    """Build DMS text for SVD (Saarbruecken Voice Database).
    SVD has: age, sex, diagnosis, recording type.
    """
    meta = json.loads(Path(metadata_path).read_text())
    samples = meta.get("samples", [])

    results = []
    for sample in samples:
        dms_parts = []

        sex = sample.get("sex", sample.get("gender", "unknown"))
        if sex in ("m", "M", "male"):
            dms_parts.append("Gender: Male.")
        elif sex in ("f", "F", "female"):
            dms_parts.append("Gender: Female.")

        age = sample.get("age")
        if age is not None and age != "unknown":
            dms_parts.append(f"Age: {age}.")

        diagnosis = sample.get("diagnosis", sample.get("label_name", "unknown"))
        if diagnosis and diagnosis != "unknown":
            dms_parts.append(f"Diagnosis: {diagnosis}.")

        rec_type = sample.get("recording_type", "unknown")
        if rec_type != "unknown":
            dms_parts.append(f"Recording type: {rec_type}.")

        dms_text = " ".join(dms_parts) if dms_parts else "No clinical information available."
        results.append({
            "mel_path": sample.get("path"),
            "label": sample.get("label"),
            "label_name": sample.get("label_name"),
            "split": sample.get("split"),
            "dms_text": dms_text,
            "source": "svd",
        })

    return results


def build_kauh_dms(metadata_path):
    """Build DMS text for KAUH dataset."""
    meta = json.loads(Path(metadata_path).read_text())
    samples = meta.get("samples", [])

    results = []
    for sample in samples:
        dms_parts = []

        sex = sample.get("sex", sample.get("gender", "unknown"))
        if sex in ("m", "M", "male", "Male"):
            dms_parts.append("Gender: Male.")
        elif sex in ("f", "F", "female", "Female"):
            dms_parts.append("Gender: Female.")

        age = sample.get("age")
        if age is not None:
            dms_parts.append(f"Age: {age}.")

        dms_text = " ".join(dms_parts) if dms_parts else "No clinical information available."
        results.append({
            "mel_path": sample.get("path"),
            "label": sample.get("label"),
            "label_name": sample.get("label_name"),
            "split": sample.get("split"),
            "dms_text": dms_text,
            "source": "kauh",
        })

    return results


# Task instruction templates (following RespLLM format)
TASK_TEMPLATES = {
    "icbhi_copd": {
        "instruction": (
            "Dataset description: This data comes from the ICBHI Respiratory Sound Database. "
            "Task description: classify whether the person has COPD given the following "
            "information and audio of the person's lung sounds. "
            "Please output 1 for COPD, and 0 for healthy."
        ),
        "labels": {0: "healthy", 1: "COPD"},
    },
    "coswara_covid": {
        "instruction": (
            "Dataset description: This data comes from the Coswara dataset. "
            "Task description: classify whether the participant has COVID-19 given the "
            "following information and audio recording. "
            "Please output 1 for COVID-19 positive, and 0 for healthy."
        ),
        "labels": {0: "healthy", 1: "COVID-19 positive"},
    },
    "coswara_smoker": {
        "instruction": (
            "Dataset description: This data comes from the Coswara dataset. "
            "Task description: classify whether the participant is a smoker given the "
            "following information and audio recording. "
            "Please output 1 for smoker, and 0 for non-smoker."
        ),
        "labels": {0: "non-smoker", 1: "smoker"},
    },
    "svd_pathology": {
        "instruction": (
            "Dataset description: This data comes from the Saarbruecken Voice Database (SVD). "
            "Task description: classify whether the voice recording indicates vocal fold "
            "pathology given the following information and audio. "
            "Please output 1 for pathological voice, and 0 for healthy voice."
        ),
        "labels": {0: "healthy voice", 1: "voice pathology"},
    },
    "kauh_obstructive": {
        "instruction": (
            "Dataset description: This data comes from the KAUH respiratory dataset. "
            "Task description: classify whether the person has obstructive airway disease "
            "given the following information and audio of the person's lung sounds. "
            "Please output 1 for obstructive disease, and 0 for healthy."
        ),
        "labels": {0: "healthy", 1: "obstructive disease"},
    },
}


def main():
    output_dir = ROOT / "data/dms_templates"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ICBHI
    icbhi_meta = ROOT / "data/mel_cache/opera_icbhi_disease/metadata.json"
    if icbhi_meta.exists():
        print("Building ICBHI DMS...")
        icbhi_dms = build_icbhi_dms(icbhi_meta)
        all_results["icbhi"] = icbhi_dms
        print(f"  {len(icbhi_dms)} samples")

    # Coswara
    coswara_meta = ROOT / "data/coswara_full/metadata.json"
    if coswara_meta.exists():
        print("Building Coswara DMS...")
        coswara_dms = build_coswara_dms(coswara_meta)
        all_results["coswara"] = coswara_dms
        covid_valid = [s for s in coswara_dms if s["covid_label"] >= 0]
        smoker_valid = [s for s in coswara_dms if s["smoker_label"] >= 0]
        print(f"  {len(coswara_dms)} total, {len(covid_valid)} COVID-valid, "
              f"{len(smoker_valid)} smoker-valid")

    # SVD
    svd_meta = ROOT / "data/mel_cache/svd_full/metadata.json"
    if svd_meta.exists():
        print("Building SVD DMS...")
        svd_dms = build_svd_dms(svd_meta)
        all_results["svd"] = svd_dms
        print(f"  {len(svd_dms)} samples")

    # KAUH
    kauh_meta = ROOT / "data/mel_cache/opera_kauh/metadata.json"
    if kauh_meta.exists():
        print("Building KAUH DMS...")
        kauh_dms = build_kauh_dms(kauh_meta)
        all_results["kauh"] = kauh_dms
        print(f"  {len(kauh_dms)} samples")

    # Save
    for key, data in all_results.items():
        out_path = output_dir / f"{key}_dms.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Saved {out_path}")

    # Save task templates
    (output_dir / "task_templates.json").write_text(
        json.dumps(TASK_TEMPLATES, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved task templates to {output_dir / 'task_templates.json'}")

    # Summary
    print(f"\n{'='*60}")
    print("DMS Template Summary")
    print(f"{'='*60}")
    for key, data in all_results.items():
        print(f"\n{key}:")
        print(f"  Samples: {len(data)}")
        if data:
            print(f"  Example DMS: {data[0]['dms_text']}")


if __name__ == "__main__":
    main()
