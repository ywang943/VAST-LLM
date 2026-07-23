"""
Prepare multiple downstream evaluation caches from Coswara data.

Tasks:
  1. covid_all       - healthy vs all COVID (breathing+cough+voice mixed)
  2. covid_breathing - healthy vs COVID, breathing-deep modality only
  3. covid_cough     - healthy vs COVID, cough-heavy modality only
  4. covid_voice     - healthy vs COVID, vowel-a/e/o modality (VOICE!)
  5. modality_clf    - 5-class: breathing/cough/vowel/counting-normal/counting-fast
  6. covid_severity  - 3-class: healthy / positive_asymp / positive_moderate

Usage: python data/prepare_coswara_tasks.py
"""

import json, random, sys, torch
from pathlib import Path
import librosa, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from respvoice.preprocessing import AudioPreprocessor

SR, N_MELS, WIN_MS, HOP_MS, TARGET_SEC = 16000, 64, 64.0, 32.0, 8.0
preprocessor = AudioPreprocessor(sr=SR, n_mels=N_MELS, win_ms=WIN_MS,
                                  hop_ms=HOP_MS, target_sec=TARGET_SEC)

SOURCE_DIR = Path("data/audio/coswara_full")
META = json.loads((SOURCE_DIR / "metadata.json").read_text())["samples"]


def make_cache(task_name, label_fn, dest, min_class_samples=10, segments=2):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    meta_out_path = dest / "metadata.json"
    if meta_out_path.exists():
        n = len(json.loads(meta_out_path.read_text())["samples"])
        print(f"  {task_name}: already exists ({n} windows)")
        return n

    samples_out, cached, skipped = [], 0, 0
    for s in META:
        label = label_fn(s)
        if label is None:
            skipped += 1; continue
        wav_path = SOURCE_DIR / s["path"]
        if not wav_path.exists():
            skipped += 1; continue
        try:
            wav, _ = librosa.load(str(wav_path), sr=SR, mono=True)
            for si in range(segments):
                start = si * int(TARGET_SEC * SR) // 2
                chunk = wav[start:start + int(TARGET_SEC * SR)]
                if len(chunk) < int(TARGET_SEC * SR) // 4: break
                mel = preprocessor.to_mel(chunk.astype(np.float32))
                fname = f"{task_name}_{cached:06d}.pt"
                torch.save(mel, str(dest / fname))
                samples_out.append({"path": fname, "label": label,
                                     "label_name": str(label),
                                     "split": "train"})
                cached += 1
        except Exception: skipped += 1

    if not samples_out:
        print(f"  {task_name}: no samples!"); return 0

    # Stratified split
    from collections import defaultdict
    by_label = defaultdict(list)
    for s in samples_out: by_label[s["label"]].append(s)

    # Check min samples
    if any(len(v) < min_class_samples for v in by_label.values()):
        print(f"  {task_name}: too few samples per class, skipping")
        return 0

    random.seed(1337)
    train, val, test = [], [], []
    for lbl, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * 0.20))
        n_val  = max(1, int(n * 0.10))
        test  += items[:n_test]
        val   += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]

    for split_name, split_items in [("train", train), ("val", val), ("test", test)]:
        for si in split_items: si["split"] = split_name

    lc = {str(k): len(v) for k, v in by_label.items()}
    meta_out = {"task": task_name, "samples": samples_out, "label_counts": lc,
                "split_counts": {"train": len(train), "val": len(val), "test": len(test)}}
    meta_out_path.write_text(json.dumps(meta_out, indent=2))
    print(f"  {task_name}: {cached} windows | labels={lc} | "
          f"train={len(train)} val={len(val)} test={len(test)}")
    return cached


def main():
    print("Preparing Coswara downstream task caches...")
    print()

    COVID_LABELS = {"positive_mild", "positive_moderate", "positive_asymp"}

    # 1. COVID all modalities
    make_cache(
        "coswara_covid_all",
        lambda s: (0 if s["label_name"] == "healthy" else
                   1 if s["label_name"] in COVID_LABELS else None),
        "data/mel_cache/coswara_covid_all",
    )

    # 2. COVID: breathing-deep only
    make_cache(
        "coswara_covid_breathing",
        lambda s: (0 if s["label_name"] == "healthy" and s["audio_type"] == "breathing-deep"
                   else 1 if s["label_name"] in COVID_LABELS and s["audio_type"] == "breathing-deep"
                   else None),
        "data/mel_cache/coswara_covid_breathing",
    )

    # 3. COVID: cough-heavy only
    make_cache(
        "coswara_covid_cough",
        lambda s: (0 if s["label_name"] == "healthy" and s["audio_type"] == "cough-heavy"
                   else 1 if s["label_name"] in COVID_LABELS and s["audio_type"] == "cough-heavy"
                   else None),
        "data/mel_cache/coswara_covid_cough",
    )

    # 4. COVID: voice (vowel-a/e/o) — our unique VOICE modality contribution
    make_cache(
        "coswara_covid_voice",
        lambda s: (0 if s["label_name"] == "healthy" and s["audio_type"].startswith("vowel")
                   else 1 if s["label_name"] in COVID_LABELS and s["audio_type"].startswith("vowel")
                   else None),
        "data/mel_cache/coswara_covid_voice",
    )

    # 5. Modality classification (5-class)
    MODALITY_MAP = {
        "breathing-deep": 0, "breathing-shallow": 0,  # breathing
        "cough-heavy": 1, "cough-shallow": 1,          # cough
        "vowel-a": 2, "vowel-e": 2, "vowel-o": 2,      # vowel
        "counting-normal": 3, "counting-fast": 3,       # counting
    }
    make_cache(
        "coswara_modality",
        lambda s: MODALITY_MAP.get(s.get("audio_type", ""), None),
        "data/mel_cache/coswara_modality",
        min_class_samples=100,
    )

    # 6. COVID severity 3-class (healthy=0, mild=1, moderate=2)
    make_cache(
        "coswara_severity",
        lambda s: (0 if s["label_name"] == "healthy"
                   else 1 if s["label_name"] in ("positive_mild", "positive_asymp")
                   else 2 if s["label_name"] == "positive_moderate"
                   else None),
        "data/mel_cache/coswara_severity",
    )

    # 7. ICBHI 4-class (all respiratory sounds: Normal/Crackle/Wheeze/Both)
    print()
    print("  Note: ICBHI 4-class requires OPERA icbhi feature labels, check separately")

    print()
    print("Done. Run run_csaf_coughvid.py-style script on new caches.")


if __name__ == "__main__":
    main()
