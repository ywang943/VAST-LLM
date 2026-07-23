"""
SVD Voice Pathology Detection — MVP-style multi-source (vowel + sentence).

Replicates the MVP (Koudounas et al., INTERSPEECH 2025) protocol on SVD:
  - Per subject: sustained vowel /a/ (normal pitch) + phrase (sentence reading)
  - Binary task: healthy (Normal*) vs pathological (everything else)
  - Speaker-independent 10-fold cross-validation
  - Report AUC (compare to MVP HuBERT IFF-TE = 95.8% on SVD)

Our method vs MVP:
  - MVP: HuBERT 94.6M, fuses vowel+sentence with learned Transformer Encoder
  - Ours: frozen HTS-AT + TPA-CSAF features, fuse vowel+sentence
  - Single-source ablation (vowel-only, phrase-only) shows multi-source gain

Fusion strategies (mirrors MVP):
  - vowel_only / phrase_only  (single-source baselines)
  - concat                    (feature concatenation)
  - mean                      (decision-level averaging)

Usage:
  python scripts/run_svd_mvp_task.py --encoder opera_ct
  python scripts/run_svd_mvp_task.py --encoder v3 \
      --ckpt checkpoints/htsat_lejepa_v3/htsat_lejepa_best.pt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from respvoice.htsat_encoder import build_htsat_encoder

SVD_CACHE = Path("data/mel_cache/svd_full")


def collect_subjects(cache_root):
    """
    Group SVD files by subject. Each subject needs vowel /a/ (normal pitch) + phrase.
    Returns list of {subj, label, vowel_path, phrase_path}.
    """
    raw = json.loads((cache_root / "metadata.json").read_text())
    by_subject = defaultdict(dict)
    labels = {}
    splits = {}
    for item in raw["samples"]:
        subj = item["subject_id"]
        labels[subj] = int(item["label"])
        splits[subj] = item["split"]
        by_subject[subj][item["source"]] = cache_root / item["path"]

    subjects = []
    for subj, parts in by_subject.items():
        if "vowel" in parts and "phrase" in parts:   # require both sources
            subjects.append({
                "subj": subj, "label": labels[subj], "split": splits[subj],
                "vowel": parts["vowel"], "phrase": parts["phrase"],
            })
    return subjects


@torch.no_grad()
def extract_feature(encoder, mel, device):
    """mel (1,64,T) → pooled feature (D,)."""
    mel = mel.unsqueeze(0).to(device)   # (1,1,64,T)
    z = encoder(mel)                     # (1, 64, D)
    return z.mean(dim=1).squeeze(0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="opera_ct", choices=["opera_ct", "checkpoint", "v3"])
    p.add_argument("--ckpt", default=None, help="LeJEPA checkpoint")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--cache", type=Path, default=SVD_CACHE)
    p.add_argument("--out", default="checkpoints/svd_mvp/svd_mvp_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build frozen encoder
    print(f"Building encoder ({args.encoder})...")
    if args.encoder in ("checkpoint", "v3"):
        if not args.ckpt or not Path(args.ckpt).exists():
            raise FileNotFoundError("--ckpt is required for checkpoint encoder")
        encoder = build_htsat_encoder(
            ckpt_path=None, use_csaf=True, freeze_backbone=True,
        )
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = {k.replace("encoder.", "", 1): v for k, v in ckpt["model_state"].items()
                 if k.startswith("encoder.")}
        encoder.load_state_dict(state, strict=False)
        print(f"  Loaded V3 weights from {args.ckpt}")
    else:
        encoder = build_htsat_encoder(use_csaf=True, freeze_backbone=True)
    encoder = encoder.to(device).eval()

    # Collect subjects
    subjects = collect_subjects(args.cache)
    n_health = sum(1 for s in subjects if s["label"] == 0)
    n_path = sum(1 for s in subjects if s["label"] == 1)
    print(f"Subjects with vowel+phrase: {len(subjects)} (healthy={n_health}, pathological={n_path})")
    if n_health < 5 or n_path < 5:
        print("WARNING: not enough subjects per class — download more SVD data first.")
        if len(subjects) == 0:
            return

    # Extract features for vowel and phrase per subject
    print("Extracting features...")
    X_vowel, X_phrase, y, groups = [], [], [], []
    for i, s in enumerate(subjects):
        try:
            mv = torch.load(s["vowel"], map_location="cpu")
            mp = torch.load(s["phrase"], map_location="cpu")
            X_vowel.append(extract_feature(encoder, mv, device))
            X_phrase.append(extract_feature(encoder, mp, device))
            y.append(s["label"]); groups.append(s["subj"])
        except Exception as e:
            print(f"  skip {s['subj']}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(subjects)}")
    X_vowel = np.array(X_vowel); X_phrase = np.array(X_phrase)
    y = np.array(y); groups = np.array(groups)

    # Fusion variants
    variants = {
        "vowel_only":  X_vowel,
        "phrase_only": X_phrase,
        "concat":      np.concatenate([X_vowel, X_phrase], axis=1),
        "mean":        (X_vowel + X_phrase) / 2,
    }

    # Speaker-independent stratified K-fold logistic-regression probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    results = {}
    for name, X in variants.items():
        skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=1337)
        aucs = []
        for tr, te in skf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(sc.transform(X[tr]), y[tr])
            prob = clf.predict_proba(sc.transform(X[te]))[:, 1]
            try:
                aucs.append(roc_auc_score(y[te], prob))
            except Exception:
                pass
        results[name] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                         "n_folds": len(aucs)}
        print(f"  {name:12s}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # Summary
    print("\n" + "=" * 55)
    print("  SVD VOICE PATHOLOGY (MVP-style, speaker-independent)")
    print("=" * 55)
    for name, r in results.items():
        print(f"  {name:12s}: {r['auc_mean']:.3f} ± {r['auc_std']:.3f}")
    print(f"  MVP HuBERT IFF-TE reference: 0.958")
    best = max(results.items(), key=lambda kv: kv[1]["auc_mean"])
    print(f"  Best (ours): {best[0]} = {best[1]['auc_mean']:.3f}")
    multi_gain = results["concat"]["auc_mean"] - max(results["vowel_only"]["auc_mean"],
                                                       results["phrase_only"]["auc_mean"])
    print(f"  Multi-source gain (concat vs best single): +{multi_gain*100:.1f}pp")

    out = {
        "encoder": args.encoder,
        "checkpoint": args.ckpt,
        "cache": str(args.cache),
        "n_subjects": len(y), "n_healthy": int((y == 0).sum()), "n_path": int((y == 1).sum()),
        "results": results,
        "mvp_reference_svd": 0.958,
        "multi_source_gain_pp": round(multi_gain * 100, 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
