"""
RQ2: Codebook quality and isotropy analysis across different encoders.

For each encoder:
  1. Extract features from mel caches
  2. Compute d_eff (effective rank of covariance)
  3. Train VQ with same protocol → codebook utilization, perplexity
  4. Train linear probe on VQ token histograms → VQ-linear AUROC

Encoders evaluated:
  - AudioSet-pretrained HTS-AT (OPERA backbone before respiratory fine-tuning)
  - OPERA-CT (contrastive learning on respiratory sounds)
  - JEPA without SIGReg (our backbone, JEPA-only pretraining)
  - VAST (Ours) = LeJEPA + SIGReg
  - Wav2vec-BERT, HuBERT-base (HuggingFace, frozen feature extractor)
"""

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.vq import VectorQuantizer

SR = 16000
WAV_LEN = SR * 8

EVAL_TASKS = {
    "icbhi_copd": {"mel_root": "data/mel_cache/opera_icbhi_disease", "n_classes": 2},
    "svd_pathology": {"mel_root": "data/mel_cache/svd_full", "n_classes": 2},
    "coswara_covid_cough": {"mel_root": "data/mel_cache/coswara_covid_cough", "n_classes": 2},
    "b2ai_voice_pathology": {"mel_root": "data/mel_cache/b2ai_voice_pathology", "n_classes": 2},
}


def compute_d_eff(features):
    """Compute effective rank (entropy of normalized eigenvalues)."""
    features = features - features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals / (eigvals.sum() + 1e-12)
    eigvals = eigvals[eigvals > 1e-12]
    d_eff = np.exp(-np.sum(eigvals * np.log(eigvals)))
    return float(d_eff)


def train_vq_and_measure(features, K=512, steps=5000, batch_size=256):
    """Train VQ on features, return utilization and perplexity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = features.shape[1]
    vq = VectorQuantizer(codebook_size=K, D=D, l2_normalize=False,
                         use_ema=True, ema_decay=0.99,
                         restart_threshold=1, restart_every=1)
    vq = vq.to(device)

    features_t = torch.from_numpy(features).float().to(device)
    N = features_t.shape[0]

    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,))
        batch = features_t[idx].unsqueeze(1)  # (B, 1, D)
        out = vq(batch)

    # Final measurement on full dataset (sampled)
    sample_size = min(N, 10000)
    idx = torch.randperm(N)[:sample_size]
    batch = features_t[idx].unsqueeze(1)
    with torch.no_grad():
        out = vq(batch)
        ids = out["ids"].squeeze(1).cpu().numpy()

    unique_codes = len(set(ids.flatten().tolist()))
    utilization = unique_codes / K

    counts = Counter(ids.flatten().tolist())
    total = sum(counts.values())
    probs = np.array([counts.get(i, 0) / total for i in range(K)])
    probs = probs[probs > 0]
    perplexity = float(np.exp(-np.sum(probs * np.log(probs))))

    vq_state = {k: v.cpu() for k, v in vq.state_dict().items()}
    return utilization, perplexity, vq_state, vq


def vq_linear_auroc_seq(seq_train, labels_train, seq_test, labels_test,
                        vq, K, device):
    """VQ-linear AUROC from sequence features (list of (T_i, D) arrays)."""
    def get_histograms(sequences, vq_model):
        histograms = np.zeros((len(sequences), K), dtype=np.float32)
        for i, seq in enumerate(sequences):
            seq_t = torch.from_numpy(seq).float().to(device).unsqueeze(0)  # (1, T, D)
            with torch.no_grad():
                out = vq_model(seq_t)
                ids = out["ids"].cpu().numpy().flatten()
            for tok in ids:
                histograms[i, int(tok)] += 1
            histograms[i] /= (len(ids) + 1e-8)
        return histograms

    hist_train = get_histograms(seq_train, vq)
    hist_test = get_histograms(seq_test, vq)

    scaler = StandardScaler()
    hist_train = scaler.fit_transform(hist_train)
    hist_test = scaler.transform(hist_test)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(hist_train, labels_train)

    if len(set(labels_test)) == 2:
        probs = clf.predict_proba(hist_test)[:, 1]
        auc = roc_auc_score(labels_test, probs)
    else:
        probs = clf.predict_proba(hist_test)
        auc = roc_auc_score(labels_test, probs, multi_class="ovr", average="macro")
    return float(auc)


def vq_linear_auroc(features_train, labels_train, features_test, labels_test,
                    vq, K, device):
    """Compute AUROC using VQ token histogram features + logistic regression."""
    def get_histogram(feats, vq_model):
        feats_t = torch.from_numpy(feats).float().to(device).unsqueeze(1)
        with torch.no_grad():
            out = vq_model(feats_t)
            ids = out["ids"].cpu().numpy().reshape(len(feats), -1)
        histograms = np.zeros((len(feats), K), dtype=np.float32)
        for i, row in enumerate(ids):
            row = np.atleast_1d(row)
            for tok in row:
                histograms[i, int(tok)] += 1
            histograms[i] /= (len(row) + 1e-8)
        return histograms

    hist_train = get_histogram(features_train, vq)
    hist_test = get_histogram(features_test, vq)

    scaler = StandardScaler()
    hist_train = scaler.fit_transform(hist_train)
    hist_test = scaler.transform(hist_test)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(hist_train, labels_train)

    if len(set(labels_test)) == 2:
        probs = clf.predict_proba(hist_test)[:, 1]
        auc = roc_auc_score(labels_test, probs)
    else:
        probs = clf.predict_proba(hist_test)
        auc = roc_auc_score(labels_test, probs, multi_class="ovr", average="macro")
    return float(auc)


def extract_htsat_features(encoder, mel_dir, device, return_sequences=False):
    """Extract features using HTS-AT based encoder.

    If return_sequences=True, returns list of (T, D) arrays (for VQ).
    Otherwise returns (N, D) mean-pooled features (for d_eff).
    """
    meta = json.loads((mel_dir / "metadata.json").read_text())
    samples = meta.get("samples", [])

    features, labels, splits = [], [], []
    for s in samples:
        if "label" not in s:
            continue
        mel_path = mel_dir / s["path"]
        if not mel_path.exists():
            continue
        mel = torch.load(str(mel_path), map_location="cpu")
        mel = mel.unsqueeze(0).to(device)
        with torch.no_grad():
            z = encoder(mel)  # (1, T, D)
            if return_sequences:
                feat = z.squeeze(0).cpu().numpy()  # (T, D)
            else:
                feat = z.squeeze(0).mean(dim=0).cpu().numpy()  # (D,)
        features.append(feat)
        labels.append(int(s["label"]))
        splits.append(s.get("split", "train"))

    if not return_sequences:
        return np.stack(features), np.array(labels), splits
    return features, np.array(labels), splits


def extract_hf_features(model_name, mel_dir, device):
    """Extract features using HuggingFace audio models (wav2vec2, HuBERT, etc.)."""
    from transformers import AutoModel, AutoFeatureExtractor
    import librosa

    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device).eval()

    meta = json.loads((mel_dir / "metadata.json").read_text())
    samples = meta.get("samples", [])

    # We need raw wav — check wav_cache or reconstruct from mel
    # For HF models we need waveform input, not mel
    # Check if wav_cache exists
    wav_dir = ROOT / "data" / "wav_cache" / mel_dir.name
    if not wav_dir.exists():
        # Try to use mel directly — won't work for wav2vec
        # Need wav files — check original_path in metadata
        pass

    features, labels, splits = [], [], []
    for s in samples:
        if "label" not in s:
            continue
        # Try wav_cache
        wav_path = wav_dir / s["path"].replace(".pt", ".npy")
        if wav_path.exists():
            wav = np.load(str(wav_path)).astype(np.float32)
        elif "original_path" in s and Path(s["original_path"]).exists():
            wav, _ = librosa.load(s["original_path"], sr=SR)
        else:
            continue

        # Normalize and pad/crop
        wav = (wav - wav.mean()) / (wav.std() + 1e-8)
        if len(wav) > WAV_LEN:
            wav = wav[:WAV_LEN]
        else:
            wav = np.pad(wav, (0, max(0, WAV_LEN - len(wav))))

        inputs = processor(wav, sampling_rate=SR, return_tensors="pt", padding=True)
        input_values = inputs.get("input_values", inputs.get("input_features"))
        input_values = input_values.to(device)

        with torch.no_grad():
            out = model(input_values)
            hidden = out.last_hidden_state  # (1, T, D)
            feat = hidden.squeeze(0).mean(dim=0).cpu().numpy()

        features.append(feat)
        labels.append(int(s["label"]))
        splits.append(s.get("split", "train"))

    if not features:
        return None, None, None
    return np.stack(features), np.array(labels), splits


def load_opera_ct(device):
    """Load OPERA-CT encoder."""
    ckpt_path = ROOT / "opera_src/cks/model/encoder-operaCT.ckpt"
    encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    # OPERA stores as "module.xxx" or "encoder.xxx"
    cleaned = {}
    for k, v in state.items():
        k2 = k.replace("module.", "").replace("encoder.", "")
        cleaned[k2] = v
    encoder.load_state_dict(cleaned, strict=False)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_vast_encoder(ckpt_path, device):
    """Load our VAST encoder (HTS-AT + CSAF + LeJEPA)."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = {k.replace("encoder.", "", 1): v
             for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
    encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/rq2_baselines/results.json")
    parser.add_argument("--vq-K", type=int, default=512)
    parser.add_argument("--vq-steps", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(Path(args.output).parent, exist_ok=True)

    # Define encoder configs
    ENCODERS = {
        "OPERA-CT": {
            "type": "htsat",
            "loader": lambda: load_opera_ct(device),
        },
        "JEPA-only (no SIGReg)": {
            "type": "htsat",
            "loader": lambda: load_vast_encoder(
                ROOT / "checkpoints/htsat_jepa_only_d768/htsat_lejepa_best.pt", device),
        },
        "VAST (Ours)": {
            "type": "htsat",
            "loader": lambda: load_vast_encoder(
                ROOT / "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt", device),
        },
    }

    # Try HuggingFace models
    hf_models = {
        "HuBERT-base": "facebook/hubert-base-ls960",
        "Wav2vec2-base": "facebook/wav2vec2-base",
    }
    try:
        from transformers import AutoModel
        for name, model_id in hf_models.items():
            ENCODERS[name] = {"type": "hf", "model_id": model_id}
    except ImportError:
        print("WARNING: transformers not available, skipping HF models")

    all_results = {}

    for enc_name, enc_cfg in ENCODERS.items():
        print(f"\n{'='*70}")
        print(f"  Encoder: {enc_name}")
        print(f"{'='*70}")

        # Extract features from all tasks (both mean-pooled and sequences)
        all_mean_features = []
        all_seq_features = []
        task_data = {}

        for task_key, task_cfg in EVAL_TASKS.items():
            mel_dir = ROOT / task_cfg["mel_root"]
            if not (mel_dir / "metadata.json").exists():
                print(f"  Skipping {task_key} - no metadata")
                continue

            print(f"  Extracting {task_key}...")
            if enc_cfg["type"] == "htsat":
                if "encoder" not in enc_cfg:
                    enc_cfg["encoder"] = enc_cfg["loader"]()
                # Mean-pooled for d_eff
                feats, labels, splits = extract_htsat_features(
                    enc_cfg["encoder"], mel_dir, device, return_sequences=False)
                # Sequences for VQ
                seq_feats, _, _ = extract_htsat_features(
                    enc_cfg["encoder"], mel_dir, device, return_sequences=True)
            elif enc_cfg["type"] == "hf":
                feats, labels, splits = extract_hf_features(
                    enc_cfg["model_id"], mel_dir, device)
                seq_feats = None
                if feats is None:
                    print(f"    No wav data available, skipping")
                    continue
            else:
                continue

            print(f"    {len(feats)} samples, D={feats.shape[1]}")
            all_mean_features.append(feats)
            if seq_feats is not None:
                all_seq_features.extend(seq_feats)

            train_mask = np.array([s in ("train", "val") for s in splits])
            test_mask = np.array([s == "test" for s in splits])
            task_data[task_key] = {
                "features": feats, "labels": labels,
                "seq_features": seq_feats,
                "train_mask": train_mask, "test_mask": test_mask,
            }

        if not all_mean_features:
            print("  No features extracted, skipping encoder")
            continue

        # Compute d_eff on mean-pooled features
        combined_mean = np.concatenate(all_mean_features, axis=0)
        d_eff = compute_d_eff(combined_mean)
        print(f"\n  d_eff = {d_eff:.1f} (D={combined_mean.shape[1]})")

        # Train VQ on sequence features (all frames concatenated)
        if all_seq_features:
            combined_seq = np.concatenate(all_seq_features, axis=0)
        else:
            combined_seq = combined_mean
        print(f"  Training VQ on {len(combined_seq)} frames (K={args.vq_K}, steps={args.vq_steps})...")
        util, perp, vq_state, vq = train_vq_and_measure(
            combined_seq, K=args.vq_K, steps=args.vq_steps)
        print(f"  Codebook utilization: {util:.3f}")
        print(f"  Perplexity: {perp:.1f}")

        # VQ-linear AUROC: use sequence features → histogram → LR
        task_aurocs = {}
        for task_key, td in task_data.items():
            if td["train_mask"].sum() == 0 or td["test_mask"].sum() == 0:
                continue
            seq = td.get("seq_features")
            if seq is None:
                seq = [td["features"][i:i+1] for i in range(len(td["features"]))]
            train_seq = [seq[i] for i in range(len(seq)) if td["train_mask"][i]]
            train_labels = td["labels"][td["train_mask"]]
            test_seq = [seq[i] for i in range(len(seq)) if td["test_mask"][i]]
            test_labels = td["labels"][td["test_mask"]]
            try:
                auc = vq_linear_auroc_seq(
                    train_seq, train_labels, test_seq, test_labels,
                    vq, args.vq_K, device)
                task_aurocs[task_key] = auc
                print(f"  VQ-linear {task_key}: AUROC={auc:.4f}")
            except Exception as e:
                print(f"  VQ-linear {task_key}: FAILED ({e})")

        avg_auroc = np.mean(list(task_aurocs.values())) if task_aurocs else 0.0

        all_results[enc_name] = {
            "d_eff": d_eff,
            "codebook_utilization": util,
            "perplexity": perp,
            "vq_linear_auroc": task_aurocs,
            "vq_linear_auroc_avg": avg_auroc,
            "D": int(combined.shape[1]),
            "N_samples": int(combined.shape[0]),
        }

        # Cleanup
        if "encoder" in enc_cfg:
            del enc_cfg["encoder"]
        del vq, combined
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<30} {'d_eff':>8} {'Util':>8} {'Perp':>8} {'Avg AUROC':>10}")
    print(f"{'='*80}")
    for name, r in all_results.items():
        print(f"{name:<30} {r['d_eff']:>8.1f} {r['codebook_utilization']:>8.3f} "
              f"{r['perplexity']:>8.1f} {r['vq_linear_auroc_avg']:>10.4f}")


if __name__ == "__main__":
    main()
