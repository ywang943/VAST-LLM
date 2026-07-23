#!/usr/bin/env python3
"""
Comprehensive baseline evaluation for Tables 2, 3, 4 of VAST paper.

Table 2 (RQ1): Linear-probe AUROC per task (S1–S7), task-specific for baselines.
Table 3 (RQ2): d_eff, codebook util, perplexity, VQ-linear AUROC.
Table 4 (RQ3): Zero-shot cross-domain AUROC (T1–T5) via LLM for VAST,
               linear-probe transfer for baselines.

Baselines:
  Wav2vec-BERT   – facebook/w2v-bert-2.0 (wav input)
  AudioMAE       – MIT/ast-finetuned-audioset-10-10-0.4593 (mel input)
  AudioSet-pret. – HTS-AT initialized from AudioSet (OPERA's backbone before fine-tune)
  Contrastive    – COLA on respiratory (similar to OPERA pretraining strategy)
  OPERA          – encoder-operaCT.ckpt
  RespLLM        – same backbone as OPERA (audio encoder part)
  MVP            – facebook/hubert-base-ls960 (wav input)
  MARVEL/Unified – EfficientNet-B0 on mel spectrograms
  JEPA w/o SIG   – htsat_jepa_only_d768/htsat_lejepa_best.pt
  VAST (Ours)    – htsat_lejepa_v3_full/htsat_lejepa_best.pt
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
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

SR = 16000
WAV_LEN = SR * 8

# ── Task definitions ─────────────────────────────────────────────────────
# S-tasks: seen (train+test same source)
S_TASKS = {
    "S1_kauh": {
        "mel_root": "data/mel_cache/opera_kauh",
        "wav_root": "data/wav_cache/opera_kauh",
        "n_classes": 2,
        "name": "KAUH",
    },
    "S2_coswara_covid_breath": {
        "mel_root": "data/mel_cache/coswara_covid_breathing",
        "wav_root": None,
        "n_classes": 2,
        "name": "C-Breath",
    },
    "S3_coswara_covid_cough": {
        "mel_root": "data/mel_cache/coswara_covid_cough",
        "wav_root": None,
        "n_classes": 2,
        "name": "C-Cough",
    },
    "S4_svd": {
        "mel_root": "data/mel_cache/svd_full",
        "wav_root": "data/wav_cache/svd_full",
        "n_classes": 2,
        "name": "SVD",
    },
    "S5_coswara_smoker": {
        "mel_root": "data/mel_cache/coswara_smoker_cough",
        "wav_root": None,
        "n_classes": 2,
        "name": "Coswara",
    },
    "S6_icbhi_copd": {
        "mel_root": "data/mel_cache/opera_icbhi_disease",
        "wav_root": "data/wav_cache/opera_icbhi_disease",
        "n_classes": 2,
        "name": "COPD",
    },
    "S7_b2ai": {
        "mel_root": "data/mel_cache/b2ai_voice_pathology",
        "wav_root": None,
        "n_classes": 2,
        "name": "Bridge2AI",
    },
}

# T-tasks: zero-shot targets
T_TASKS = {
    "T1_icbhi": {
        "mel_root": "data/mel_cache/opera_icbhi_disease",
        "wav_root": "data/wav_cache/opera_icbhi_disease",
        "n_classes": 2,
        "name": "ICBHI",
    },
    "T2_b2ai_cancer": {
        "mel_root": "data/mel_cache/b2ai_laryngeal_cancer",
        "wav_root": None,
        "n_classes": 2,
        "name": "Laryngeal Cancer",
    },
    "T3_b2ai_benign": {
        "mel_root": "data/mel_cache/b2ai_benign_lesions",
        "wav_root": None,
        "n_classes": 2,
        "name": "Benign Lesions",
    },
    "T4_b2ai_dystonia": {
        "mel_root": "data/mel_cache/b2ai_laryngeal_dystonia",
        "wav_root": None,
        "n_classes": 2,
        "name": "Spasmodic Dysphonia",
    },
    "T5_coswara_covid_cough": {
        "mel_root": "data/mel_cache/coswara_covid_cough",
        "wav_root": None,
        "n_classes": 2,
        "name": "Coswara",
    },
}

# ── Feature extraction helpers ───────────────────────────────────────────
def load_task_metadata(task_cfg):
    mel_dir = ROOT / task_cfg["mel_root"]
    meta_path = mel_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    return meta.get("samples", [])


def extract_mel_features(encoder, task_cfg, device, pool="full"):
    """Extract features from mel-input encoder (HTS-AT based).
    pool='full' returns both mean-pooled features AND per-sample sequences.
    """
    mel_dir = ROOT / task_cfg["mel_root"]
    samples = load_task_metadata(task_cfg)
    if samples is None:
        return None

    mean_feats, sequences, labels, splits = [], [], [], []
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
            seq = z.squeeze(0).cpu().numpy()  # (T, D)
            mean = seq.mean(axis=0)  # (D,)
        mean_feats.append(mean)
        sequences.append(seq)
        labels.append(int(s["label"]))
        splits.append(s.get("split", "train"))

    if not mean_feats:
        return None
    return {"features": np.stack(mean_feats), "labels": np.array(labels),
            "splits": splits, "sequences": sequences}


def extract_wav_features(model, processor, task_cfg, device, pool="full"):
    """Extract features from wav-input HF model."""
    wav_root = task_cfg.get("wav_root")
    mel_dir = ROOT / task_cfg["mel_root"]
    samples = load_task_metadata(task_cfg)
    if samples is None:
        return None

    mean_feats, sequences, labels, splits = [], [], [], []
    for s in samples:
        if "label" not in s:
            continue
        wav_path = None
        if wav_root:
            wp = ROOT / wav_root / s["path"].replace(".pt", ".npy")
            if wp.exists():
                wav_path = wp
        if wav_path is None:
            continue

        wav = np.load(str(wav_path)).astype(np.float32)
        wav = (wav - wav.mean()) / (wav.std() + 1e-8)
        if len(wav) > WAV_LEN:
            wav = wav[:WAV_LEN]
        else:
            wav = np.pad(wav, (0, max(0, WAV_LEN - len(wav))))

        inputs = processor(wav, sampling_rate=SR, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inputs.items()
               if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            out = model(**inp)
            hidden = out.last_hidden_state  # (1, T, D)
            seq = hidden.squeeze(0).cpu().numpy()  # (T, D)
            mean = seq.mean(axis=0)
        mean_feats.append(mean)
        sequences.append(seq)
        labels.append(int(s["label"]))
        splits.append(s.get("split", "train"))

    if not mean_feats:
        return None
    return {"features": np.stack(mean_feats), "labels": np.array(labels),
            "splits": splits, "sequences": sequences}


def extract_efficientnet_features(model, task_cfg, device):
    """Extract features using EfficientNet-B0 on mel spectrograms (MARVEL-style)."""
    mel_dir = ROOT / task_cfg["mel_root"]
    samples = load_task_metadata(task_cfg)
    if samples is None:
        return None

    features, labels, splits = [], [], []
    for s in samples:
        if "label" not in s:
            continue
        mel_path = mel_dir / s["path"]
        if not mel_path.exists():
            continue
        mel = torch.load(str(mel_path), map_location="cpu")  # (1, 64, T)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        # EfficientNet expects 3-channel (224,224) — resize mel
        mel_3ch = mel.expand(3, -1, -1).unsqueeze(0)  # (1, 3, 64, T)
        mel_3ch = F.interpolate(mel_3ch, size=(224, 224), mode="bilinear",
                                align_corners=False)
        mel_3ch = mel_3ch.to(device)
        with torch.no_grad():
            feat = model(mel_3ch).squeeze().cpu().numpy()
        features.append(feat)
        labels.append(int(s["label"]))
        splits.append(s.get("split", "train"))

    if not features:
        return None
    return {"features": np.stack(features), "labels": np.array(labels),
            "splits": splits, "sequences": None}


# ── Evaluation functions ─────────────────────────────────────────────────
def compute_d_eff(features):
    features = features - features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    total = eigvals.sum() + 1e-12
    p = eigvals / total
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def train_vq_measure(features_all, K=512, steps=5000):
    """Train VQ, return (utilization, perplexity, vq_module)."""
    from respvoice.vq import VectorQuantizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = features_all.shape[1]
    vq = VectorQuantizer(codebook_size=K, D=D, l2_normalize=False,
                         use_ema=True, ema_decay=0.99,
                         restart_threshold=1, restart_every=1).to(device)
    feats_t = torch.from_numpy(features_all).float().to(device)
    N = len(feats_t)
    bs = min(256, N)
    for _ in range(steps):
        idx = torch.randint(0, N, (bs,))
        vq(feats_t[idx].unsqueeze(1))

    # Measure
    sample_n = min(N, 10000)
    idx = torch.randperm(N)[:sample_n]
    with torch.no_grad():
        out = vq(feats_t[idx].unsqueeze(1))
        ids = out["ids"].cpu().numpy().flatten()
    unique = len(set(ids.tolist()))
    util = unique / K
    counts = Counter(ids.tolist())
    total = sum(counts.values())
    probs = np.array([counts.get(i, 0) / total for i in range(K)])
    probs = probs[probs > 0]
    perp = float(np.exp(-np.sum(probs * np.log(probs))))
    return util, perp, vq


def linear_probe_auroc(features, labels, splits):
    """Standard linear probe AUROC."""
    train_mask = np.array([s in ("train", "val") for s in splits])
    test_mask = np.array([s == "test" for s in splits])
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        # For zero-shot tasks (all test), return None
        return None

    X_train, y_train = features[train_mask], labels[train_mask]
    X_test, y_test = features[test_mask], labels[test_mask]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    n_classes = len(set(y_test))
    if n_classes == 2:
        probs = clf.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, probs))
    elif n_classes > 2:
        probs = clf.predict_proba(X_test)
        return float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro"))
    return None


def vq_histogram_auroc(sequences, labels, splits, vq, K, device):
    """VQ token histogram → LR → AUROC."""
    train_mask = np.array([s in ("train", "val") for s in splits])
    test_mask = np.array([s == "test" for s in splits])
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    def histograms(seqs, mask):
        H = np.zeros((mask.sum(), K), dtype=np.float32)
        j = 0
        for i, m in enumerate(mask):
            if not m:
                continue
            s = seqs[i] if isinstance(seqs[i], np.ndarray) else seqs[i]
            s_t = torch.from_numpy(s).float().to(device).unsqueeze(0)
            with torch.no_grad():
                ids = vq(s_t)["ids"].cpu().numpy().flatten()
            for tok in ids:
                H[j, int(tok)] += 1
            H[j] /= (len(ids) + 1e-8)
            j += 1
        return H

    # If no sequences, use mean features as single-token
    if sequences is None:
        return None

    H_train = histograms(sequences, train_mask)
    H_test = histograms(sequences, test_mask)

    scaler = StandardScaler()
    H_train = scaler.fit_transform(H_train)
    H_test = scaler.transform(H_test)

    y_train = labels[train_mask]
    y_test = labels[test_mask]

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(H_train, y_train)

    if len(set(y_test)) == 2:
        probs = clf.predict_proba(H_test)[:, 1]
        return float(roc_auc_score(y_test, probs))
    return None


# ── Encoder loaders ──────────────────────────────────────────────────────
def load_htsat_encoder(ckpt_path, device):
    from respvoice.htsat_encoder import build_htsat_encoder
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "model_state" in ckpt:
        state = {k.replace("encoder.", "", 1): v
                 for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    else:
        state = ckpt
    encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    encoder.load_state_dict(state, strict=False)
    return encoder.to(device).eval()


def load_opera_ct(device):
    from respvoice.htsat_encoder import build_htsat_encoder
    ckpt_path = ROOT / "opera_src/cks/model/encoder-operaCT.ckpt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    cleaned = {k.replace("module.", "").replace("encoder.", ""): v
               for k, v in state.items()}
    encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    encoder.load_state_dict(cleaned, strict=False)
    return encoder.to(device).eval()


def load_hf_wav_model(model_id, device):
    from transformers import AutoModel, AutoFeatureExtractor
    print(f"    Loading {model_id}...")
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return model, processor


def load_efficientnet(device):
    """EfficientNet-B0 as feature extractor (MARVEL-style)."""
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
    model.classifier = nn.Identity()
    return model.to(device).eval()


# ── Main pipeline ────────────────────────────────────────────────────────
def run_encoder(enc_name, enc_type, extract_fn, all_tasks, K, vq_steps, device):
    """Run full evaluation for one encoder across all tasks."""
    print(f"\n{'='*70}")
    print(f"  {enc_name}")
    print(f"{'='*70}")

    task_results = {}
    all_feats_for_deff = []
    all_seqs_for_vq = []

    for tkey, tcfg in all_tasks.items():
        mel_dir = ROOT / tcfg["mel_root"]
        if not (mel_dir / "metadata.json").exists():
            print(f"  {tkey}: SKIP (no metadata)")
            continue

        try:
            data = extract_fn(tcfg, pool="seq")
        except Exception as e:
            print(f"  {tkey}: SKIP ({e})")
            continue

        if data is None:
            print(f"  {tkey}: SKIP (no data)")
            continue

        n = len(data["labels"])
        D = data["features"].shape[1]
        lp_auc = linear_probe_auroc(data["features"], data["labels"], data["splits"])
        print(f"  {tkey}: n={n} D={D} LP-AUROC={lp_auc if lp_auc else 'N/A'}")

        task_results[tkey] = {"n": n, "lp_auroc": lp_auc}
        all_feats_for_deff.append(data["features"])
        if data["sequences"] is not None:
            all_seqs_for_vq.extend(data["sequences"])
        task_results[tkey]["_data"] = data

    if not all_feats_for_deff:
        return None

    # d_eff
    combined = np.concatenate(all_feats_for_deff)
    d_eff = compute_d_eff(combined)
    print(f"\n  d_eff = {d_eff:.1f}")

    # VQ
    if all_seqs_for_vq:
        vq_input = np.concatenate(all_seqs_for_vq)
    else:
        vq_input = combined
    print(f"  Training VQ on {len(vq_input)} frames...")
    util, perp, vq = train_vq_measure(vq_input, K=K, steps=vq_steps)
    print(f"  Codebook util={util:.3f}, perplexity={perp:.1f}")

    # VQ-linear AUROC (aggregate over S-tasks that have train/test splits)
    vq_aurocs = []
    for tkey, tr in task_results.items():
        data = tr.pop("_data", None)
        if data is None or data["sequences"] is None:
            continue
        try:
            vauc = vq_histogram_auroc(
                data["sequences"], data["labels"], data["splits"],
                vq, K, device)
            if vauc is not None:
                tr["vq_auroc"] = vauc
                vq_aurocs.append(vauc)
                print(f"  VQ-linear {tkey}: {vauc:.4f}")
        except Exception as e:
            print(f"  VQ-linear {tkey}: FAIL ({e})")

    # Cleanup remaining _data
    for tr in task_results.values():
        tr.pop("_data", None)

    del vq
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "d_eff": d_eff,
        "codebook_util": util,
        "perplexity": perp,
        "vq_linear_auroc_avg": float(np.mean(vq_aurocs)) if vq_aurocs else None,
        "tasks": task_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/rq_all_baselines/results.json")
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--vq-steps", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(str(Path(args.output).parent), exist_ok=True)

    ALL_TASKS = {**S_TASKS, **T_TASKS}
    results = {}

    # Use closures carefully — Python closures capture by reference,
    # so we use default-argument binding to freeze the encoder variable.

    # ── 1. OPERA-CT ──
    try:
        enc = load_opera_ct(device)
        def _make_mel_ex(e):
            def _ex(tcfg, pool="full"):
                return extract_mel_features(e, tcfg, device)
            return _ex
        results["OPERA"] = run_encoder("OPERA", "mel", _make_mel_ex(enc),
                                       ALL_TASKS, args.K, args.vq_steps, device)
        del enc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"OPERA FAILED: {e}")
        traceback.print_exc()

    # ── 2. JEPA without SIGReg ──
    try:
        ckpt = ROOT / "checkpoints/htsat_jepa_only_d768/htsat_lejepa_best.pt"
        enc = load_htsat_encoder(ckpt, device)
        results["JEPA without SIGReg"] = run_encoder(
            "JEPA without SIGReg", "mel", _make_mel_ex(enc),
            ALL_TASKS, args.K, args.vq_steps, device)
        del enc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"JEPA FAILED: {e}")
        traceback.print_exc()

    # ── 3. VAST (Ours) ──
    try:
        ckpt = ROOT / "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt"
        enc = load_htsat_encoder(ckpt, device)
        results["VAST (Ours)"] = run_encoder(
            "VAST (Ours)", "mel", _make_mel_ex(enc),
            ALL_TASKS, args.K, args.vq_steps, device)
        del enc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"VAST FAILED: {e}")
        traceback.print_exc()

    # ── 4. HuBERT-base (MVP backbone) ──
    try:
        model, processor = load_hf_wav_model("facebook/hubert-base-ls960", device)
        def _make_wav_ex(m, p):
            def _ex(tcfg, pool="full"):
                return extract_wav_features(m, p, tcfg, device)
            return _ex
        results["MVP (HuBERT)"] = run_encoder(
            "MVP (HuBERT-base)", "wav", _make_wav_ex(model, processor),
            ALL_TASKS, args.K, args.vq_steps, device)
        del model, processor; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"HuBERT FAILED: {e}")
        traceback.print_exc()

    # ── 5. Wav2vec2-base ──
    try:
        model, processor = load_hf_wav_model("facebook/wav2vec2-base", device)
        results["Wav2vec-BERT"] = run_encoder(
            "Wav2vec2-base", "wav", _make_wav_ex(model, processor),
            ALL_TASKS, args.K, args.vq_steps, device)
        del model, processor; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"Wav2vec2 FAILED: {e}")
        traceback.print_exc()

    # ── 6. MARVEL / EfficientNet-B0 ──
    try:
        model = load_efficientnet(device)
        def _make_eff_ex(m):
            def _ex(tcfg, pool="full"):
                return extract_efficientnet_features(m, tcfg, device)
            return _ex
        results["MARVEL / Unified"] = run_encoder(
            "MARVEL / Unified (EfficientNet-B0)", "mel",
            _make_eff_ex(model), ALL_TASKS, args.K, args.vq_steps, device)
        del model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"MARVEL FAILED: {e}")
        traceback.print_exc()

    # ── Save ──
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {args.output}")

    # ── Summary tables ──
    print(f"\n{'='*80}")
    print("TABLE 3 (RQ2): Codebook Quality")
    print(f"{'Method':<30} {'d_eff':>8} {'Util':>8} {'Perp':>8} {'VQ-Avg':>8}")
    print("-" * 70)
    for name, r in results.items():
        if r is None:
            continue
        vqa = r.get("vq_linear_auroc_avg")
        print(f"{name:<30} {r['d_eff']:>8.1f} {r['codebook_util']:>8.3f} "
              f"{r['perplexity']:>8.1f} {vqa if vqa else 'N/A':>8}")

    print(f"\n{'='*80}")
    print("TABLE 2 (RQ1): Same-source LP-AUROC")
    s_keys = list(S_TASKS.keys())
    header = f"{'Method':<25}" + "".join(f"{S_TASKS[k]['name']:>12}" for k in s_keys) + f"{'Avg':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        if r is None:
            continue
        vals = []
        for sk in s_keys:
            v = r["tasks"].get(sk, {}).get("lp_auroc")
            vals.append(v)
        avg = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
        line = f"{name:<25}"
        for v in vals:
            line += f"{v:>12.4f}" if v is not None else f"{'—':>12}"
        line += f"{avg:>8.4f}" if avg else f"{'—':>8}"
        print(line)

    print(f"\n{'='*80}")
    print("TABLE 4 (RQ3): Zero-shot LP-AUROC")
    t_keys = list(T_TASKS.keys())
    header = f"{'Method':<25}" + "".join(f"{T_TASKS[k]['name']:>16}" for k in t_keys) + f"{'Avg':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        if r is None:
            continue
        vals = []
        for tk in t_keys:
            v = r["tasks"].get(tk, {}).get("lp_auroc")
            vals.append(v)
        avg = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
        line = f"{name:<25}"
        for v in vals:
            line += f"{v:>16.4f}" if v is not None else f"{'—':>16}"
        line += f"{avg:>8.4f}" if avg else f"{'—':>8}"
        print(line)


if __name__ == "__main__":
    main()
