#!/usr/bin/env python3
"""RQ3 external audio+text baselines: CLAP and SpeechTokenizer.

CLAP: concatenate CLAP audio embedding and CLAP text embedding of DMS/task text,
then train the same source->target logistic heads as RQ3 audio baselines.

SpeechTokenizer: extract RVQ code histograms from waveform, concatenate a simple
TF-IDF DMS representation, then train the same source->target heads. This is a
fast baseline for a speech-tokenizer+text pipeline without training another LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))
sys.path.insert(0, str(ROOT / "external" / "SpeechTokenizer"))
sys.path.insert(0, str(ROOT / "external" / "CLAP" / "src"))

from scripts.run_rq3_audio_baselines import (  # noqa: E402
    EVAL_COLUMNS as LEGACY_EVAL_COLUMNS,
    SOURCE_FOR_TARGET as LEGACY_SOURCE_FOR_TARGET,
    TARGET_EXCLUDE_DIAGNOSES,
)
from scripts.run_rq3_llm import TASKS  # noqa: E402


SR = 16000
WAV_LEN = SR * 8
CLAP_SR = 48000
_CLAP_PROCESSOR = None
_CLAP_MODEL = None

NEW_TABLE_EVAL_COLUMNS = {
    "T1_LaryngealCancer": "b2ai_laryngeal_cancer",
    "T2_BenignVocal": "b2ai_benign_lesions",
    "T3_LaryngealDystonia": "b2ai_laryngeal_dystonia",
    "T4_CovidBreath": "coswara_covid_breathing",
    "T5_SmokerBreath": "coswara_smoker_breathing",
    "T6_UKCovidCough": "uk_covid_cough",
    "T7_SVD_VP": "svd_pathology",
}

NEW_TABLE_SOURCE_FOR_TARGET = {
    "b2ai_laryngeal_cancer": "b2ai_voice_pathology",
    "b2ai_benign_lesions": "b2ai_voice_pathology",
    "b2ai_laryngeal_dystonia": "b2ai_voice_pathology",
    "coswara_covid_breathing": "coswara_covid_cough",
    "coswara_smoker_breathing": "coswara_smoker_cough",
    "uk_covid_cough": "coswara_covid_cough",
    "svd_pathology": "svd_pathology",
}


class RQ3TaskDataset(Dataset):
    def __init__(self, task_key: str):
        self.task_key = task_key
        cfg = TASKS[task_key]
        self.cfg = cfg
        self.mel_root = ROOT / cfg["mel_root"]
        meta = json.loads((self.mel_root / "metadata.json").read_text())
        self.samples = [s for s in meta.get("samples", []) if "label" in s]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "idx": idx,
            "mel_path": s["path"],
            "label": int(s["label"]),
            "split": s.get("split", "train"),
            "dms_text": s.get("dms_text", ""),
            "pid": str(s.get("pid", s.get("participant_id", ""))),
            "diagnosis": str(s.get("diagnosis", "")),
            "sample": s,
        }


def collate_meta(batch):
    return batch


class WaveLoader:
    def __init__(self):
        self.inv = None
        self.gl = None

    def _ensure_inverse(self, device):
        if self.inv is None:
            self.inv = torchaudio.transforms.InverseMelScale(
                n_stft=513, n_mels=64, sample_rate=SR, f_min=50, f_max=8000
            ).to(device)
            self.gl = torchaudio.transforms.GriffinLim(
                n_fft=1024, win_length=1024, hop_length=512, n_iter=2, power=2.0
            ).to(device)

    def _candidate_wav_path(self, task_key, mel_path, sample):
        cfg = TASKS[task_key]
        if cfg.get("wav_root"):
            return ROOT / cfg["wav_root"] / mel_path.replace(".pt", ".npy")
        direct_roots = {
            "icbhi_copd": "data/wav_cache/opera_icbhi_disease",
            "copd_severity": "data/wav_cache/opera_copd",
            "svd_pathology": "data/wav_cache/svd_full",
            "svd_pathology_target": "data/wav_cache/svd_full",
            "kauh_obstructive": "data/wav_cache/opera_kauh",
        }
        if task_key == "svd_pathology_target":
            src = sample.get("source_path", mel_path)
            return ROOT / direct_roots[task_key] / src.replace(".pt", ".npy")
        if task_key in direct_roots:
            return ROOT / direct_roots[task_key] / mel_path.replace(".pt", ".npy")
        return None

    def load(self, task_key, mel_root, meta, device):
        wav_path = self._candidate_wav_path(task_key, meta["mel_path"], meta["sample"])
        if wav_path is not None and wav_path.exists():
            wav = np.load(str(wav_path)).astype(np.float32)
            source = "wav_cache"
            wav = torch.from_numpy(wav)
        else:
            self._ensure_inverse(device)
            mel = torch.load(mel_root / meta["mel_path"], map_location="cpu", weights_only=True)
            mel = mel.squeeze(0).to(device)
            linear = torch.exp(mel).clamp_min(1e-6)
            with torch.no_grad():
                wav = self.gl(self.inv(linear)).detach().cpu()
            source = "fallback_mel_griffinlim"
        wav = wav.float().flatten()
        if wav.numel() >= WAV_LEN:
            wav = wav[:WAV_LEN]
        else:
            wav = F.pad(wav, (0, WAV_LEN - wav.numel()))
        wav = (wav - wav.mean()) / (wav.std() + 1e-8)
        return wav, source

    def load_batch(self, task_key, mel_root, metas, device):
        wavs = [None] * len(metas)
        sources = [None] * len(metas)
        fallback = []
        fallback_idx = []
        for i, meta in enumerate(metas):
            wav_path = self._candidate_wav_path(task_key, meta["mel_path"], meta["sample"])
            if wav_path is not None and wav_path.exists():
                wav = torch.from_numpy(np.load(str(wav_path)).astype(np.float32)).float().flatten()
                wavs[i] = wav
                sources[i] = "wav_cache"
            else:
                mel = torch.load(mel_root / meta["mel_path"], map_location="cpu", weights_only=True)
                fallback.append(mel.squeeze(0).float())
                fallback_idx.append(i)

        if fallback:
            self._ensure_inverse(device)
            mel = torch.stack(fallback, dim=0).to(device)
            linear = torch.exp(mel).clamp_min(1e-6)
            with torch.no_grad():
                rec = self.gl(self.inv(linear)).detach().cpu()
            for i, wav in zip(fallback_idx, rec):
                wavs[i] = wav.float().flatten()
                sources[i] = "fallback_mel_griffinlim"

        out = []
        for wav in wavs:
            if wav.numel() >= WAV_LEN:
                wav = wav[:WAV_LEN]
            else:
                wav = F.pad(wav, (0, WAV_LEN - wav.numel()))
            out.append(((wav - wav.mean()) / (wav.std() + 1e-8)).contiguous())
        return out, sources


def load_task_meta(task_key):
    ds = RQ3TaskDataset(task_key)
    rows = []
    for i in range(len(ds)):
        rows.append(ds[i])
    return ds, rows


def task_text(task_key, dms_text):
    instr = TASKS[task_key]["instruction"]
    return f"{instr} Clinical information: {dms_text}" if dms_text else instr


def load_feature_cache(cache):
    data = torch.load(cache, map_location="cpu", weights_only=False)
    data["X"] = np.nan_to_num(data["X"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return data


def fixed_text_features(texts, args):
    vectorizer = HashingVectorizer(
        n_features=args.text_dim,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
    )
    return vectorizer.transform(texts).toarray().astype(np.float32)


@torch.no_grad()
def extract_clap_features(task_key, args, device):
    global _CLAP_PROCESSOR, _CLAP_MODEL
    from transformers import ClapModel, ClapProcessor

    cache = ROOT / args.cache_dir / f"clap_{task_key}.pt"
    if cache.exists() and not args.force:
        return load_feature_cache(cache)

    ds, rows = load_task_meta(task_key)
    if _CLAP_PROCESSOR is None or _CLAP_MODEL is None:
        _CLAP_PROCESSOR = ClapProcessor.from_pretrained(args.clap_model)
        _CLAP_MODEL = ClapModel.from_pretrained(args.clap_model).to(device).eval()
    processor = _CLAP_PROCESSOR
    model = _CLAP_MODEL
    wloader = WaveLoader()

    audio_feats, text_feats, wav_sources = [], [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        wav_tensors, sources = wloader.load_batch(task_key, ds.mel_root, batch, device)
        wavs = [w.numpy() for w in wav_tensors]
        wav_sources.extend(sources)
        wavs_48k = []
        for wav in wavs:
            t = torch.from_numpy(wav).unsqueeze(0)
            t = torchaudio.functional.resample(t, SR, CLAP_SR).squeeze(0).numpy()
            wavs_48k.append(t)
        ain = processor(audios=wavs_48k, sampling_rate=CLAP_SR, return_tensors="pt", padding=True)
        ain = {k: v.to(device) for k, v in ain.items() if torch.is_tensor(v)}
        af = model.get_audio_features(**ain).float()
        af = F.normalize(af, dim=-1).cpu()

        texts = [task_text(task_key, r["dms_text"]) for r in batch]
        tin = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        tin = {k: v.to(device) for k, v in tin.items() if torch.is_tensor(v)}
        tf = model.get_text_features(**tin).float()
        tf = F.normalize(tf, dim=-1).cpu()

        audio_feats.append(af)
        text_feats.append(tf)
        print(f"  CLAP {task_key}: {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    out = pack_features(rows, torch.cat([torch.cat(audio_feats), torch.cat(text_feats)], dim=1).numpy())
    out["wav_sources"] = wav_sources
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


@torch.no_grad()
def extract_speechtokenizer_features(task_key, args, device):
    from speechtokenizer import SpeechTokenizer

    cache = ROOT / args.cache_dir / f"speechtokenizer_{task_key}.pt"
    if cache.exists() and not args.force:
        data = load_feature_cache(cache)
        _, rows = load_task_meta(task_key)
        code_dim = args.spt_nq * args.spt_codebook_size
        codes = data["X"][:, :code_dim].astype(np.float32)
        texts = [task_text(task_key, r["dms_text"]) for r in rows]
        text_X = fixed_text_features(texts, args)
        data["X"] = np.concatenate([codes, text_X], axis=1).astype(np.float32)
        return data

    ckpt_dir = ROOT / args.speechtokenizer_dir / "speechtokenizer_hubert_avg"
    model = SpeechTokenizer.load_from_checkpoint(
        str(ckpt_dir / "config.json"), str(ckpt_dir / "SpeechTokenizer.pt")
    ).to(device).eval()
    ds, rows = load_task_meta(task_key)
    wloader = WaveLoader()
    feats, texts, wav_sources = [], [], []

    for r in rows:
        wav, src = wloader.load(task_key, ds.mel_root, r, device)
        wav_sources.append(src)
        codes = model.encode(wav.view(1, 1, -1).to(device), n_q=args.spt_nq)
        codes = codes.squeeze(1).detach().cpu().numpy()
        hists = []
        for q in range(codes.shape[0]):
            hist = np.bincount(codes[q].astype(np.int64), minlength=args.spt_codebook_size)
            hist = hist.astype(np.float32) / max(hist.sum(), 1.0)
            hists.append(hist)
        feats.append(np.concatenate(hists, axis=0))
        texts.append(task_text(task_key, r["dms_text"]))
        if len(feats) % 100 == 0:
            print(f"  SpeechTokenizer {task_key}: {len(feats)}/{len(rows)}", flush=True)

    text_X = fixed_text_features(texts, args)
    X = np.concatenate([np.asarray(feats, dtype=np.float32), text_X], axis=1)
    out = pack_features(rows, X)
    out["wav_sources"] = wav_sources
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def pack_features(rows, X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "X": X.astype(np.float32),
        "y": np.asarray([r["label"] for r in rows], dtype=np.int64),
        "split": np.asarray([r["split"] for r in rows], dtype=object),
        "pid": np.asarray([r["pid"] for r in rows], dtype=object),
        "diagnosis": np.asarray([r["diagnosis"] for r in rows], dtype=object),
        "mel_path": np.asarray([r["mel_path"] for r in rows], dtype=object),
    }


def train_head(X, y, seed):
    clf = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    clf.fit(X, y)
    return clf


def score_binary(clf, X, y):
    probs = clf.predict_proba(X)
    classes = list(clf.named_steps["logisticregression"].classes_)
    pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
    scores = probs[:, pos_idx]
    pred = clf.predict(X)
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = float("nan")
    return {"auroc": float(auc), "accuracy": float(accuracy_score(y, pred)), "n": int(len(y))}


def get_train_mask(features, source, target, strict):
    data = features[source]
    mask = data["split"] == "train" if source == "icbhi_copd" else np.isin(data["split"], ["train", "val"])
    if strict and source == "b2ai_voice_pathology":
        exclude_diag = TARGET_EXCLUDE_DIAGNOSES.get(target, set())
        if exclude_diag:
            mask &= ~np.isin(data["diagnosis"], list(exclude_diag))
        target_pids = set(features[target]["pid"].tolist())
        if target_pids:
            mask &= ~np.isin(data["pid"], list(target_pids))
    return mask


def get_eval_mask(features, target, source, strict):
    data = features[target]
    if target == source:
        return data["split"] == "test"
    if strict and target.startswith("coswara_"):
        return data["split"] == "test"
    return np.ones(len(data["y"]), dtype=bool)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.table == "new":
        eval_columns = NEW_TABLE_EVAL_COLUMNS
        source_for_target = NEW_TABLE_SOURCE_FOR_TARGET
    else:
        eval_columns = LEGACY_EVAL_COLUMNS
        source_for_target = LEGACY_SOURCE_FOR_TARGET

    all_tasks = sorted(set(eval_columns.values()) | set(source_for_target.values()))
    if args.max_tasks:
        all_tasks = all_tasks[:args.max_tasks]
    print(f"Device={device}; tasks={all_tasks}", flush=True)

    features = {}
    for task_key in all_tasks:
        if args.method == "clap":
            features[task_key] = extract_clap_features(task_key, args, device)
        elif args.method == "speechtokenizer":
            features[task_key] = extract_speechtokenizer_features(task_key, args, device)
        else:
            raise ValueError(args.method)

    results = {}
    for col, target in eval_columns.items():
        if target not in features:
            continue
        source = source_for_target[target]
        if source not in features:
            print(f"Skipping {col}: missing source features {source}", flush=True)
            continue
        source_data = features[source]
        train_mask = get_train_mask(features, source, target, args.strict)
        eval_mask = get_eval_mask(features, target, source, args.strict)
        clf = train_head(source_data["X"][train_mask], source_data["y"][train_mask], args.seed)
        res = score_binary(clf, features[target]["X"][eval_mask], features[target]["y"][eval_mask])
        results[col] = res
        print(f"{args.method} {col}: AUROC={res['auroc']:.4f} ACC={res['accuracy']:.4f} n={res['n']}", flush=True)

    aucs = [v["auroc"] for v in results.values() if not np.isnan(v["auroc"])]
    results["Avg"] = {"auroc": float(np.mean(aucs)) if aucs else float("nan")}
    return {"method": args.method, "table": args.table, "strict": args.strict, "results": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["clap", "speechtokenizer"], required=True)
    p.add_argument("--table", choices=["legacy", "new"], default="legacy")
    p.add_argument("--output", default="")
    p.add_argument("--cache-dir", default="checkpoints/rq3_external_baselines/feature_cache")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--clap-model", default="laion/clap-htsat-unfused")
    p.add_argument("--speechtokenizer-dir", default="models/speechtokenizer_fnlp")
    p.add_argument("--spt-nq", type=int, default=1)
    p.add_argument("--spt-codebook-size", type=int, default=1024)
    p.add_argument("--text-dim", type=int, default=256)
    args = p.parse_args()

    out = run(args)
    output = Path(args.output or f"checkpoints/rq3_external_baselines/{args.method}_rq3_audio_text.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
