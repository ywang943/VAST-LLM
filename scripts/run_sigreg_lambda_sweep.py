#!/usr/bin/env python3
"""Single-knob SIGReg lambda sweep.

For each lambda:
  1. pretrain the same HTS-AT+CSAF LeJEPA model with only --lam-sig changed;
  2. train the same EMA VQ protocol on the frozen encoder;
  3. measure clip-level effective rank and VQ util/perplexity;
  4. evaluate the discrete-token path via VQ-token histograms on RQ3 source->target.

This is intentionally cheaper than the final 150-epoch model; it is a controlled
parameter-sensitivity experiment, not a replacement for the main checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from respvoice.htsat_encoder import build_htsat_encoder  # noqa: E402
from respvoice.vq import VectorQuantizer  # noqa: E402
from data.respvoice_datasets import CachedMelDataset  # noqa: E402
from scripts.run_rq3_audio_baselines import (  # noqa: E402
    EVAL_COLUMNS,
    SOURCE_FOR_TARGET,
    TARGET_EXCLUDE_DIAGNOSES,
)
from scripts.run_rq3_llm import TASKS  # noqa: E402


DEFAULT_CACHES = [
    "opera_icbhi_disease",
    "svd_full",
    "coswara_covid_cough",
    "b2ai_voice_pathology",
]


class MelLabelDataset(Dataset):
    def __init__(self, task_key):
        cfg = TASKS[task_key]
        self.root = ROOT / cfg["mel_root"]
        meta = json.loads((self.root / "metadata.json").read_text())
        self.samples = [s for s in meta.get("samples", []) if "label" in s]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mel = torch.load(self.root / s["path"], map_location="cpu", weights_only=True).float()
        return {
            "mel": mel,
            "label": int(s["label"]),
            "split": s.get("split", "train"),
            "pid": str(s.get("pid", s.get("participant_id", ""))),
            "diagnosis": str(s.get("diagnosis", "")),
        }


def collate(batch):
    return {
        "mel": torch.stack([b["mel"] for b in batch]),
        "label": np.asarray([b["label"] for b in batch], dtype=np.int64),
        "split": np.asarray([b["split"] for b in batch], dtype=object),
        "pid": np.asarray([b["pid"] for b in batch], dtype=object),
        "diagnosis": np.asarray([b["diagnosis"] for b in batch], dtype=object),
    }


def run_cmd(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd), flush=True)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def lam_tag(lam):
    return f"lam_{lam:g}".replace(".", "p")


def load_encoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=True, use_csaf=True)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  encoder load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return encoder.to(device).eval()


def load_vq(vq_path, device):
    data = torch.load(vq_path, map_location="cpu", weights_only=False)
    vq = VectorQuantizer(
        codebook_size=int(data["codebook_size"]),
        D=int(data["D"]),
        l2_normalize=bool(data.get("l2_normalize", True)),
    )
    vq.load_state_dict(data["vq_state"], strict=False)
    return vq.to(device).eval(), data


@torch.no_grad()
def clip_level_deff(encoder, caches, device, max_samples=6000):
    datasets = []
    for name in caches:
        root = ROOT / "data/mel_cache" / name
        ds = CachedMelDataset(root=str(root), meta_file=str(root / "metadata.json"), include_labels=False)
        datasets.append(ds)
    loader = DataLoader(ConcatDataset(datasets), batch_size=64, shuffle=True, num_workers=2)
    feats, seen = [], 0
    for batch in loader:
        mel = batch["mel"].to(device)
        z = encoder(mel).mean(dim=1).float().cpu().numpy()
        feats.append(z)
        seen += z.shape[0]
        if seen >= max_samples:
            break
    Z = np.concatenate(feats, axis=0)[:max_samples]
    Z = Z - Z.mean(axis=0, keepdims=True)
    cov = (Z.T @ Z) / max(len(Z), 1)
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    p = eig / (eig.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


@torch.no_grad()
def extract_token_hist(task_key, encoder, vq, K, device):
    ds = MelLabelDataset(task_key)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate)
    X, y, split, pid, diagnosis = [], [], [], [], []
    for batch in loader:
        ids = vq(encoder(batch["mel"].to(device)))["ids"].reshape(batch["mel"].size(0), -1)
        for row in ids.cpu().numpy():
            hist = np.bincount(row.astype(np.int64), minlength=K).astype(np.float32)
            hist /= max(hist.sum(), 1.0)
            X.append(hist)
        y.append(batch["label"])
        split.extend(batch["split"].tolist())
        pid.extend(batch["pid"].tolist())
        diagnosis.extend(batch["diagnosis"].tolist())
    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.concatenate(y),
        "split": np.asarray(split, dtype=object),
        "pid": np.asarray(pid, dtype=object),
        "diagnosis": np.asarray(diagnosis, dtype=object),
    }


def train_mask(features, source, target):
    data = features[source]
    mask = data["split"] == "train" if source == "icbhi_copd" else np.isin(data["split"], ["train", "val"])
    if source == "b2ai_voice_pathology":
        exclude_diag = TARGET_EXCLUDE_DIAGNOSES.get(target, set())
        if exclude_diag:
            mask &= ~np.isin(data["diagnosis"], list(exclude_diag))
        target_pids = set(features[target]["pid"].tolist())
        if target_pids:
            mask &= ~np.isin(data["pid"], list(target_pids))
    return mask


def eval_mask(features, target, source):
    data = features[target]
    if target == source:
        return data["split"] == "test"
    if target.startswith("coswara_"):
        return data["split"] == "test"
    return np.ones(len(data["y"]), dtype=bool)


def token_path_auroc(encoder, vq, K, device):
    tasks = sorted(set(EVAL_COLUMNS.values()) | set(SOURCE_FOR_TARGET.values()))
    features = {tk: extract_token_hist(tk, encoder, vq, K, device) for tk in tasks}
    out = {}
    for col, target in EVAL_COLUMNS.items():
        source = SOURCE_FOR_TARGET[target]
        tr = train_mask(features, source, target)
        ev = eval_mask(features, target, source)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(features[source]["X"][tr], features[source]["y"][tr])
        probs = clf.predict_proba(features[target]["X"][ev])
        classes = list(clf.named_steps["logisticregression"].classes_)
        pos = classes.index(1) if 1 in classes else len(classes) - 1
        score = probs[:, pos]
        pred = clf.predict(features[target]["X"][ev])
        y = features[target]["y"][ev]
        try:
            auc = roc_auc_score(y, score)
        except ValueError:
            auc = float("nan")
        out[col] = {"auroc": float(auc), "accuracy": float(accuracy_score(y, pred)), "n": int(len(y))}
    aucs = [v["auroc"] for v in out.values() if not np.isnan(v["auroc"])]
    out["Avg"] = {"auroc": float(np.mean(aucs)) if aucs else float("nan")}
    return out


def plot_summary(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    lams = [r["lambda"] for r in rows]
    auc = [r["token_path_avg_auroc"] for r in rows]
    util = [r["vq_utilization"] for r in rows]
    deff = [r["d_eff"] for r in rows]
    plt.rcParams.update({"font.size": 13, "pdf.fonttype": 42, "ps.fonttype": 42})
    for x, xlabel, name in [
        (util, "Codebook utilization", "auroc_vs_utilization"),
        (deff, r"Effective rank $d_{eff}$", "auroc_vs_deff"),
        (lams, r"SIGReg weight $\lambda$", "metrics_vs_lambda"),
    ]:
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        ax.set_facecolor("#f2f2f2")
        ax.grid(color="white", linewidth=1.2)
        ax.plot(x, auc, marker="o", linewidth=2.5, color="#b84e4e", label="Token AUROC")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Discrete-token AUROC")
        for xi, yi in zip(x, auc):
            ax.text(xi, yi + 0.005, f"{yi:.3f}", ha="center", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.005, 0.02, 0.05, 0.1])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vq-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--pretrain-caches", nargs="+", default=DEFAULT_CACHES)
    p.add_argument("--out-dir", default="checkpoints/sigreg_lambda_sweep")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for lam in args.lambdas:
        tag = lam_tag(lam)
        lam_dir = out_dir / tag
        ckpt_dir = lam_dir / "pretrain"
        vq_path = lam_dir / "vq.pt"
        result_path = lam_dir / "result.json"
        lam_dir.mkdir(parents=True, exist_ok=True)
        encoder_ckpt = ckpt_dir / "htsat_lejepa_best.pt"

        if args.force or not encoder_ckpt.exists():
            run_cmd([
                sys.executable, "-u", "scripts/run_htsat_lejepa_pretrain.py",
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--grad-accum", str(args.grad_accum),
                "--lr", "1e-4",
                "--backbone-lr", "1e-5",
                "--warmup-epochs", "1",
                "--lam-sig", str(lam),
                "--checkpoint-dir", str(ckpt_dir),
                "--pretrain-caches", *args.pretrain_caches,
            ], lam_dir / "pretrain.log")

        if args.force or not vq_path.exists():
            run_cmd([
                sys.executable, "-u", "scripts/train_mel_htsat_vq.py",
                "--encoder-ckpt", str(encoder_ckpt),
                "--include", *args.pretrain_caches,
                "--no-skip-derived",
                "--steps", str(args.vq_steps),
                "--batch-size", str(args.batch_size),
                "--out", str(vq_path.relative_to(ROOT)),
            ], lam_dir / "vq.log")

        encoder = load_encoder(encoder_ckpt, device)
        vq, vq_data = load_vq(vq_path, device)
        K = int(vq_data["codebook_size"])
        deff = clip_level_deff(encoder, args.pretrain_caches, device)
        token_results = token_path_auroc(encoder, vq, K, device)
        row = {
            "lambda": lam,
            "encoder_ckpt": str(encoder_ckpt),
            "vq_ckpt": str(vq_path),
            "d_eff": deff,
            "vq_utilization": float(vq_data.get("final_sampled_utilization", float("nan"))),
            "vq_perplexity": float(vq_data.get("final_sampled_perplexity", float("nan"))),
            "token_path_avg_auroc": float(token_results["Avg"]["auroc"]),
            "token_path_results": token_results,
        }
        result_path.write_text(json.dumps(row, indent=2))
        rows.append(row)
        del encoder, vq
        torch.cuda.empty_cache()
        print(json.dumps(row, indent=2), flush=True)

    summary = {"rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_summary(rows, out_dir)
    print(f"Saved {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
