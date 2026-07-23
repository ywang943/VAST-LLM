#!/usr/bin/env python3
"""SIGReg lambda sensitivity without text.

This script keeps the evaluation on the encoder side:
  - clip-level d_eff on the same RQ2 task pool;
  - matched VQ utilization/perplexity;
  - frozen VAST encoder -> pooled features -> linear probe AUROC on one T task.

The short lambda sweep checkpoints are plotted as a controlled sensitivity curve.
The paper/main default model is included as a separate lambda=0.02 reference point
so the figure can be read against the main RQ2 table without mixing protocols.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from respvoice.htsat_encoder import build_htsat_encoder  # noqa: E402
from respvoice.vq import VectorQuantizer  # noqa: E402


RQ2_TASKS = {
    "icbhi_copd": {"mel_root": "data/mel_cache/opera_icbhi_disease"},
    "svd_pathology": {"mel_root": "data/mel_cache/svd_full"},
    "coswara_covid_cough": {"mel_root": "data/mel_cache/coswara_covid_cough"},
    "b2ai_voice_pathology": {"mel_root": "data/mel_cache/b2ai_voice_pathology"},
}

T_TASKS = {
    "svd_pathology": {
        "name": "T7 SVD V+P",
        "mel_root": "data/mel_cache/svd_full",
        "note": "uses the labeled SVD train/val/test split for supervised VAST-LP",
    },
    "coswara_covid_breathing": {
        "name": "T4 Covid Breath",
        "mel_root": "data/mel_cache/coswara_covid_breathing",
        "note": "uses available labeled train/val/test split for supervised VAST-LP",
    },
    "coswara_smoker_breathing": {
        "name": "T5 Smoker Breath",
        "mel_root": "data/mel_cache/coswara_smoker_breathing",
        "note": "uses available labeled train/val/test split for supervised VAST-LP",
    },
}

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


class LabeledMelDataset(Dataset):
    def __init__(self, mel_root: str):
        self.root = ROOT / mel_root
        meta = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        raw = meta.get("samples", meta if isinstance(meta, list) else [])
        self.samples = [
            s for s in raw
            if "label" in s and (self.root / s["path"]).exists()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        mel = torch.load(str(self.root / s["path"]), map_location="cpu", weights_only=True).float()
        return {
            "mel": mel,
            "label": int(s["label"]),
            "split": s.get("split", "train"),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch):
    return {
        "mel": torch.stack([b["mel"] for b in batch], dim=0),
        "label": np.asarray([b["label"] for b in batch], dtype=np.int64),
        "split": np.asarray([b["split"] for b in batch], dtype=object),
    }


def lam_tag(lam: float) -> str:
    return f"lam_{lam:g}".replace(".", "p")


def load_encoder(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=True, use_csaf=True)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{ckpt_path} load mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device).eval()


def load_vq(vq_path: Path, device: torch.device):
    data = torch.load(str(vq_path), map_location="cpu", weights_only=False)
    vq = VectorQuantizer(
        codebook_size=int(data["codebook_size"]),
        D=int(data["D"]),
        l2_normalize=bool(data.get("l2_normalize", True)),
    )
    missing, unexpected = vq.load_state_dict(data["vq_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{vq_path} VQ mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    for p in vq.parameters():
        p.requires_grad = False
    return vq.to(device).eval(), data


@torch.no_grad()
def extract_features(encoder, mel_root: str, device: torch.device, batch_size: int, workers: int):
    ds = LabeledMelDataset(mel_root)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    feats, labels, splits = [], [], []
    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        z = encoder(mel)
        mean = z.mean(dim=1)
        std = z.std(dim=1)
        feats.append(mean.cpu().numpy())
        labels.append(batch["label"])
        splits.extend(batch["split"].tolist())
    return {
        "mean": np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "splits": np.asarray(splits, dtype=object),
    }


@torch.no_grad()
def measure_vq_usage(encoder, vq, mel_roots, device, batch_size, workers, max_batches):
    counts = torch.zeros(vq.codebook_size, device=device)
    seen_samples = 0
    seen_tokens = 0
    for mel_root in mel_roots:
        ds = LabeledMelDataset(mel_root)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        )
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            mel = batch["mel"].to(device, non_blocking=True)
            ids = vq(encoder(mel))["ids"].reshape(-1)
            counts += torch.bincount(ids, minlength=vq.codebook_size).float()
            seen_samples += int(mel.size(0))
            seen_tokens += int(ids.numel())
    used = int((counts > 0).sum().item())
    probs = counts / (counts.sum() + 1e-10)
    perplexity = float((-(probs * (probs + 1e-10).log()).sum()).exp().item())
    return {
        "utilization": float(used / vq.codebook_size),
        "perplexity": perplexity,
        "used_codes": used,
        "seen_samples": seen_samples,
        "seen_tokens": seen_tokens,
    }


def effective_rank(features: np.ndarray) -> float:
    x = features.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    p = eig / (eig.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def auroc_from_probs(y_true, probs, classes):
    if len(np.unique(y_true)) == 2:
        pos_idx = list(classes).index(1) if 1 in classes else len(classes) - 1
        return float(roc_auc_score(y_true, probs[:, pos_idx]))
    return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))


def fit_linear_probe(features: np.ndarray, labels: np.ndarray, splits: np.ndarray):
    train_mask = np.isin(splits, ["train", "val"])
    test_mask = splits == "test"
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError("Selected T task has no train/val or no test split")

    x_train_all = features[train_mask]
    y_train_all = labels[train_mask]
    x_test = features[test_mask]
    y_test = labels[test_mask]
    if len(np.unique(y_train_all)) < 2 or len(np.unique(y_test)) < 2:
        raise RuntimeError("Selected T task has fewer than two classes in train or test")

    counts = np.bincount(y_train_all)
    counts = counts[counts > 0]
    n_splits = int(min(5, counts.min()))

    best_c = 1.0
    cv_scores = {}
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for c in C_GRID:
            scores = []
            for tr, va in cv.split(x_train_all, y_train_all):
                scaler = StandardScaler()
                x_tr = scaler.fit_transform(x_train_all[tr])
                x_va = scaler.transform(x_train_all[va])
                clf = LogisticRegression(
                    max_iter=4000,
                    C=c,
                    solver="lbfgs",
                    class_weight="balanced",
                )
                clf.fit(x_tr, y_train_all[tr])
                try:
                    scores.append(auroc_from_probs(y_train_all[va], clf.predict_proba(x_va), clf.classes_))
                except ValueError:
                    pass
            if scores:
                cv_scores[c] = float(np.mean(scores))
    if cv_scores:
        best_c = max(cv_scores, key=cv_scores.get)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_all)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(
        max_iter=4000,
        C=best_c,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(x_train, y_train_all)
    probs = clf.predict_proba(x_test)
    pred = clf.predict(x_test)
    return {
        "auroc": auroc_from_probs(y_test, probs, clf.classes_),
        "accuracy": float(accuracy_score(y_test, pred)),
        "best_c": float(best_c),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "class_counts_train": {str(i): int((y_train_all == i).sum()) for i in np.unique(y_train_all)},
        "class_counts_test": {str(i): int((y_test == i).sum()) for i in np.unique(y_test)},
    }


def build_rows(args):
    rows = []
    sweep_root = ROOT / args.sweep_dir
    for lam in args.lambdas:
        tag = lam_tag(lam)
        rows.append({
            "label": f"lambda={lam:g}",
            "kind": "controlled_sweep",
            "lambda": float(lam),
            "encoder_ckpt": str(sweep_root / tag / "pretrain" / "htsat_lejepa_best.pt"),
            "vq_ckpt": str(sweep_root / tag / "vq.pt"),
        })
    if args.include_main_reference:
        rows.append({
            "label": "main default lambda=0.02",
            "kind": "main_reference",
            "lambda": 0.02,
            "encoder_ckpt": str(ROOT / args.main_encoder_ckpt),
            "vq_ckpt": str(ROOT / args.main_vq_ckpt),
            "paper_table4": {
                "d_eff": 192.5,
                "codebook_utilization": 0.992,
                "perplexity": 321.6,
            },
        })
    return rows


def plot_results(rows, task_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    controlled = sorted([r for r in rows if r["kind"] == "controlled_sweep"], key=lambda r: r["lambda"])
    refs = [r for r in rows if r["kind"] == "main_reference"]

    metrics = [
        ("d_eff_clip_level", r"clip-level $d_{\mathrm{eff}}$"),
        ("codebook_utilization", "codebook utilization"),
        ("perplexity", "perplexity"),
        ("lp_auroc", f"{task_name} VAST-LP AUROC"),
    ]
    colors = ["#3b6ea8", "#4f8f5a", "#8f5fa8", "#b65353"]
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    axes = axes.ravel()
    xs = [r["lambda"] for r in controlled]
    for ax, (key, ylabel), color in zip(axes, metrics, colors):
        ys = [r[key] for r in controlled]
        ax.set_facecolor("#f2f2f2")
        ax.grid(color="white", linewidth=1.2)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(1.0)
        ax.plot(xs, ys, marker="o", linewidth=2.4, markersize=7, color=color, label="controlled sweep")
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.3f}" if y < 10 else f"{y:.1f}", fontsize=11,
                    fontweight="bold", ha="center", va="bottom", color="#222222")
        if refs:
            rx = [r["lambda"] for r in refs]
            ry = [r[key] for r in refs]
            ax.scatter(rx, ry, marker="*", s=190, color="#222222", edgecolor="white",
                       linewidth=0.8, zorder=5, label="main default")
            for x, y in zip(rx, ry):
                ax.text(x, y, f"{y:.3f}" if y < 10 else f"{y:.1f}", fontsize=11,
                        fontweight="bold", ha="left", va="top", color="#111111")
        ax.set_xlabel(r"SIGReg weight $\lambda$")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(out_dir / "sigreg_lambda_encoder_lp.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "sigreg_lambda_encoder_lp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", default="checkpoints/sigreg_lambda_sweep")
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.005, 0.02, 0.05, 0.1])
    p.add_argument("--t-task", choices=list(T_TASKS), default="svd_pathology")
    p.add_argument("--main-encoder-ckpt", default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt")
    p.add_argument("--main-vq-ckpt", default="checkpoints/vq/mel_htsat_v3_full_vq_K512_ema20k_all.pt")
    p.add_argument("--include-main-reference", action="store_true", default=True)
    p.add_argument("--no-main-reference", dest="include_main_reference", action="store_false")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--vq-max-batches-per-root", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="checkpoints/sigreg_lambda_encoder_lp")
    args = p.parse_args()

    seed_everything(args.seed)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"T task: {args.t_task} ({T_TASKS[args.t_task]['name']})")

    rows = []
    rq2_roots = [cfg["mel_root"] for cfg in RQ2_TASKS.values()]
    for spec in build_rows(args):
        print("\n" + "=" * 80)
        print(spec["label"])
        print("=" * 80)
        encoder_ckpt = Path(spec["encoder_ckpt"])
        vq_ckpt = Path(spec["vq_ckpt"])
        if not encoder_ckpt.exists():
            raise FileNotFoundError(encoder_ckpt)
        if not vq_ckpt.exists():
            raise FileNotFoundError(vq_ckpt)

        encoder = load_encoder(encoder_ckpt, device)
        vq, vq_data = load_vq(vq_ckpt, device)

        rq2_feats = []
        for cfg in RQ2_TASKS.values():
            data = extract_features(encoder, cfg["mel_root"], device, args.batch_size, args.num_workers)
            rq2_feats.append(data["mean"])
        combined = np.concatenate(rq2_feats, axis=0)
        deff = effective_rank(combined)
        print(f"clip-level d_eff={deff:.3f} on {combined.shape[0]} clips")

        measured = measure_vq_usage(
            encoder,
            vq,
            rq2_roots,
            device,
            args.batch_size,
            args.num_workers,
            None if args.vq_max_batches_per_root <= 0 else args.vq_max_batches_per_root,
        )
        print(f"VQ util={measured['utilization']:.3f} perplexity={measured['perplexity']:.1f}")

        task_cfg = T_TASKS[args.t_task]
        t_data = extract_features(encoder, task_cfg["mel_root"], device, args.batch_size, args.num_workers)
        lp = fit_linear_probe(t_data["mean"], t_data["labels"], t_data["splits"])
        print(
            f"{task_cfg['name']} LP AUROC={lp['auroc']:.4f} "
            f"ACC={lp['accuracy']:.4f} C={lp['best_c']} n={lp['n_train']}/{lp['n_test']}"
        )

        row = {
            **spec,
            "d_eff_clip_level": deff,
            "codebook_utilization": measured["utilization"],
            "perplexity": measured["perplexity"],
            "vq_metadata": {
                "codebook_size": int(vq_data["codebook_size"]),
                "D": int(vq_data["D"]),
                "final_sampled_utilization": vq_data.get("final_sampled_utilization"),
                "final_sampled_perplexity": vq_data.get("final_sampled_perplexity"),
                "encoder_checkpoint": vq_data.get("encoder_checkpoint"),
            },
            "lp_task": args.t_task,
            "lp_task_name": task_cfg["name"],
            "lp_auroc": lp["auroc"],
            "lp_accuracy": lp["accuracy"],
            "lp": lp,
        }
        rows.append(row)
        (out_dir / f"{spec['kind']}_{lam_tag(spec['lambda'])}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        del encoder, vq
        torch.cuda.empty_cache()

    summary = {
        "protocol": (
            "no text; frozen encoder mean-pooled clip representations; RQ2 d_eff on "
            "icbhi/svd/coswara-cough/b2ai; VQ usage remeasured on same RQ2 task pool; "
            f"linear probe on {args.t_task}"
        ),
        "t_task": args.t_task,
        "t_task_name": T_TASKS[args.t_task]["name"],
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        f"# SIGReg lambda encoder-only sensitivity ({T_TASKS[args.t_task]['name']})",
        "",
        "| Row | lambda | d_eff | Util. | Perplexity | LP AUROC | LP ACC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['kind']} | {r['lambda']:.3g} | {r['d_eff_clip_level']:.1f} | "
            f"{r['codebook_utilization']:.3f} | {r['perplexity']:.1f} | "
            f"{r['lp_auroc']:.3f} | {r['lp_accuracy']:.3f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    plot_results(rows, T_TASKS[args.t_task]["name"], out_dir)

    paper_dir = ROOT / "paper_figure"
    paper_dir.mkdir(exist_ok=True)
    for suffix in ["pdf", "png"]:
        src = out_dir / f"sigreg_lambda_encoder_lp.{suffix}"
        dst = paper_dir / f"sigreg_lambda_encoder_lp.{suffix}"
        dst.write_bytes(src.read_bytes())

    print(f"\nSaved: {out_dir / 'summary.md'}")
    print(f"Figure: {out_dir / 'sigreg_lambda_encoder_lp.pdf'}")
    print(f"Paper copy: {paper_dir / 'sigreg_lambda_encoder_lp.pdf'}")


if __name__ == "__main__":
    main()
