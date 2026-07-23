"""Lightweight OPERA feature baseline on the official icbhidisease split.

This mirrors OPERA's `linear_evaluation_icbhidisease` protocol without
Pytorch Lightning, which is unreliable/slow in this Windows environment.

Inputs:
  opera_src/feature/icbhidisease_eval/{operaCT_feature.npy,labels.npy,split.npy}

Protocol:
  - keep Healthy vs COPD
  - use OPERA split.npy for train/test
  - split train into train/val with random_state=1337, stratified
  - train a linear head for 64 epochs
  - decay LR by 0.97 per epoch
  - checkpoint by validation AUROC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "opera_src" / "feature" / "icbhidisease_eval"


def load_split(feature_name: str):
    x = np.load(FEATURE_DIR / f"{feature_name}_feature.npy").squeeze().astype("float32")
    labels_raw = np.load(FEATURE_DIR / "labels.npy", allow_pickle=True)
    split_raw = np.load(FEATURE_DIR / "split.npy", allow_pickle=True)

    mask = (labels_raw == "Healthy") | (labels_raw == "COPD")
    x = x[mask]
    split_raw = split_raw[mask]
    y = np.asarray([0 if label == "Healthy" else 1 for label in labels_raw[mask]], dtype="int64")

    trainval_idx = np.where(split_raw != "test")[0]
    test_idx = np.where(split_raw == "test")[0]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.2,
        random_state=1337,
        stratify=y[trainval_idx],
    )
    return x, y, train_idx, val_idx, test_idx


def make_loader(x, y, idx, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(y[idx]))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs_all, pred_all, y_all = [], [], []
    loss_sum, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        probs = F.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(1)
        loss_sum += loss.item() * yb.numel()
        n += yb.numel()
        probs_all.append(probs.cpu().numpy())
        pred_all.append(pred.cpu().numpy())
        y_all.append(yb.cpu().numpy())

    y_true = np.concatenate(y_all)
    y_pred = np.concatenate(pred_all)
    y_prob = np.concatenate(probs_all)
    return {
        "loss": loss_sum / max(1, n),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "recall_healthy": float(((y_pred == 0) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())),
        "recall_copd": float(((y_pred == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())),
    }


def run_once(args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x, y, train_idx, val_idx, test_idx = load_split(args.feature)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(x.shape[1], 2)).to(device)
    nn.init.normal_(model[0].weight, mean=0.0, std=0.01)
    nn.init.zeros_(model[0].bias)

    train_loader = make_loader(x, y, train_idx, args.batch_size, True)
    val_loader = make_loader(x, y, val_idx, args.batch_size, False)
    test_loader = make_loader(x, y, test_idx, args.batch_size, False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = None
    best_val_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            l2 = sum(param.pow(2).sum() for param in model.parameters())
            loss = loss + args.l2_strength * l2
            loss.backward()
            opt.step()

        for group in opt.param_groups:
            group["lr"] *= args.lr_decay

        val = evaluate(model, val_loader, device)
        if val["auroc"] > best_val_auc:
            best_val_auc = val["auroc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    final = {
        "val": evaluate(model, val_loader, device),
        "test": evaluate(model, test_loader, device),
    }
    if best_state is not None:
        model.load_state_dict(best_state)
    best = {
        "val": evaluate(model, val_loader, device),
        "test": evaluate(model, test_loader, device),
    }

    return {
        "seed": seed,
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
        "label_counts": {
            "train": np.bincount(y[train_idx], minlength=2).tolist(),
            "val": np.bincount(y[val_idx], minlength=2).tolist(),
            "test": np.bincount(y[test_idx], minlength=2).tolist(),
        },
        "best": best,
        "final": final,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", default="operaCT")
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-decay", type=float, default=0.97)
    p.add_argument("--l2-strength", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", default="./checkpoints/opera_feature_baseline/operaCT_icbhidisease.json")
    args = p.parse_args()

    results = [run_once(args, seed) for seed in args.seeds]
    test_aurocs = [r["best"]["test"]["auroc"] for r in results]
    summary = {
        "feature": args.feature,
        "protocol": "OPERA icbhidisease Healthy-vs-COPD, official split, valid-AUC selection",
        "epochs": args.epochs,
        "lr": args.lr,
        "l2_strength": args.l2_strength,
        "results": results,
        "best_test_auroc_mean": float(np.mean(test_aurocs)),
        "best_test_auroc_std": float(np.std(test_aurocs)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
