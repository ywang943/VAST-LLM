"""Run a cached OPERA-style downstream task using a RespVoice checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from data.respvoice_datasets import CachedMelDataset
from respvoice.downstream import DownstreamHead, LinearProbe
from respvoice.model import RespVoiceModel


def load_compatible_state(model, state):
    current = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"[load] skipped shape-incompatible keys: {skipped}")
    return missing, unexpected


def split_by_meta(dataset):
    split = {"train": [], "val": [], "test": []}
    for idx, sample in enumerate(dataset.samples):
        split[sample["split"]].append(idx)
    return Subset(dataset, split["train"]), Subset(dataset, split["val"]), Subset(dataset, split["test"])


@torch.no_grad()
def evaluate(model, loader, device, use_quantized, n_classes):
    model.eval()
    probs_all, preds_all, labels_all = [], [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        mel = batch["mel"].to(device)
        labels = batch["label"].to(device)
        out = model.forward_stage3(mel, use_quantized=use_quantized)
        logits = out["logits"]
        loss = F.cross_entropy(logits, labels)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)
        loss_sum += loss.item() * labels.numel()
        n += labels.numel()
        probs_all.append(probs.cpu().numpy())
        preds_all.append(preds.cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    probs = np.concatenate(probs_all)
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    if n_classes == 2:
        auroc = roc_auc_score(labels, probs[:, 1])
    else:
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    return {
        "loss": loss_sum / max(1, n),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "auroc": float(auroc),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(n_classes))).tolist(),
    }


def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype("float32")
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def labels_from_subset(subset):
    return np.asarray([int(subset.dataset.samples[i]["label"]) for i in subset.indices], dtype="int64")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-classes", type=int, required=True)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--linear-probe", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cache = Path(args.cache)
    dataset = CachedMelDataset(str(cache), str(cache / "metadata.json"), include_labels=True)
    train_ds, val_ds, test_ds = split_by_meta(dataset)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RespVoiceModel(ckpt["config"].model)
    head = LinearProbe(ckpt["config"].model.D, args.n_classes) if args.linear_probe else DownstreamHead(
        ckpt["config"].model.D, args.n_classes, use_regression=False
    )
    model.set_downstream_head(head)
    load_compatible_state(model, ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.linear_probe:
        model.freeze_encoder()
        model.freeze_vq()
        params = list(model.head.parameters())
    else:
        params = list(model.parameters())

    weights = None
    if not args.no_class_weights:
        weights = class_weights(labels_from_subset(train_ds), args.n_classes).to(device)
    opt = AdamW(params, lr=args.lr, weight_decay=5e-2)
    use_quantized = not args.continuous
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_path = out_dir / "stage3_best_auc.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for batch in train_loader:
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad()
            logits = model.forward_stage3(mel, use_quantized=use_quantized)["logits"]
            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            n += labels.numel()
        val = evaluate(model, val_loader, device, use_quantized, args.n_classes)
        print(
            f"[{epoch:03d}] train_loss={loss_sum/max(1,n):.4f} "
            f"train_acc={correct/max(1,n):.3f} val_auc={val['auroc']:.4f} val_acc={val['accuracy']:.3f}"
        )
        if val["auroc"] > best_auc:
            best_auc = val["auroc"]
            torch.save({"model_state": model.state_dict(), "config": ckpt["config"]}, best_path)

    final_path = out_dir / "stage3_final_auc.pt"
    torch.save({"model_state": model.state_dict(), "config": ckpt["config"]}, final_path)

    summary = {
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "n_classes": args.n_classes,
        "seed": args.seed,
        "representation": "z_cont" if args.continuous else "z_q",
        "linear_probe": args.linear_probe,
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
        "class_weights": None if weights is None else weights.detach().cpu().tolist(),
        "checkpoints": {},
    }
    for name in ["stage3_best_auc.pt", "stage3_final_auc.pt"]:
        saved = torch.load(out_dir / name, map_location="cpu", weights_only=False)
        m = RespVoiceModel(saved["config"].model)
        h = LinearProbe(saved["config"].model.D, args.n_classes) if args.linear_probe else DownstreamHead(
            saved["config"].model.D, args.n_classes, use_regression=False
        )
        m.set_downstream_head(h)
        load_compatible_state(m, saved["model_state"])
        m.to(device)
        summary["checkpoints"][name] = {
            "val": evaluate(m, val_loader, device, use_quantized, args.n_classes),
            "test": evaluate(m, test_loader, device, use_quantized, args.n_classes),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
