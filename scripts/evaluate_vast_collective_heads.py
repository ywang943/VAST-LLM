#!/usr/bin/env python3
"""Evaluate task-specific linear heads saved by collective VAST adaptation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from respvoice.htsat_encoder import build_htsat_encoder  # noqa: E402
from scripts.run_rq1_mel_linear_probe import S_TASKS  # noqa: E402


class TestMelDataset(Dataset):
    def __init__(self, task_key, cfg):
        self.task_key = task_key
        self.root = ROOT / cfg["mel_root"]
        raw = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        samples = raw.get("samples", raw if isinstance(raw, list) else [])
        self.samples = [
            s for s in samples
            if s.get("split") == "test" and "label" in s and (self.root / s["path"]).exists()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.load(str(self.root / s["path"]), map_location="cpu"), int(s["label"])


def collate(batch):
    return {
        "mel": torch.stack([x[0] for x in batch]),
        "label": torch.tensor([x[1] for x in batch], dtype=torch.long),
    }


def auroc(y_true, probs):
    if probs.shape[1] == 2:
        return float(roc_auc_score(y_true, probs[:, 1]))
    return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ROOT / args.ckpt), map_location="cpu", weights_only=False)
    enc_state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=True, use_csaf=True)
    encoder.load_state_dict(enc_state, strict=True)
    encoder.to(device).eval()

    heads = nn.ModuleDict()
    task_classes = ckpt.get("meta", {}).get("task_classes", {})
    for task_key, n_cls in task_classes.items():
        heads[task_key] = nn.Linear(768, int(n_cls))
    heads.load_state_dict(ckpt["heads"], strict=True)
    heads.to(device).eval()

    results = {}
    with torch.no_grad():
        for task_key, cfg in S_TASKS.items():
            ds = TestMelDataset(task_key, cfg)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate,
                pin_memory=torch.cuda.is_available(),
            )
            probs, labels = [], []
            for batch in loader:
                mel = batch["mel"].to(device, non_blocking=True)
                feat = encoder(mel).mean(dim=1)
                logits = heads[task_key](feat)
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                labels.append(batch["label"].numpy())
            p = np.concatenate(probs, axis=0)
            y = np.concatenate(labels, axis=0)
            results[task_key] = {
                "auroc": auroc(y, p),
                "accuracy": float(accuracy_score(y, p.argmax(axis=1))),
                "n_test": int(len(y)),
            }
            print(f"{task_key}: AUROC={results[task_key]['auroc']:.4f} Acc={results[task_key]['accuracy']:.4f}")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    vals = [x["auroc"] for x in results.values()]
    print(f"Avg={float(np.mean(vals)):.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
