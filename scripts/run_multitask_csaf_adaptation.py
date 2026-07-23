"""Compare downstream fusion strategies with a frozen HTS-AT backbone.

For every task and seed, only the task-side module is trained:
  - stage4: final HTS-AT stage + attention pool + classifier
  - concat: direct channel concatenation of e1/e2/e3/e4 + same head
  - csaf: pretrained TPA-CSAF + same head, with CSAF fine-tuned per task
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, TensorDataset

from respvoice.csa_fusion import CrossScaleAttentionFusion
from respvoice.downstream import AttentionPool
from scripts.run_frozen_multitask_benchmark import (
    TASKS, MultiCacheDataset, encode_stages, load_encoder, metric, split_indices,
)


class Stage4Probe(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.pool = AttentionPool(768)
        self.head = nn.Linear(768, n_classes)

    def forward(self, stages):
        return self.head(self.pool(stages[3]))


class DirectConcatProbe(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.pool = AttentionPool(192 + 384 + 768 + 768)
        self.head = nn.Linear(192 + 384 + 768 + 768, n_classes)

    def forward(self, stages):
        return self.head(self.pool(torch.cat(stages, dim=-1)))


class CSAFProbe(nn.Module):
    def __init__(self, n_classes, csaf_state):
        super().__init__()
        self.csaf = CrossScaleAttentionFusion(
            D=768, n_scales=4, n_heads=8, depth=2,
            scale_dims=(192, 384, 768, 768),
        )
        self.csaf.load_state_dict(csaf_state)
        self.pool = AttentionPool(768)
        self.head = nn.Linear(768, n_classes)

    def forward(self, stages):
        return self.head(self.pool(self.csaf(stages)))


@torch.no_grad()
def cache_stages(encoder, dataset, device, batch_size):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True,
    )
    collected = [[], [], [], []]
    labels = []
    encoder.eval()
    for batch in loader:
        waveform = batch.get("wav")
        if waveform is not None:
            waveform = waveform.to(device, non_blocking=True)
        stages = encode_stages(
            encoder, batch["mel"].to(device, non_blocking=True), waveform
        )
        for output, stage in zip(collected, stages):
            output.append(stage.cpu())
        labels.append(batch["label"])
    return [torch.cat(parts) for parts in collected], torch.cat(labels)


def build_probe(variant, n_classes, csaf_state):
    if variant == "stage4":
        return Stage4Probe(n_classes)
    if variant == "concat":
        return DirectConcatProbe(n_classes)
    return CSAFProbe(n_classes, csaf_state)


def evaluate_probe(probe, loader, device, n_classes):
    probe.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            stages = [tensor.to(device, non_blocking=True) for tensor in batch[:-1]]
            logits.append(probe(stages).cpu())
            labels.append(batch[-1])
    return metric(torch.cat(logits), torch.cat(labels), n_classes)


def run_seed(stages, labels, splits, variant, n_classes, csaf_state,
             seed, epochs, lr, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = TensorDataset(*stages, labels)
    train_idx, val_idx, test_idx = splits
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=32, shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False)

    probe = build_probe(variant, n_classes, csaf_state).to(device)
    params = list(probe.parameters())
    counts = torch.bincount(labels[train_idx], minlength=n_classes).float()
    weights = (counts.sum() / (counts.clamp_min(1) * n_classes)).to(device)
    optimizer = AdamW(params, lr=lr, weight_decay=1e-2)

    best_auc, best_state = -float("inf"), None
    for _ in range(epochs):
        probe.train()
        for batch in train_loader:
            batch_stages = [tensor.to(device, non_blocking=True) for tensor in batch[:-1]]
            y = batch[-1].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.cross_entropy(probe(batch_stages), y, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        val_result = evaluate_probe(probe, val_loader, device, n_classes)
        score = val_result["auroc"]
        if np.isfinite(score) and score > best_auc:
            best_auc = score
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    result = evaluate_probe(probe, test_loader, device, n_classes)
    result["trainable_params"] = sum(p.numel() for p in probe.parameters())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--variants", nargs="+", choices=("stage4", "concat", "csaf"),
                        default=["stage4", "concat", "csaf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="checkpoints/csaf_multitask_adapt/results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, checkpoint = load_encoder(args.checkpoint, device)
    csaf_state = {
        key: value.detach().cpu().clone()
        for key, value in encoder.csaf.state_dict().items()
    }
    results = {
        "protocol": "frozen HTS-AT + task-specific fusion adaptation",
        "checkpoint": args.checkpoint,
        "pretrain_epoch": checkpoint.get("epoch"),
        "tasks": {},
    }

    for task_key in args.tasks:
        cfg = TASKS[task_key]
        dataset = MultiCacheDataset(
            cfg["roots"], cfg.get("wav_roots") if hasattr(encoder, "hubert_cnn") else None
        )
        splits = split_indices(dataset, cfg["split"])
        print(f"\n{cfg['name']}: caching four frozen HTS-AT stages")
        stages, labels = cache_stages(encoder, dataset, device, args.batch_size)
        task_results = {"note": cfg.get("note"), "variants": {}}
        for variant in args.variants:
            per_seed = []
            for seed in args.seeds:
                result = run_seed(
                    stages, labels, splits, variant, cfg["n_classes"],
                    csaf_state, seed, args.epochs, args.lr, device,
                )
                per_seed.append(result)
                print(f"  {variant} seed={seed}: AUROC={result['auroc']:.4f}")
            aucs = [result["auroc"] for result in per_seed]
            task_results["variants"][variant] = {
                "auroc_mean": float(np.nanmean(aucs)),
                "auroc_std": float(np.nanstd(aucs)),
                "trainable_params": per_seed[0]["trainable_params"],
                "per_seed": per_seed,
            }
            print(
                f"  {variant}: {np.nanmean(aucs):.4f} +/- "
                f"{np.nanstd(aucs):.4f}"
            )
        results["tasks"][task_key] = task_results

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
