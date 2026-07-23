#!/usr/bin/env python3
"""Collective train-split adaptation for the VAST encoder before RQ1 LP.

This trains one shared VAST encoder with task-specific linear heads on the
train/val splits of S1-S7 only. The saved checkpoint keeps the usual
``encoder.*`` keys so it can be evaluated by run_rq1_mel_linear_probe.py as a
frozen encoder with fresh logistic probes.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from respvoice.htsat_encoder import build_htsat_encoder  # noqa: E402
from scripts.run_rq1_mel_linear_probe import S_TASKS  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RQ1TrainDataset(Dataset):
    def __init__(self, tasks):
        self.items = []
        self.task_classes = {}
        for task_key, cfg in tasks.items():
            root = ROOT / cfg["mel_root"]
            raw = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            samples = raw.get("samples", raw if isinstance(raw, list) else [])
            labels = sorted({int(s["label"]) for s in samples if "label" in s})
            self.task_classes[task_key] = max(labels) + 1
            for s in samples:
                if s.get("split", "train") not in ("train", "val") or "label" not in s:
                    continue
                path = root / s["path"]
                if path.exists():
                    self.items.append({
                        "mel_path": path,
                        "task_key": task_key,
                        "label": int(s["label"]),
                    })

        counts = Counter((x["task_key"], x["label"]) for x in self.items)
        self.weights = [1.0 / counts[(x["task_key"], x["label"])] for x in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "mel": torch.load(str(item["mel_path"]), map_location="cpu"),
            "task_key": item["task_key"],
            "label": item["label"],
        }


def collate(batch):
    return {
        "mel": torch.stack([b["mel"] for b in batch]),
        "task_key": [b["task_key"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


class MultiTaskAdapter(nn.Module):
    def __init__(self, encoder, task_classes):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({
            task_key: nn.Linear(768, n_cls) for task_key, n_cls in task_classes.items()
        })

    def forward(self, mel):
        z = self.encoder(mel)
        return z.mean(dim=1)


def encoder_state_from_ckpt(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }


def disable_position_encoding(encoder, freeze=True):
    if hasattr(encoder.htsat, "ape"):
        encoder.htsat.ape = False
    if hasattr(encoder.htsat, "absolute_pos_embed"):
        with torch.no_grad():
            encoder.htsat.absolute_pos_embed.zero_()
        if freeze:
            encoder.htsat.absolute_pos_embed.requires_grad_(False)

    zeroed = 0
    for module in encoder.modules():
        table = getattr(module, "relative_position_bias_table", None)
        if table is None:
            continue
        with torch.no_grad():
            table.zero_()
        if freeze:
            table.requires_grad_(False)
        zeroed += 1
    return zeroed


def load_trainable_encoder(ckpt_path, device, variant):
    use_csaf = variant != "no_csaf_stage4"
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=False, use_csaf=use_csaf)
    missing, unexpected = encoder.load_state_dict(encoder_state_from_ckpt(ROOT / ckpt_path), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{variant} checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if variant == "no_pos_encoding":
        disable_position_encoding(encoder, freeze=True)
    return encoder.to(device).train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ckpt", default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt")
    parser.add_argument("--output-dir", default="checkpoints/htsat_lejepa_v3_rq1_collective")
    parser.add_argument(
        "--variant",
        default="full",
        choices=["full", "no_csaf_stage4", "no_pos_encoding"],
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RQ1TrainDataset(S_TASKS)
    sampler = WeightedRandomSampler(ds.weights, num_samples=len(ds), replacement=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    encoder = load_trainable_encoder(args.init_ckpt, device, args.variant)
    model = MultiTaskAdapter(encoder, ds.task_classes).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.heads.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    meta = {
        "init_ckpt": str(ROOT / args.init_ckpt),
        "task_classes": ds.task_classes,
        "n_train": len(ds),
        "protocol": "collective supervised adaptation on RQ1 train/val splits only; evaluation uses fresh frozen linear probes",
        "variant": args.variant,
        "use_csaf": args.variant != "no_csaf_stage4",
        "position_encoding": (
            "absolute disabled; relative bias zeroed/frozen"
            if args.variant == "no_pos_encoding" else "enabled"
        ),
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        pbar = tqdm(loader, desc=f"Ep{epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            mel = batch["mel"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            feats = model(mel)
            loss = feats.new_tensor(0.0)
            used = 0
            task_to_idx = defaultdict(list)
            for i, task_key in enumerate(batch["task_key"]):
                task_to_idx[task_key].append(i)
            for task_key, idxs in task_to_idx.items():
                idx = torch.tensor(idxs, device=device)
                logits = model.heads[task_key](feats.index_select(0, idx))
                loss = loss + nn.functional.cross_entropy(logits, y.index_select(0, idx))
                used += 1
            loss = loss / max(1, used)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if args.variant == "no_pos_encoding":
                disable_position_encoding(model.encoder, freeze=True)
            n = mel.size(0)
            total_loss += float(loss.item()) * n
            total_n += n
            pbar.set_postfix(loss=f"{total_loss / max(1, total_n):.4f}")

        avg_loss = total_loss / max(1, total_n)
        print(f"[Ep{epoch}] loss={avg_loss:.4f}")
        state = {
            "model_state": {f"encoder.{k}": v.cpu() for k, v in model.encoder.state_dict().items()},
            "heads": {k: v.cpu() for k, v in model.heads.state_dict().items()},
            "epoch": epoch,
            "loss": avg_loss,
            "meta": meta,
        }
        torch.save(state, out_dir / f"htsat_lejepa_ep{epoch}.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, out_dir / "htsat_lejepa_best.pt")
            print(f"  -> saved best loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
