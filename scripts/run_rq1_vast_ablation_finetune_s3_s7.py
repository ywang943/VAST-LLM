#!/usr/bin/env python3
"""Fine-tuned VAST ablations on S3 and S7.

This is the fair architecture-ablation protocol:
  1. Initialize every variant from the same VAST checkpoint.
  2. Fine-tune the variant on S3/S7 train+val with task-specific linear heads.
  3. Freeze the fine-tuned encoder and evaluate the same RQ1 LP protocol:
     mean pooling + standardized logistic regression + train-only C selection.

Variants:
  full             : HTS-AT + CSAF fusion.
  no_csaf_stage4   : disables CSAF and returns stage-4 features only.
  no_pos_encoding  : disables absolute position embedding and zeros/freezes
                     relative-position bias tables during fine-tuning/eval.
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

import scripts.run_rq1_mel_linear_probe as rq1  # noqa: E402
from respvoice.htsat_encoder import build_htsat_encoder  # noqa: E402


TASKS = {
    "S3_coswara_covid_exhale": rq1.S_TASKS["S3_coswara_covid_exhale"],
    "S7_b2ai": rq1.S_TASKS["S7_b2ai"],
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_position_encoding(encoder, freeze=True):
    if hasattr(encoder.htsat, "ape"):
        encoder.htsat.ape = False
    if hasattr(encoder.htsat, "absolute_pos_embed"):
        with torch.no_grad():
            encoder.htsat.absolute_pos_embed.zero_()
        if freeze and hasattr(encoder.htsat.absolute_pos_embed, "requires_grad"):
            encoder.htsat.absolute_pos_embed.requires_grad_(False)

    zeroed = 0
    for module in encoder.modules():
        table = getattr(module, "relative_position_bias_table", None)
        if table is not None:
            with torch.no_grad():
                table.zero_()
            if freeze:
                table.requires_grad_(False)
            zeroed += 1
    return zeroed


def encoder_state_from_ckpt(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    return {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }


def build_variant_encoder(variant, init_ckpt, device):
    use_csaf = variant != "no_csaf_stage4"
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=False, use_csaf=use_csaf)
    missing, unexpected = encoder.load_state_dict(encoder_state_from_ckpt(init_ckpt), strict=False)
    if unexpected:
        raise RuntimeError(f"{variant}: unexpected keys while loading: {unexpected[:8]}")
    if variant != "no_csaf_stage4" and missing:
        raise RuntimeError(f"{variant}: missing keys while loading: {missing[:8]}")
    meta = {"use_csaf": use_csaf, "position_encoding": "enabled"}
    if variant == "no_pos_encoding":
        meta["relative_position_bias_tables_zeroed"] = disable_position_encoding(encoder, freeze=True)
        meta["position_encoding"] = "absolute disabled; relative bias zeroed/frozen"
    return encoder.to(device), meta


class AblationTrainDataset(Dataset):
    def __init__(self):
        self.items = []
        self.task_classes = {}
        for task_key, cfg in TASKS.items():
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
                        "path": path,
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
            "mel": torch.load(str(item["path"]), map_location="cpu"),
            "task_key": item["task_key"],
            "label": item["label"],
        }


def collate(batch):
    return {
        "mel": torch.stack([b["mel"] for b in batch]),
        "task_key": [b["task_key"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


class MultiTaskHeads(nn.Module):
    def __init__(self, encoder, task_classes):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({
            task_key: nn.Linear(768, n_cls)
            for task_key, n_cls in task_classes.items()
        })

    def forward_features(self, mel):
        return self.encoder(mel).mean(dim=1)


def train_variant(variant, init_ckpt, out_dir, args, device):
    ds = AblationTrainDataset()
    sampler = WeightedRandomSampler(ds.weights, num_samples=len(ds), replacement=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    encoder, variant_meta = build_variant_encoder(variant, init_ckpt, device)
    model = MultiTaskHeads(encoder, ds.task_classes).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.heads.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    best = float("inf")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "variant": variant,
        "init_ckpt": str(init_ckpt),
        "task_classes": ds.task_classes,
        "n_train_val": len(ds),
        **variant_meta,
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n_total = 0.0, 0
        pbar = tqdm(loader, desc=f"{variant} Ep{epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            mel = batch["mel"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            feat = model.forward_features(mel)
            task_to_idx = defaultdict(list)
            for i, task_key in enumerate(batch["task_key"]):
                task_to_idx[task_key].append(i)
            loss = feat.new_tensor(0.0)
            for task_key, idxs in task_to_idx.items():
                idx = torch.tensor(idxs, device=device)
                logits = model.heads[task_key](feat.index_select(0, idx))
                loss = loss + nn.functional.cross_entropy(logits, y.index_select(0, idx))
            loss = loss / max(1, len(task_to_idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if variant == "no_pos_encoding":
                disable_position_encoding(model.encoder, freeze=True)
            total += float(loss.item()) * mel.size(0)
            n_total += mel.size(0)
            pbar.set_postfix(loss=f"{total / max(1, n_total):.4f}")
        avg = total / max(1, n_total)
        print(f"  [{variant} Ep{epoch}] loss={avg:.4f}")
        state = {
            "model_state": {f"encoder.{k}": v.cpu() for k, v in model.encoder.state_dict().items()},
            "heads": {k: v.cpu() for k, v in model.heads.state_dict().items()},
            "epoch": epoch,
            "loss": avg,
            "meta": meta,
        }
        torch.save(state, out_dir / f"{variant}_ep{epoch}.pt")
        if avg < best:
            best = avg
            torch.save(state, out_dir / f"{variant}_best.pt")
            print(f"    -> saved best {best:.4f}")
    del model
    torch.cuda.empty_cache()
    return out_dir / f"{variant}_best.pt", meta


@torch.no_grad()
def evaluate_variant(variant, ckpt_path, args, device):
    use_csaf = variant != "no_csaf_stage4"
    encoder = rq1.load_vast_like_encoder(ckpt_path, device, use_csaf=use_csaf)
    if variant == "no_pos_encoding":
        disable_position_encoding(encoder, freeze=True)
    row = {}
    for task_key, cfg in TASKS.items():
        print(f"  Eval {variant} {task_key}")
        data = rq1.extract_task_features(
            encoder,
            cfg,
            device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pool_names=["mean"],
        )
        row[task_key] = {
            "mean": rq1.fit_eval_probe(data["features"]["mean"], data["labels"], data["splits"])
        }
        res = row[task_key]["mean"]
        print(f"    AUROC={res['auroc']:.4f} C={res['best_c']} n={res['n_train']}/{res['n_test']}")
    del encoder
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ckpt", default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt")
    parser.add_argument("--output", default="checkpoints/ablations/rq1_vast_s3_s7_finetuned_ablations.json")
    parser.add_argument("--work-dir", default="checkpoints/ablations/rq1_vast_s3_s7_finetune_ckpts")
    parser.add_argument("--variants", nargs="+", default=["full", "no_csaf_stage4", "no_pos_encoding"],
                        choices=["full", "no_csaf_stage4", "no_pos_encoding"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--c-grid", default="0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    rq1.C_GRID = [float(x) for x in args.c_grid.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_ckpt = ROOT / args.init_ckpt
    work_dir = ROOT / args.work_dir

    results = {
        "_meta": {
            "init_ckpt": str(init_ckpt),
            "tasks": list(TASKS.keys()),
            "protocol": "variant fine-tuned on S3/S7 train+val, then frozen mean-pooled logistic LP",
            "c_grid": rq1.C_GRID,
            "epochs": args.epochs,
            "lr": args.lr,
            "head_lr": args.head_lr,
        }
    }
    for variant in args.variants:
        print(f"\n{'=' * 78}\nTRAIN {variant}\n{'=' * 78}")
        best_ckpt, meta = train_variant(variant, init_ckpt, work_dir, args, device)
        print(f"\n{'=' * 78}\nEVAL {variant}\n{'=' * 78}")
        row = evaluate_variant(variant, best_ckpt, args, device)
        row["_meta"] = {**meta, "best_ckpt": str(best_ckpt)}
        results[variant] = row

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    headers = ["Variant", "S3 Covid Exhale", "S7 Bridge2AI", "Avg"]
    md = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for variant in args.variants:
        vals = [results[variant][task]["mean"]["auroc"] for task in TASKS]
        md.append("| " + " | ".join([variant, *[f"{v:.4f}" for v in vals], f"{float(np.mean(vals)):.4f}"]) + " |")
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved: {out}")
    print(f"Saved: {md_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
