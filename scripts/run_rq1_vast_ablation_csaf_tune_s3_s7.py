#!/usr/bin/env python3
"""RQ1 VAST CSAF ablations on S3/S7 with task-level CSAF tuning.

Protocol:
  * all variants initialize from the same VAST checkpoint;
  * each target task is tuned separately using train split only;
  * HTS-AT backbone is frozen;
  * full/no-pos tune CSAF + a task head, no-CSAF uses frozen stage-4 only;
  * best checkpoint is selected by validation AUROC;
  * final AUROC uses the same frozen-encoder LP protocol as Table 2.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_rq1_mel_linear_probe as rq1  # noqa: E402
from respvoice.htsat_encoder import build_htsat_encoder  # noqa: E402


TASKS = {
    "S3_coswara_covid_exhale": rq1.S_TASKS["S3_coswara_covid_exhale"],
    "S7_b2ai": rq1.S_TASKS["S7_b2ai"],
}

VARIANTS = ["full", "no_csaf_stage4", "no_pos_encoding"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    n_tables = 0
    for module in encoder.modules():
        table = getattr(module, "relative_position_bias_table", None)
        if table is None:
            continue
        with torch.no_grad():
            table.zero_()
        if freeze:
            table.requires_grad_(False)
        n_tables += 1
    return n_tables


def build_variant_encoder(variant, init_ckpt, device):
    encoder = build_htsat_encoder(
        ckpt_path=None,
        freeze_backbone=False,
        use_csaf=(variant != "no_csaf_stage4"),
    )
    missing, unexpected = encoder.load_state_dict(encoder_state_from_ckpt(init_ckpt), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{variant} checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if variant == "no_pos_encoding":
        disable_position_encoding(encoder, freeze=True)

    for p in encoder.parameters():
        p.requires_grad_(False)
    if variant in ("full", "no_pos_encoding"):
        for name, p in encoder.named_parameters():
            if name.startswith("csaf."):
                p.requires_grad_(True)

    return encoder.to(device)


class SplitMelDataset(Dataset):
    def __init__(self, task_cfg, split):
        self.root = ROOT / task_cfg["mel_root"]
        raw = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        samples = raw.get("samples", raw if isinstance(raw, list) else [])
        self.samples = [
            s for s in samples
            if s.get("split", "train") == split
            and "label" in s
            and (self.root / s["path"]).exists()
        ]
        labels = [int(s["label"]) for s in self.samples]
        self.n_classes = max(labels) + 1
        counts = Counter(labels)
        self.weights = [1.0 / counts[int(s["label"])] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "mel": torch.load(str(self.root / s["path"]), map_location="cpu"),
            "label": int(s["label"]),
        }


def collate(batch):
    return {
        "mel": torch.stack([b["mel"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def auroc_from_logits(y_true, logits):
    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    if len(np.unique(y_true)) == 2:
        return float(roc_auc_score(y_true, probs[:, 1]))
    return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))


@torch.no_grad()
def validate(encoder, head, loader, device, variant):
    encoder.eval()
    head.eval()
    ys, logits = [], []
    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        feat = encoder(mel).mean(dim=1)
        logits.append(head(feat).cpu())
        ys.append(batch["label"].numpy())
        if variant == "no_pos_encoding":
            disable_position_encoding(encoder, freeze=True)
    return auroc_from_logits(np.concatenate(ys), torch.cat(logits, dim=0))


def train_one_task(variant, task_key, task_cfg, init_ckpt, work_dir, args, device):
    train_ds = SplitMelDataset(task_cfg, "train")
    val_ds = SplitMelDataset(task_cfg, "val")
    sampler = WeightedRandomSampler(train_ds.weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    encoder = build_variant_encoder(variant, init_ckpt, device)
    head = nn.Linear(768, train_ds.n_classes).to(device)
    params = [p for p in list(encoder.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    best_path = work_dir / f"{variant}_{task_key}_best.pt"
    work_dir.mkdir(parents=True, exist_ok=True)

    trainable_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  {variant} {task_key}: trainable_encoder={trainable_encoder:,}")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        total, n_total = 0.0, 0
        for batch in train_loader:
            mel = batch["mel"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            feat = encoder(mel).mean(dim=1)
            loss = nn.functional.cross_entropy(head(feat), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if variant == "no_pos_encoding":
                disable_position_encoding(encoder, freeze=True)
            total += float(loss.item()) * mel.size(0)
            n_total += mel.size(0)
        val_auc = validate(encoder, head, val_loader, device, variant)
        train_loss = total / max(1, n_total)
        print(f"    ep{epoch:02d} loss={train_loss:.4f} val_auroc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_state": {f"encoder.{k}": v.cpu() for k, v in encoder.state_dict().items()},
                    "head": head.state_dict(),
                    "meta": {
                        "variant": variant,
                        "task_key": task_key,
                        "init_ckpt": str(init_ckpt),
                        "best_val_auroc": best_auc,
                        "trainable_encoder_params": trainable_encoder,
                    },
                },
                best_path,
            )
    del encoder, head
    torch.cuda.empty_cache()
    return best_path, best_auc


@torch.no_grad()
def evaluate_lp(variant, task_key, task_cfg, ckpt_path, args, device):
    encoder = rq1.load_vast_like_encoder(
        ckpt_path,
        device,
        use_csaf=(variant != "no_csaf_stage4"),
    )
    if variant == "no_pos_encoding":
        disable_position_encoding(encoder, freeze=True)
    data = rq1.extract_task_features(
        encoder,
        task_cfg,
        device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pool_names=["mean"],
    )
    res = rq1.fit_eval_probe(data["features"]["mean"], data["labels"], data["splits"])
    del encoder
    torch.cuda.empty_cache()
    print(f"    LP {variant} {task_key}: AUROC={res['auroc']:.4f} C={res['best_c']}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ckpt", default="checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt")
    parser.add_argument("--output", default="checkpoints/ablations/rq1_vast_s3_s7_csaf_tuned_ablations.json")
    parser.add_argument("--work-dir", default="checkpoints/ablations/rq1_vast_s3_s7_csaf_tune_ckpts")
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
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
            "protocol": "per-task train-only tuning; frozen HTS-AT; CSAF/head trainable where applicable; best by val AUROC; final Table-2 LP",
            "init_ckpt": str(init_ckpt),
            "c_grid": rq1.C_GRID,
            "epochs": args.epochs,
            "lr": args.lr,
        }
    }
    for variant in args.variants:
        results[variant] = {}
        print(f"\n{'=' * 78}\n{variant}\n{'=' * 78}")
        for task_key, task_cfg in TASKS.items():
            best_ckpt, best_val = train_one_task(variant, task_key, task_cfg, init_ckpt, work_dir, args, device)
            lp = evaluate_lp(variant, task_key, task_cfg, best_ckpt, args, device)
            results[variant][task_key] = {
                "mean": lp,
                "best_val_auroc": best_val,
                "best_ckpt": str(best_ckpt),
            }

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = [
        "| Variant | S3 Coswara Covid Exhale | S7 Bridge2AI | Avg |",
        "| --- | --- | --- | --- |",
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
