"""
Run the current complete local RespVoice pipeline.

Pretraining data:
  - ICBHI respiratory sounds
  - CoughVID cough sounds
  - Coswara breathing/cough/vowels/counting

Supervised task:
  - ICBHI binary classification: normal vs disease

Assumes mel caches were created by data/prepare_mel_cache.py.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from data.respvoice_datasets import CachedMelDataset
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer


def stratified_split(dataset, val_ratio=0.2, seed=2026):
    labels = [int(x["label"]) for x in dataset.samples]
    by_label = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)

    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []
    for idxs in by_label.values():
        perm = torch.tensor(idxs)[torch.randperm(len(idxs), generator=gen)].tolist()
        n_val = max(1, int(len(perm) * val_ratio))
        val_idx.extend(perm[:n_val])
        train_idx.extend(perm[n_val:])
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def labels_from_subset(subset):
    return [int(subset.dataset.samples[i]["label"]) for i in subset.indices]


def class_weights_from_labels(labels, n_classes):
    counts = torch.bincount(torch.tensor(labels), minlength=n_classes).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * n_classes)
    return weights


@torch.no_grad()
def evaluate_binary(model, loader, device, use_quantized=True):
    model.eval()
    all_pred, all_label = [], []
    all_prob = []
    loss_sum, n = 0.0, 0
    for batch in loader:
        mel = batch["mel"].to(device)
        labels = batch["label"].to(device)
        out = model.forward_stage3(mel, use_quantized=use_quantized)
        loss = F.cross_entropy(out["logits"], labels)
        pred = out["logits"].argmax(1)
        prob = out["logits"].softmax(1)[:, 1]
        all_pred.extend(pred.cpu().tolist())
        all_label.extend(labels.cpu().tolist())
        all_prob.extend(prob.cpu().tolist())
        loss_sum += loss.item() * labels.numel()
        n += labels.numel()

    cm = [[0, 0], [0, 0]]
    for y, p in zip(all_label, all_pred):
        cm[y][p] += 1
    per_class_recall = []
    for c in range(2):
        denom = sum(cm[c])
        per_class_recall.append(cm[c][c] / denom if denom else 0.0)
    acc = sum(cm[c][c] for c in range(2)) / max(1, sum(sum(r) for r in cm))
    bal_acc = sum(per_class_recall) / 2
    try:
        auroc = roc_auc_score(all_label, all_prob)
    except ValueError:
        auroc = None
    return {
        "loss": loss_sum / max(1, n),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auroc": auroc,
        "confusion_matrix": cm,
        "recall_normal": per_class_recall[0],
        "recall_disease": per_class_recall[1],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-cache", default="./data/mel_cache/pretrain")
    p.add_argument("--label-cache", default="./data/mel_cache/icbhi_binary")
    p.add_argument("--checkpoint-dir", default="./checkpoints/full_local")
    p.add_argument("--log-dir", default="./logs/full_local")
    p.add_argument("--epochs-stage1", type=int, default=3)
    p.add_argument("--epochs-stage2", type=int, default=3)
    p.add_argument("--epochs-stage3", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--fine-tune", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = RespVoiceConfig(
        model=ModelConfig(
            D=args.dim,
            codebook_size=args.codebook_size,
            encoder_layers=2,
            encoder_heads=4,
            predictor_layers=1,
            n_sigreg_slices=32,
            backbone="custom",
        ),
        train=TrainConfig(
            stage1_epochs=args.epochs_stage1,
            stage2_epochs=args.epochs_stage2,
            stage3_epochs=args.epochs_stage3,
            batch_size=args.batch_size,
            stage1_lr=3e-4,
            stage2_lr=1e-4,
            stage3_lr=5e-4 if args.fine_tune else 1e-3,
            lam_sig=0.01,
            lam_recon=0.05,
            warmup_ratio=0.1,
            num_workers=0,
        ),
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    pretrain_ds = CachedMelDataset(
        root=args.pretrain_cache,
        meta_file=str(Path(args.pretrain_cache) / "metadata.json"),
        include_labels=False,
    )
    label_ds = CachedMelDataset(
        root=args.label_cache,
        meta_file=str(Path(args.label_cache) / "metadata.json"),
        include_labels=True,
    )
    train_ds, val_ds = stratified_split(label_ds, val_ratio=0.2)
    class_weights = class_weights_from_labels(labels_from_subset(train_ds), n_classes=2)

    pretrain_loader = DataLoader(
        pretrain_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RespVoiceModel(cfg.model)
    trainer = Trainer(cfg, model)

    print("=== RespVoice full local run ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"pretrain windows: {len(pretrain_ds)}")
    print(f"supervised train/val windows: {len(train_ds)}/{len(val_ds)}")
    print(f"mode: {'fine-tune' if args.fine_tune else 'linear-probe'}")
    print(f"class weights: {class_weights.tolist()}")

    trainer.train_stage1(pretrain_loader)
    trainer.train_stage2(pretrain_loader)
    trainer.train_stage3(
        train_loader,
        val_loader=val_loader,
        n_classes=2,
        linear_probe=not args.fine_tune,
        use_quantized=True,
        class_weights=class_weights,
    )

    model.eval()
    eval_metrics = evaluate_binary(model, val_loader, trainer.device, use_quantized=True)
    preview = next(iter(val_loader))
    batch = preview["mel"].to(trainer.device)
    labels = preview["label"]
    with torch.no_grad():
        out = model.forward_stage3(batch, use_quantized=True)
        pred = out["logits"].argmax(1).cpu()
        ids = model.encode_to_tokens(batch[:2])

    summary = {
        "pretrain_windows": len(pretrain_ds),
        "supervised_train_windows": len(train_ds),
        "supervised_val_windows": len(val_ds),
        "mode": "fine-tune" if args.fine_tune else "linear-probe",
        "token_shape": list(ids.shape),
        "token_min": int(ids.min().item()),
        "token_max": int(ids.max().item()),
        "val_preview_labels": labels[:16].tolist(),
        "val_preview_pred": pred[:16].tolist(),
        "eval": eval_metrics,
    }
    out_path = Path(args.checkpoint_dir) / "run_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
