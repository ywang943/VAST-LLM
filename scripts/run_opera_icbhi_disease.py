"""
Run OPERA-aligned ICBHI disease evaluation.

Protocol mirrored from OPERA's linear_evaluation_icbhidisease:
  - Use OPERA's official ICBHI train/test split.
  - Keep only Healthy vs COPD.
  - Split OPERA train into train/val with random_state=1337, stratified.
  - Select best checkpoint by validation AUROC, then report fixed test metrics.

The classifier can use either continuous z_cont representations or VQ z_q.
For OPERA comparison, continuous mode is the closer protocol.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from data.respvoice_datasets import CachedMelDataset
from respvoice.config import ModelConfig, RespVoiceConfig, TrainConfig
from respvoice.downstream import DownstreamHead, LinearProbe
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer
from scripts.run_full_local import class_weights_from_labels, evaluate_binary, labels_from_subset


def official_split(dataset):
    trainval, test = [], []
    for idx, sample in enumerate(dataset.samples):
        if sample["split"] == "test":
            test.append(idx)
        else:
            trainval.append(idx)

    labels = [int(dataset.samples[i]["label"]) for i in trainval]
    train_idx, val_idx = train_test_split(
        trainval,
        test_size=0.2,
        random_state=1337,
        stratify=labels,
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test)


def build_model(args, cfg):
    model = RespVoiceModel(cfg.model)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        raw = state["model_state"]
        current = model.state_dict()
        # Filter: skip keys with shape mismatch (e.g., VQ codebook with different K)
        compatible = {
            k: v for k, v in raw.items()
            if k in current and current[k].shape == v.shape
        }
        skipped = [k for k in raw if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        if skipped:
            print(f"[build_model] skipped {len(skipped)} shape-incompatible keys: "
                  f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    return model


def train_stage3_auc(
    cfg,
    model,
    train_loader,
    val_loader,
    n_classes,
    device,
    use_quantized,
    linear_probe,
    class_weights,
):
    from respvoice.downstream import LinearProbe

    head = LinearProbe(cfg.model.D, n_classes) if linear_probe else DownstreamHead(
        cfg.model.D, n_classes, use_regression=False
    )
    model.set_downstream_head(head)
    model.to(device)

    if linear_probe:
        model.freeze_encoder()
        model.freeze_vq()
        params = list(model.head.parameters())
    else:
        params = list(model.parameters())

    opt = AdamW(params, lr=cfg.train.stage3_lr, weight_decay=cfg.train.weight_decay)
    best_auc = -1.0
    best_path = Path(cfg.checkpoint_dir) / "stage3_best_auc.pt"
    final_path = Path(cfg.checkpoint_dir) / "stage3_final_auc.pt"
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.log_dir) / "train_log.jsonl"
    class_weights = class_weights.to(device) if class_weights is not None else None

    for epoch in range(1, cfg.train.stage3_epochs + 1):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for batch in train_loader:
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad()
            out = model.forward_stage3(mel, use_quantized=use_quantized)
            loss = F.cross_entropy(out["logits"], labels, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            loss_sum += loss.item() * labels.numel()
            correct += (out["logits"].argmax(1) == labels).sum().item()
            n += labels.numel()

        val = evaluate_binary(model, val_loader, device, use_quantized=use_quantized)
        val_auc = -1.0 if val["auroc"] is None else float(val["auroc"])
        print(
            f"  [S3-AUC Ep{epoch}] loss={loss_sum/max(1,n):.4f} "
            f"acc={correct/max(1,n):.3f} val_auc={val_auc:.4f} val_acc={val['accuracy']:.3f}"
        )
        # Write epoch log for monitoring
        import json as _json
        with open(log_path, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps({
                "stage": "stage3", "epoch": epoch,
                "train_loss": round(loss_sum / max(1, n), 4),
                "train_acc": round(correct / max(1, n), 4),
                "val_auroc": round(val_auc, 4),
                "val_acc": round(float(val["accuracy"]), 4),
            }) + "\n")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)

    torch.save({"model_state": model.state_dict(), "config": cfg}, final_path)
    return {"best_auc": best_auc, "best_path": str(best_path), "final_path": str(final_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-cache", default=None, help="If set, run Stage 1/2 before Stage 3")
    p.add_argument("--label-cache", default="./data/mel_cache/opera_icbhi_disease")
    p.add_argument("--init-checkpoint", default=None, help="Optional Stage 2 checkpoint to initialize from")
    p.add_argument("--checkpoint-dir", default="./checkpoints/opera_icbhi_disease")
    p.add_argument("--log-dir", default="./logs/opera_icbhi_disease")
    p.add_argument("--epochs-stage1", type=int, default=5)
    p.add_argument("--epochs-stage2", type=int, default=5)
    p.add_argument("--epochs-stage3", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--encoder-heads", type=int, default=4)
    p.add_argument("--predictor-layers", type=int, default=1)
    p.add_argument("--sigreg-slices", type=int, default=32)
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--continuous", action="store_true", help="Use z_cont for downstream")
    p.add_argument("--linear-probe", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-l2-norm", action="store_true", help="Disable VQ L2 normalization (ablation)")
    p.add_argument("--no-ema", action="store_true", help="Disable EMA codebook update (ablation)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = RespVoiceConfig(
        model=ModelConfig(
            D=args.dim,
            codebook_size=args.codebook_size,
            encoder_layers=args.encoder_layers,
            encoder_heads=args.encoder_heads,
            predictor_layers=args.predictor_layers,
            n_sigreg_slices=args.sigreg_slices,
            backbone="custom",
            vq_use_ema=not args.no_ema,
            vq_l2_normalize=not args.no_l2_norm,
        ),
        train=TrainConfig(
            stage1_epochs=args.epochs_stage1,
            stage2_epochs=args.epochs_stage2,
            stage3_epochs=args.epochs_stage3,
            batch_size=args.batch_size,
            stage1_lr=3e-4,
            stage2_lr=1e-4,
            stage3_lr=3e-4 if not args.linear_probe else 1e-3,
            lam_sig=0.01,
            lam_recon=0.05,
            warmup_ratio=0.1,
            num_workers=0,
        ),
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    label_cache = Path(args.label_cache)
    label_ds = CachedMelDataset(
        root=str(label_cache),
        meta_file=str(label_cache / "metadata.json"),
        include_labels=True,
    )
    train_ds, val_ds, test_ds = official_split(label_ds)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    weights = None if args.no_class_weights else class_weights_from_labels(labels_from_subset(train_ds), 2)

    model = build_model(args, cfg)
    trainer = Trainer(cfg, model)

    if args.pretrain_cache:
        pretrain_cache = Path(args.pretrain_cache)
        pretrain_ds = CachedMelDataset(
            root=str(pretrain_cache),
            meta_file=str(pretrain_cache / "metadata.json"),
            include_labels=False,
        )
        pretrain_loader = DataLoader(
            pretrain_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            generator=train_generator,
        )
        trainer.train_stage1(pretrain_loader)
        trainer.train_stage2(pretrain_loader)

    use_quantized = not args.continuous
    stage3 = train_stage3_auc(
        cfg,
        model,
        train_loader,
        val_loader,
        n_classes=2,
        device=trainer.device,
        use_quantized=use_quantized,
        linear_probe=args.linear_probe,
        class_weights=weights,
    )

    summary = {
        "protocol": "OPERA icbhidisease Healthy-vs-COPD",
        "downstream_representation": "z_cont" if args.continuous else "z_q",
        "linear_probe": args.linear_probe,
        "seed": args.seed,
        "model_dim": args.dim,
        "encoder_layers": args.encoder_layers,
        "encoder_heads": args.encoder_heads,
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
        "class_weights": None if weights is None else weights.tolist(),
        "stage3_selection": "valid_auc",
        "stage3": stage3,
        "checkpoints": {},
    }

    for name in ("stage3_best_auc.pt", "stage3_final_auc.pt"):
        ckpt = torch.load(Path(args.checkpoint_dir) / name, map_location="cpu", weights_only=False)
        m = RespVoiceModel(ckpt["config"].model)
        if args.linear_probe:
            m.set_downstream_head(LinearProbe(ckpt["config"].model.D, n_classes=2))
        else:
            m.set_downstream_head(DownstreamHead(ckpt["config"].model.D, n_classes=2, use_regression=False))
        m.load_state_dict(ckpt["model_state"], strict=False)
        m.to(trainer.device)
        summary["checkpoints"][name] = {
            "val": evaluate_binary(m, val_loader, trainer.device, use_quantized=use_quantized),
            "test": evaluate_binary(m, test_loader, trainer.device, use_quantized=use_quantized),
        }

    out = Path(args.checkpoint_dir) / "opera_official_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
