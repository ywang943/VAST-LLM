"""
Run Stage 3 fine-tuning with continuous encoder representations (z_cont).

This is useful for OPERA-style downstream comparisons, where the benchmark
evaluates continuous acoustic representations rather than VQ token IDs.

Example:
  python scripts/run_stage3_continuous.py --checkpoint checkpoints/opera_t7_zenodo_ft/stage2_final.pt --label-cache data/mel_cache/icbhi_copd_healthy --checkpoint-dir checkpoints/opera_t7_zenodo_cont_ft
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data.respvoice_datasets import CachedMelDataset
from respvoice.downstream import DownstreamHead
from respvoice.model import RespVoiceModel
from respvoice.trainer import Trainer
from scripts.run_full_local import (
    class_weights_from_labels,
    evaluate_binary,
    labels_from_subset,
    stratified_split,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Stage 2 checkpoint path")
    p.add_argument("--label-cache", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    cfg.train.stage3_epochs = args.epochs
    cfg.train.stage3_lr = args.lr
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.log_dir = args.log_dir or str(Path(args.checkpoint_dir).parent.parent / "logs" / Path(args.checkpoint_dir).name)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    label_cache = Path(args.label_cache)
    label_ds = CachedMelDataset(
        root=str(label_cache),
        meta_file=str(label_cache / "metadata.json"),
        include_labels=True,
    )
    train_ds, val_ds = stratified_split(label_ds, val_ratio=0.2)
    weights = class_weights_from_labels(labels_from_subset(train_ds), n_classes=2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RespVoiceModel(cfg.model)
    model.load_state_dict(state["model_state"], strict=False)
    trainer = Trainer(cfg, model)
    trainer.train_stage3(
        train_loader,
        val_loader,
        n_classes=2,
        linear_probe=False,
        use_quantized=False,
        class_weights=weights,
    )

    summary = {}
    for name in ("stage3_best.pt", "stage3_final.pt"):
        ckpt = torch.load(Path(cfg.checkpoint_dir) / name, map_location="cpu", weights_only=False)
        m = RespVoiceModel(ckpt["config"].model)
        m.set_downstream_head(DownstreamHead(ckpt["config"].model.D, n_classes=2, use_regression=False))
        m.load_state_dict(ckpt["model_state"], strict=False)
        m.to(trainer.device)
        summary[name] = evaluate_binary(m, val_loader, trainer.device, use_quantized=False)

    with open(Path(cfg.checkpoint_dir) / "continuous_eval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
