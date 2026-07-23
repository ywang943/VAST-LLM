"""
Multi-stage trainer for RespVoice.

Stage 1: LeJEPA self-supervised pretraining
Stage 2: VQ tokenizer training (encoder frozen)
Stage 3: Downstream fine-tuning (linear probe or full fine-tune)
"""

import os
import json
import time
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from .config import RespVoiceConfig
from .model import RespVoiceModel
from .downstream import DownstreamHead, downstream_loss, LinearProbe


class Trainer:
    def __init__(self, cfg: RespVoiceConfig, model: RespVoiceModel):
        self.cfg = cfg
        self.model = model
        self.device = self._get_device()
        self.model.to(self.device)
        self.log = []

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[Trainer] No GPU found — running on CPU. Training will be slow.")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Stage 1: LeJEPA pretraining
    # ------------------------------------------------------------------

    def train_stage1(self, dataloader: DataLoader):
        print("\n=== Stage 1: LeJEPA Pretraining ===")
        cfg_t = self.cfg.train
        cfg_m = self.cfg.model

        # All encoder + predictor + sigreg parameters
        params = (
            list(self.model.encoder.parameters())
            + list(self.model.predictor.parameters())
        )
        opt = AdamW(params, lr=cfg_t.stage1_lr, weight_decay=cfg_t.weight_decay)
        scheduler = self._make_scheduler(opt, cfg_t.stage1_epochs, len(dataloader),
                                         cfg_t.stage1_lr, cfg_t.warmup_ratio)

        best_loss = float("inf")
        for epoch in range(1, cfg_t.stage1_epochs + 1):
            self.model.train()
            epoch_stats = {"loss": 0, "jepa": 0, "sigreg": 0, "n": 0}

            for batch in tqdm(dataloader, desc=f"S1 Ep{epoch}/{cfg_t.stage1_epochs}",
                               leave=False, disable=len(dataloader) < 5):
                mel = batch["mel"].to(self.device)
                opt.zero_grad()

                out = self.model.forward_stage1(mel, lam_sig=cfg_t.lam_sig)
                out["loss"].backward()
                nn.utils.clip_grad_norm_(params, cfg_t.grad_clip)
                opt.step()
                scheduler.step()

                B = mel.size(0)
                epoch_stats["loss"]   += out["loss"].item() * B
                epoch_stats["jepa"]   += out["loss_jepa"].item() * B
                epoch_stats["sigreg"] += out["loss_sigreg"].item() * B
                epoch_stats["n"]      += B

            n = epoch_stats["n"]
            avg = {k: v / n for k, v in epoch_stats.items() if k != "n"}
            print(f"  [S1 Ep{epoch}] loss={avg['loss']:.4f}  "
                  f"jepa={avg['jepa']:.4f}  sigreg={avg['sigreg']:.4f}")
            self._log("stage1", epoch, avg)

            if avg["loss"] < best_loss:
                best_loss = avg["loss"]
                self._save("stage1_best.pt")

        self._save("stage1_final.pt")
        print(f"Stage 1 done. Best loss: {best_loss:.4f}")

    # ------------------------------------------------------------------
    # Stage 2: VQ tokenizer training
    # ------------------------------------------------------------------

    def train_stage2(self, dataloader: DataLoader):
        print("\n=== Stage 2: VQ Tokenizer Training ===")
        cfg_t = self.cfg.train

        self.model.freeze_encoder()
        params = list(self.model.vq.parameters()) + list(self.model.decoder.parameters())
        opt = AdamW(params, lr=cfg_t.stage2_lr, weight_decay=cfg_t.weight_decay)
        scheduler = self._make_scheduler(opt, cfg_t.stage2_epochs, len(dataloader),
                                         cfg_t.stage2_lr, cfg_t.warmup_ratio)

        best_util = 0.0
        for epoch in range(1, cfg_t.stage2_epochs + 1):
            self.model.train()
            stats = {"loss": 0, "vq": 0, "recon": 0, "util": 0, "perp": 0, "restarted": 0, "n": 0}

            for batch in tqdm(dataloader, desc=f"S2 Ep{epoch}/{cfg_t.stage2_epochs}",
                               leave=False, disable=len(dataloader) < 5):
                mel = batch["mel"].to(self.device)
                opt.zero_grad()

                out = self.model.forward_stage2(mel, lam_recon=cfg_t.lam_recon)
                out["loss"].backward()
                nn.utils.clip_grad_norm_(params, cfg_t.grad_clip)
                opt.step()
                scheduler.step()

                B = mel.size(0)
                stats["loss"]      += out["loss"].item() * B
                stats["vq"]        += out["loss_vq"].item() * B
                stats["recon"]     += out["loss_recon"].item() * B
                stats["util"]      += out["util"] * B
                stats["perp"]      += out["perplexity"] * B
                stats["restarted"] += out.get("n_restarted", 0)
                stats["n"]         += B

            n = stats["n"]
            avg = {k: v / n for k, v in stats.items() if k not in ("n", "restarted")}
            print(f"  [S2 Ep{epoch}] loss={avg['loss']:.4f}  vq={avg['vq']:.4f}  "
                  f"util={avg['util']:.3f}  perp={avg['perp']:.1f}  "
                  f"restarted={stats['restarted']}")
            self._log("stage2", epoch, avg)

            if avg["util"] > best_util:
                best_util = avg["util"]
                self._save("stage2_best.pt")

        self.model.unfreeze_encoder()
        self._save("stage2_final.pt")
        print(f"Stage 2 done. Best codebook utilization: {best_util:.3f}")

    # ------------------------------------------------------------------
    # Stage 3: Downstream fine-tuning
    # ------------------------------------------------------------------

    def train_stage3(self, train_loader: DataLoader,
                     val_loader: Optional[DataLoader],
                     n_classes: int = 2,
                     linear_probe: bool = True,
                     use_quantized: bool = True,
                     class_weights: Optional[torch.Tensor] = None):
        print(f"\n=== Stage 3: Downstream ({'Linear Probe' if linear_probe else 'Fine-tune'}) ===")
        cfg_t = self.cfg.train

        head = LinearProbe(self.cfg.model.D, n_classes) if linear_probe else \
               DownstreamHead(self.cfg.model.D, n_classes, use_regression=False)
        self.model.set_downstream_head(head)
        self.model.head.to(self.device)

        if linear_probe:
            self.model.freeze_encoder()
            self.model.freeze_vq()
            params = list(self.model.head.parameters())
        else:
            params = list(self.model.parameters())

        opt = AdamW(params, lr=cfg_t.stage3_lr, weight_decay=cfg_t.weight_decay)
        scheduler = self._make_scheduler(opt, cfg_t.stage3_epochs, len(train_loader),
                                         cfg_t.stage3_lr, cfg_t.warmup_ratio)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        best_val = float("inf")
        for epoch in range(1, cfg_t.stage3_epochs + 1):
            self.model.train()
            stats = {"loss": 0, "correct": 0, "n": 0}

            for batch in tqdm(train_loader, desc=f"S3 Ep{epoch}/{cfg_t.stage3_epochs}",
                               leave=False, disable=len(train_loader) < 5):
                mel    = batch["mel"].to(self.device)
                labels = batch["label"].to(self.device)
                opt.zero_grad()

                out = self.model.forward_stage3(mel, use_quantized=use_quantized)

                if linear_probe:
                    loss = nn.functional.cross_entropy(out["logits"], labels, weight=class_weights)
                else:
                    if class_weights is None:
                        loss = downstream_loss(out, labels, lam_reg=cfg_t.lam_reg)
                    else:
                        loss = nn.functional.cross_entropy(out["logits"], labels, weight=class_weights)

                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg_t.grad_clip)
                opt.step()
                scheduler.step()

                B = mel.size(0)
                stats["loss"]    += loss.item() * B
                stats["correct"] += (out["logits"].argmax(1) == labels).sum().item()
                stats["n"]       += B

            n = stats["n"]
            acc = stats["correct"] / n
            avg_loss = stats["loss"] / n
            print(f"  [S3 Ep{epoch}] loss={avg_loss:.4f}  acc={acc:.3f}", end="")

            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, use_quantized, class_weights)
                print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}", end="")
                if val_loss < best_val:
                    best_val = val_loss
                    self._save("stage3_best.pt")
            print()

            self._log("stage3", epoch, {"loss": avg_loss, "acc": acc})

        self.model.unfreeze_encoder()
        self._save("stage3_final.pt")

    @torch.no_grad()
    def _validate(self, loader: DataLoader, use_quantized: bool,
                  class_weights: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        self.model.eval()
        loss_sum, correct, n = 0.0, 0, 0
        for batch in loader:
            mel    = batch["mel"].to(self.device)
            labels = batch["label"].to(self.device)
            out    = self.model.forward_stage3(mel, use_quantized=use_quantized)
            weight = class_weights.to(self.device) if class_weights is not None else None
            loss   = nn.functional.cross_entropy(out["logits"], labels, weight=weight)
            B = mel.size(0)
            loss_sum += loss.item() * B
            correct  += (out["logits"].argmax(1) == labels).sum().item()
            n        += B
        return loss_sum / n, correct / n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_scheduler(self, opt, epochs, steps_per_epoch, lr, warmup_ratio):
        total = epochs * steps_per_epoch
        warmup = max(1, int(total * warmup_ratio))
        s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup)
        s2 = CosineAnnealingLR(opt, T_max=total - warmup, eta_min=lr * 0.01)
        return SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup])

    def _save(self, name: str):
        path = os.path.join(self.cfg.checkpoint_dir, name)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.cfg,
        }, path)

    def _log(self, stage: str, epoch: int, metrics: dict):
        entry = {"stage": stage, "epoch": epoch, **metrics}
        self.log.append(entry)
        log_path = os.path.join(self.cfg.log_dir, "train_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @classmethod
    def load_checkpoint(cls, path: str, model: RespVoiceModel) -> "Trainer":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        trainer = cls(ckpt["config"], model)
        print(f"[Trainer] Loaded checkpoint from {path}")
        return trainer
