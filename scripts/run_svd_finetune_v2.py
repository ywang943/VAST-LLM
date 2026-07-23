"""
SVD Voice Pathology — Full fine-tuning v2 (aggressive regularization).

Changes vs v1:
  - Freeze backbone first 2 stages, only fine-tune stages 3-4 + CSAF + head
  - Much lower backbone LR (2e-6)
  - Higher dropout (0.5)
  - Label smoothing (0.1)
  - Stronger weight decay (0.05)
  - Gradient accumulation for effective batch 32
  - Longer patience (20)
  - SpecAugment-style time/freq masking on mel input
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from respvoice.htsat_encoder import build_htsat_encoder

SVD_CACHE = Path("data/mel_cache/svd_full")


class SVDPairDataset(Dataset):
    def __init__(self, subjects, indices=None, augment=False):
        self.subjects = [subjects[i] for i in indices] if indices is not None else subjects
        self.augment = augment

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        s = self.subjects[idx]
        vowel = torch.load(s["vowel"], map_location="cpu")
        phrase = torch.load(s["phrase"], map_location="cpu")
        if self.augment:
            vowel = self._spec_augment(vowel)
            phrase = self._spec_augment(phrase)
        return vowel, phrase, s["label"]

    @staticmethod
    def _spec_augment(mel, n_freq_masks=2, freq_mask_width=8,
                      n_time_masks=2, time_mask_width=20):
        mel = mel.clone()
        _, n_mel, n_time = mel.shape
        for _ in range(n_freq_masks):
            f = random.randint(0, freq_mask_width)
            f0 = random.randint(0, max(0, n_mel - f))
            mel[:, f0:f0+f, :] = 0
        for _ in range(n_time_masks):
            t = random.randint(0, time_mask_width)
            t0 = random.randint(0, max(0, n_time - t))
            mel[:, :, t0:t0+t] = 0
        return mel


def collate_fn(batch):
    vowels = torch.stack([b[0] for b in batch])
    phrases = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return vowels, phrases, labels


class SVDFineTuneModelV2(nn.Module):
    def __init__(self, encoder, fusion="concat", feat_dim=768, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion

        if fusion == "concat":
            in_dim = feat_dim * 2
        elif fusion == "attention":
            in_dim = feat_dim
            self.source_attn = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        else:
            in_dim = feat_dim

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def encode(self, mel):
        z = self.encoder(mel)
        return z.mean(dim=1)

    def forward(self, vowel_mel, phrase_mel):
        zv = self.encode(vowel_mel)
        zp = self.encode(phrase_mel)

        if self.fusion == "concat":
            z = torch.cat([zv, zp], dim=1)
        elif self.fusion == "attention":
            stacked = torch.stack([zv, zp], dim=1)
            w = F.softmax(self.source_attn(stacked), dim=1)
            z = (stacked * w).sum(dim=1)
        else:
            z = (zv + zp) / 2

        return self.head(z)


def _rebuild_encoder(init_state, encoder_type):
    if encoder_type == "checkpoint":
        encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    else:
        encoder = build_htsat_encoder(use_csaf=True)
    encoder.load_state_dict(init_state, strict=False)
    return encoder


def train_one_fold(subjects, train_idx, val_idx, test_idx,
                   init_state, args, device, fold_idx):
    random.seed(fold_idx)
    np.random.seed(fold_idx)
    torch.manual_seed(fold_idx)
    torch.cuda.manual_seed_all(fold_idx)

    train_ds = SVDPairDataset(subjects, train_idx, augment=True)
    val_ds = SVDPairDataset(subjects, val_idx, augment=False)
    test_ds = SVDPairDataset(subjects, test_idx, augment=False)

    g = torch.Generator().manual_seed(fold_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2,
                              pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)

    encoder = _rebuild_encoder(init_state, args.encoder)

    # Freeze early stages (0, 1) — only fine-tune stages 2, 3 + norm + CSAF
    for p in encoder.htsat.patch_embed.parameters():
        p.requires_grad = False
    for p in encoder.htsat.layers[0].parameters():
        p.requires_grad = False
    for p in encoder.htsat.layers[1].parameters():
        p.requires_grad = False
    if encoder.htsat.ape and encoder.htsat.absolute_pos_embed is not None:
        encoder.htsat.absolute_pos_embed.requires_grad = False
    # pool1, pool2 are simple reshape ops, but freeze them too
    for p in encoder.pool1.parameters():
        p.requires_grad = False
    for p in encoder.pool2.parameters():
        p.requires_grad = False

    model = SVDFineTuneModelV2(encoder, fusion=args.fusion, dropout=args.dropout).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if fold_idx == 0:
        print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # Parameter groups
    stage34_params = (list(encoder.htsat.layers[2].parameters()) +
                      list(encoder.htsat.layers[3].parameters()) +
                      list(encoder.htsat.norm.parameters()))
    csaf_params = list(encoder.csaf.parameters()) if hasattr(encoder, 'csaf') else []
    head_params = list(model.head.parameters())
    if hasattr(model, 'source_attn'):
        head_params += list(model.source_attn.parameters())

    optimizer = AdamW([
        {"params": stage34_params, "lr": args.backbone_lr},
        {"params": csaf_params, "lr": args.csaf_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    labels = [subjects[i]["label"] for i in train_idx]
    counts = torch.bincount(torch.tensor(labels), minlength=2).float()
    weights = (counts.sum() / (counts.clamp_min(1) * 2)).to(device)

    best_auc, best_state, no_improve = -1.0, None, 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        optimizer.zero_grad()

        for batch_idx, (vowel, phrase, label) in enumerate(train_loader):
            vowel = vowel.to(device, non_blocking=True)
            phrase = phrase.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits = model(vowel, phrase)
            loss = F.cross_entropy(logits, label, weight=weights,
                                   label_smoothing=args.label_smoothing)
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0 or batch_idx == len(train_loader) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum
            n_batch += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batch, 1)

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for vowel, phrase, label in val_loader:
                vowel = vowel.to(device, non_blocking=True)
                phrase = phrase.to(device, non_blocking=True)
                logits = model(vowel, phrase)
                val_probs.append(F.softmax(logits, dim=1)[:, 1].cpu())
                val_labels.append(label)
        vp = torch.cat(val_probs).numpy()
        vl = torch.cat(val_labels).numpy()
        try:
            val_auc = roc_auc_score(vl, vp)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}, "
                  f"val_auc={val_auc:.4f}, best={best_auc:.4f}")

        if no_improve >= args.patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for vowel, phrase, label in test_loader:
            vowel = vowel.to(device, non_blocking=True)
            phrase = phrase.to(device, non_blocking=True)
            logits = model(vowel, phrase)
            test_probs.append(F.softmax(logits, dim=1)[:, 1].cpu())
            test_labels.append(label)
    tp = torch.cat(test_probs).numpy()
    tl = torch.cat(test_labels).numpy()
    try:
        test_auc = roc_auc_score(tl, tp)
    except ValueError:
        test_auc = 0.5

    print(f"  Fold {fold_idx}: test_AUC={test_auc:.4f} (best_val={best_auc:.4f})")

    del model, encoder
    torch.cuda.empty_cache()
    return test_auc, best_auc


def collect_subjects(cache_root):
    raw = json.loads((cache_root / "metadata.json").read_text())
    by_subject = defaultdict(dict)
    labels = {}
    for item in raw["samples"]:
        subj = item["subject_id"]
        labels[subj] = int(item["label"])
        by_subject[subj][item["source"]] = cache_root / item["path"]
    subjects = []
    for subj, parts in by_subject.items():
        if "vowel" in parts and "phrase" in parts:
            subjects.append({"subj": subj, "label": labels[subj],
                             "vowel": parts["vowel"], "phrase": parts["phrase"]})
    return subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="opera_ct",
                        choices=["opera_ct", "checkpoint"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--fusion", default="concat",
                        choices=["concat", "attention", "mean"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--csaf-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", type=Path, default=SVD_CACHE)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}, Fusion: {args.fusion}")
    print(f"Backbone LR: {args.backbone_lr}, CSAF LR: {args.csaf_lr}, "
          f"Head LR: {args.head_lr}")
    print(f"Dropout: {args.dropout}, Label smoothing: {args.label_smoothing}, "
          f"Weight decay: {args.weight_decay}")

    if args.encoder == "checkpoint":
        if not args.ckpt:
            raise ValueError("--ckpt required")
        encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = {k.replace("encoder.", "", 1): v
                 for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
        encoder.load_state_dict(state, strict=False)
    else:
        encoder = build_htsat_encoder(use_csaf=True)

    init_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    del encoder

    subjects = collect_subjects(args.cache)
    y = np.array([s["label"] for s in subjects])
    groups = np.array([s["subj"] for s in subjects])
    print(f"Subjects: {len(subjects)} "
          f"(healthy={(y==0).sum()}, pathological={(y==1).sum()})")

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=1337)
    fold_aucs = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(y, y, groups)):
        n_val = max(1, int(len(train_val_idx) * 0.15))
        perm = torch.randperm(len(train_val_idx),
                              generator=torch.Generator().manual_seed(fold_idx)).numpy()
        train_idx = train_val_idx[perm[n_val:]]
        val_idx = train_val_idx[perm[:n_val]]

        print(f"\nFold {fold_idx}: train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)}")

        test_auc, best_val = train_one_fold(
            subjects, train_idx, val_idx, test_idx,
            init_state, args, device, fold_idx,
        )
        fold_aucs.append(test_auc)

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    print(f"\n{'='*55}")
    print(f"  RESULT: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  MVP reference: 0.958")
    print(f"{'='*55}")

    output = {
        "encoder": args.encoder,
        "checkpoint": args.ckpt,
        "fusion": args.fusion,
        "protocol": "partial fine-tune (stages 3-4 + CSAF + head), SpecAugment, label smoothing",
        "n_subjects": len(subjects),
        "backbone_lr": args.backbone_lr,
        "csaf_lr": args.csaf_lr,
        "head_lr": args.head_lr,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "auc_mean": round(mean_auc, 4),
        "auc_std": round(std_auc, 4),
        "per_fold": [round(a, 4) for a in fold_aucs],
        "mvp_reference": 0.958,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
