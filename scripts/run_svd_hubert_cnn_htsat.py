"""
SVD Voice Pathology — HuBERT CNN frontend + HTS-AT + CSAF (Plan B).

Architecture:
  Raw waveform 16kHz
    → HuBERT CNN (7 layers, 4.2M params, frozen or fine-tuned)
    → (B, 512, T) → projection → (B, T, 768)
    → reshape to pseudo-spectrogram (B, 1, F, T')
    → HTS-AT backbone (31M)
    → CSAF (10M)
    → classifier

This tests whether using HuBERT's learned filterbank instead of Mel
spectrograms improves our multi-scale pipeline.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import HubertModel

from respvoice.htsat_encoder import build_htsat_encoder

SVD_META = Path("data/mel_cache/svd_full")
WAV_CACHE = Path("data/wav_cache/svd_full")
SR = 16000
MAX_SEC = 5.0


def collect_subjects():
    meta = json.loads((SVD_META / "metadata.json").read_text())
    by_subj = defaultdict(dict)
    labels = {}
    for item in meta["samples"]:
        subj = item["subject_id"]
        labels[subj] = int(item["label"])
        by_subj[subj][item["source"]] = str(
            WAV_CACHE / item["path"].replace(".pt", ".npy"))
    subjects = []
    for subj, parts in by_subj.items():
        if "vowel" in parts and "phrase" in parts:
            subjects.append({"subj": subj, "label": labels[subj],
                             "vowel": parts["vowel"], "phrase": parts["phrase"]})
    return subjects


class SVDWavDataset(Dataset):
    def __init__(self, subjects, indices, augment=False):
        self.subjects = [subjects[i] for i in indices]
        self.augment = augment
        self.max_samples = int(SR * MAX_SEC)

    def __len__(self):
        return len(self.subjects)

    def _pad_or_trim(self, audio, target):
        if len(audio) >= target:
            return audio[:target]
        return np.pad(audio, (0, target - len(audio)))

    def _augment(self, audio, p=0.2):
        if random.random() > p:
            return audio
        aug = random.choice(["noise", "pitch"])
        if aug == "noise":
            audio = audio + np.random.normal(0, 0.005, len(audio))
        elif aug == "pitch":
            audio = librosa.effects.pitch_shift(
                audio, sr=SR, n_steps=random.uniform(-2, 2))
        return audio.astype(np.float32)

    def __getitem__(self, idx):
        s = self.subjects[idx]
        vowel = np.load(s["vowel"]).astype(np.float32)
        phrase = np.load(s["phrase"]).astype(np.float32)
        vowel = self._pad_or_trim(vowel, self.max_samples)
        phrase = self._pad_or_trim(phrase, self.max_samples)
        if self.augment:
            vowel = self._augment(vowel, p=0.1)
            phrase = self._augment(phrase, p=0.25)
        # Normalize
        vowel = (vowel - vowel.mean()) / (vowel.std() + 1e-8)
        phrase = (phrase - phrase.mean()) / (phrase.std() + 1e-8)
        return (torch.tensor(vowel), torch.tensor(phrase),
                torch.tensor(s["label"], dtype=torch.long))


def collate_fn(batch):
    vowels = torch.stack([b[0] for b in batch])
    phrases = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return vowels, phrases, labels


class HuBERTCNNHTSATModel(nn.Module):
    """HuBERT CNN frontend → HTS-AT + CSAF → classifier."""

    def __init__(self, htsat_encoder, freeze_cnn=True):
        super().__init__()
        hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.cnn = hubert.feature_extractor  # 7-layer CNN, outputs (B, 512, T)
        self.cnn_proj = hubert.feature_projection  # 512 → 768

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # Adapter: HuBERT CNN outputs (B, T, 768) with T~249 for 5s
        # HTS-AT expects (B, 1, 64, T) mel input
        # We reshape 768-dim features into 64 mel bins × 12 channels ≈ 768
        # Then conv to 1 channel
        self.adapter = nn.Sequential(
            nn.Conv2d(12, 1, kernel_size=1),
            nn.BatchNorm2d(1),
        )

        self.htsat_encoder = htsat_encoder

        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def encode_wav(self, wav):
        """wav (B, samples) → (B, D) pooled feature."""
        cnn_out = self.cnn(wav)          # (B, 512, T)
        cnn_out = cnn_out.transpose(1, 2)  # (B, T, 512)
        proj = self.cnn_proj(cnn_out)    # (B, T, 768)

        B, T, D = proj.shape
        # Reshape 768 → 12 channels × 64 freq bins
        x = proj.view(B, T, 12, 64)     # (B, T, 12, 64)
        x = x.permute(0, 2, 3, 1)       # (B, 12, 64, T)
        x = self.adapter(x)             # (B, 1, 64, T)

        z = self.htsat_encoder(x)        # (B, seq, 768)
        return z.mean(dim=1)             # (B, 768)

    def forward(self, vowel_wav, phrase_wav):
        zv = self.encode_wav(vowel_wav)
        zp = self.encode_wav(phrase_wav)
        z = torch.cat([zv, zp], dim=1)   # (B, 1536)
        # Need to adjust head input
        z = (zv + zp) / 2                # (B, 768) mean fusion for simplicity
        return self.head(z)


def train_one_fold(subjects, train_idx, val_idx, test_idx,
                   encoder_init_state, args, device, fold_idx):
    random.seed(fold_idx)
    np.random.seed(fold_idx)
    torch.manual_seed(fold_idx)
    torch.cuda.manual_seed_all(fold_idx)

    train_ds = SVDWavDataset(subjects, train_idx, augment=True)
    val_ds = SVDWavDataset(subjects, val_idx, augment=False)
    test_ds = SVDWavDataset(subjects, test_idx, augment=False)

    g = torch.Generator().manual_seed(fold_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=4)

    # Rebuild encoder per fold
    htsat_encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
    htsat_encoder.load_state_dict(encoder_init_state, strict=False)

    model = HuBERTCNNHTSATModel(htsat_encoder, freeze_cnn=args.freeze_cnn).to(device)

    if fold_idx == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")

    labels_arr = [subjects[i]["label"] for i in train_idx]
    counts = torch.bincount(torch.tensor(labels_arr), minlength=2).float()
    weights = (counts.sum() / (counts.clamp_min(1) * 2)).to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc, best_state, no_improve = -1.0, None, 0
    grad_accum = args.grad_accum

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, n_step = 0.0, 0
        optimizer.zero_grad()

        for batch_idx, (vowel, phrase, label) in enumerate(train_loader):
            vowel = vowel.to(device)
            phrase = phrase.to(device)
            label = label.to(device)
            logits = model(vowel, phrase)
            loss = F.cross_entropy(logits, label, weight=weights) / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(train_loader) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * grad_accum
            n_step += 1

        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for vowel, phrase, label in val_loader:
                logits = model(vowel.to(device), phrase.to(device))
                vp.append(F.softmax(logits, dim=1)[:, 1].cpu())
                vl.append(label)
        vp = torch.cat(vp).numpy()
        vl = torch.cat(vl).numpy()
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

        if (epoch + 1) % 5 == 0:
            print(f"    Ep {epoch+1}: loss={epoch_loss/n_step:.4f}, "
                  f"val={val_auc:.4f}, best={best_auc:.4f}")

        if no_improve >= args.patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    tp, tl = [], []
    with torch.no_grad():
        for vowel, phrase, label in test_loader:
            logits = model(vowel.to(device), phrase.to(device))
            tp.append(F.softmax(logits, dim=1)[:, 1].cpu())
            tl.append(label)
    tp = torch.cat(tp).numpy()
    tl = torch.cat(tl).numpy()
    try:
        test_auc = roc_auc_score(tl, tp)
    except ValueError:
        test_auc = 0.5

    print(f"  Fold {fold_idx}: test={test_auc:.4f} (val={best_auc:.4f})")
    del model
    torch.cuda.empty_cache()
    return test_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="opera_ct",
                   choices=["opera_ct", "checkpoint"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--freeze-cnn", action="store_true", default=True)
    p.add_argument("--no-freeze-cnn", dest="freeze_cnn", action="store_false")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}, Freeze CNN: {args.freeze_cnn}")

    # Build base HTS-AT encoder
    if args.encoder == "checkpoint" and args.ckpt:
        encoder = build_htsat_encoder(ckpt_path=None, use_csaf=True)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = {k.replace("encoder.", "", 1): v
                 for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
        encoder.load_state_dict(state, strict=False)
    else:
        encoder = build_htsat_encoder(use_csaf=True)
    encoder_init_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    del encoder

    subjects = collect_subjects()
    y = np.array([s["label"] for s in subjects])
    groups = np.array([s["subj"] for s in subjects])
    print(f"Subjects: {len(subjects)} (H={(y==0).sum()}, P={(y==1).sum()})")

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=1337)
    fold_aucs = []

    for fold_idx, (tv_idx, test_idx) in enumerate(skf.split(y, y, groups)):
        n_val = max(1, int(len(tv_idx) * 0.15))
        perm = torch.randperm(len(tv_idx),
                              generator=torch.Generator().manual_seed(fold_idx)).numpy()
        train_idx = tv_idx[perm[n_val:]]
        val_idx = tv_idx[perm[:n_val]]

        print(f"\nFold {fold_idx}: tr={len(train_idx)} val={len(val_idx)} te={len(test_idx)}")
        auc = train_one_fold(subjects, train_idx, val_idx, test_idx,
                             encoder_init_state, args, device, fold_idx)
        fold_aucs.append(auc)

    m, s = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    print(f"\n{'='*55}")
    print(f"  HuBERT-CNN + HTS-AT + CSAF: {m:.4f} ± {s:.4f}")
    print(f"{'='*55}")

    output = {
        "method": "HuBERT CNN frontend + HTS-AT + CSAF",
        "encoder": args.encoder, "freeze_cnn": args.freeze_cnn,
        "n_subjects": len(subjects),
        "auc_mean": round(m, 4), "auc_std": round(s, 4),
        "per_fold": [round(a, 4) for a in fold_aucs],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
