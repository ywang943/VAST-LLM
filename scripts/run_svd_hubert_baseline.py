"""
SVD Voice Pathology — HuBERT baseline (replicating MVP protocol).

Directly fine-tunes HuBERT-base on raw SVD waveforms using the same protocol
as MVP (Koudounas et al., INTERSPEECH 2025):
  - HuBERT-base (94.6M params) from HuggingFace
  - Raw waveform input, 16kHz, 5s max
  - 10-fold speaker-independent CV
  - AdamW, lr=5e-5, weight_decay=0.01
  - Data augmentation: noise/speed/pitch

This serves as an upper-bound reference to verify the MVP result (0.958 AUC)
and compare against our HTS-AT + CSAF approach.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).parent.parent))

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
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SVD_CACHE = Path("data/mel_cache/svd_full")
SVD_ARCHIVES = Path("data/downloads/svd_archives")
SR = 16000
MAX_DURATION = 5.0
MAX_SAMPLES = int(SR * MAX_DURATION)


# ── Dataset (raw waveform from NSP files) ────────────────────────────────────

def parse_nsp(data):
    """Parse Saarbruecken NSP format → numpy array."""
    import struct
    if data[:4] == b'FORM':
        offset = 8
        while offset < len(data) - 8:
            chunk_id = data[offset:offset+4]
            chunk_size = struct.unpack('>I', data[offset+4:offset+8])[0]
            if chunk_id == b'DS16':
                offset += 8
                sub_id = data[offset:offset+4]
                if sub_id == b'HEDR':
                    hedr_size = struct.unpack('>I', data[offset+4:offset+8])[0]
                    sr_bytes = data[offset+8:offset+12]
                    file_sr = struct.unpack('<I', sr_bytes)[0]
                    body_offset = offset + 8 + hedr_size
                    body_id = data[body_offset:body_offset+4]
                    if body_id == b'BODY':
                        body_size = struct.unpack('>I', data[body_offset+4:body_offset+8])[0]
                        pcm = data[body_offset+8:body_offset+8+body_size]
                        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                        return audio, file_sr
                break
            offset += 8 + chunk_size
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, 50000


def collect_subjects_with_paths(cache_root, archives_dir):
    """Collect subject metadata with archive paths for raw audio loading."""
    meta = json.loads((cache_root / "metadata.json").read_text())
    by_subject = defaultdict(dict)
    labels = {}
    for item in meta["samples"]:
        subj = item["subject_id"]
        labels[subj] = int(item["label"])
        by_subject[subj][item["source"]] = {
            "archive": item.get("archive", ""),
            "member": item.get("member", ""),
        }
    subjects = []
    for subj, parts in by_subject.items():
        if "vowel" in parts and "phrase" in parts:
            subjects.append({
                "subj": subj, "label": labels[subj],
                "vowel_archive": str(archives_dir / parts["vowel"]["archive"]),
                "vowel_member": parts["vowel"]["member"],
                "phrase_archive": str(archives_dir / parts["phrase"]["archive"]),
                "phrase_member": parts["phrase"]["member"],
            })
    return subjects


def load_raw_audio(archive_path, member_path):
    """Load raw waveform from SVD zip/nsp, resample to 16kHz."""
    with ZipFile(archive_path, 'r') as zf:
        data = zf.read(member_path)
    audio, file_sr = parse_nsp(data)
    if file_sr != SR:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
    return audio


class SVDWaveformDataset(Dataset):
    def __init__(self, subjects, indices, feature_extractor,
                 max_duration=MAX_DURATION, augment=False, fusion="concat"):
        self.subjects = [subjects[i] for i in indices]
        self.fe = feature_extractor
        self.max_samples = int(SR * max_duration)
        self.augment = augment
        self.fusion = fusion

    def __len__(self):
        return len(self.subjects)

    def _augment(self, audio):
        if random.random() > 0.25:
            return audio
        aug_type = random.choice([1, 2, 3, 4])
        if aug_type == 1:
            noise = np.random.normal(0, 0.005, len(audio))
            audio = audio + noise
        elif aug_type == 2:
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.2))
        elif aug_type == 3:
            audio = librosa.effects.pitch_shift(audio, sr=SR,
                                                 n_steps=random.uniform(-3, 3))
        elif aug_type == 4:
            noise = np.random.normal(0, 0.005, len(audio))
            audio = audio + noise
            audio = librosa.effects.pitch_shift(audio, sr=SR,
                                                 n_steps=random.uniform(-2, 2))
        return audio.astype(np.float32)

    def __getitem__(self, idx):
        s = self.subjects[idx]
        try:
            vowel = load_raw_audio(s["vowel_archive"], s["vowel_member"])
            phrase = load_raw_audio(s["phrase_archive"], s["phrase_member"])
        except Exception as e:
            vowel = np.zeros(self.max_samples, dtype=np.float32)
            phrase = np.zeros(self.max_samples, dtype=np.float32)

        if self.augment:
            vowel = self._augment(vowel)
            phrase = self._augment(phrase)

        if self.fusion == "concat":
            audio = np.concatenate([vowel, phrase])
        elif self.fusion == "vowel_only":
            audio = vowel
        elif self.fusion == "phrase_only":
            audio = phrase
        else:
            audio = np.concatenate([vowel, phrase])

        inputs = self.fe(
            audio.squeeze(),
            sampling_rate=SR,
            return_tensors="pt",
            max_length=self.max_samples * (2 if self.fusion == "concat" else 1),
            truncation=True,
            padding="max_length",
        )
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_fold(subjects, train_idx, val_idx, test_idx,
                   args, device, fold_idx):
    random.seed(fold_idx)
    np.random.seed(fold_idx)
    torch.manual_seed(fold_idx)

    fe = AutoFeatureExtractor.from_pretrained(args.model_name)

    train_ds = SVDWaveformDataset(subjects, train_idx, fe,
                                  augment=True, fusion=args.fusion)
    val_ds = SVDWaveformDataset(subjects, val_idx, fe,
                                augment=False, fusion=args.fusion)
    test_ds = SVDWaveformDataset(subjects, test_idx, fe,
                                 augment=False, fusion=args.fusion)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4)

    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name, num_labels=2,
    ).to(device)

    if fold_idx == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params/1e6:.1f}M")

    # Class weights
    labels = [subjects[i]["label"] for i in train_idx]
    counts = torch.bincount(torch.tensor(labels), minlength=2).float()
    weights = (counts.sum() / (counts.clamp_min(1) * 2)).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc, best_state, no_improve = -1.0, None, 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            labels_batch = batch["labels"].to(device)
            outputs = model(input_values=input_values, labels=labels_batch)
            loss = F.cross_entropy(outputs.logits, labels_batch, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                outputs = model(input_values=input_values)
                probs = F.softmax(outputs.logits, dim=1)[:, 1].cpu()
                val_probs.append(probs)
                val_labels.append(batch["labels"])
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

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}, "
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
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            outputs = model(input_values=input_values)
            probs = F.softmax(outputs.logits, dim=1)[:, 1].cpu()
            test_probs.append(probs)
            test_labels.append(batch["labels"])
    tp = torch.cat(test_probs).numpy()
    tl = torch.cat(test_labels).numpy()
    try:
        test_auc = roc_auc_score(tl, tp)
    except ValueError:
        test_auc = 0.5

    print(f"  Fold {fold_idx}: test_AUC={test_auc:.4f} (best_val={best_auc:.4f})")
    del model
    torch.cuda.empty_cache()
    return test_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="facebook/hubert-base-ls960")
    parser.add_argument("--fusion", default="concat",
                        choices=["concat", "vowel_only", "phrase_only"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cache", type=Path, default=SVD_CACHE)
    parser.add_argument("--archives", type=Path, default=SVD_ARCHIVES)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Fusion: {args.fusion}")

    subjects = collect_subjects_with_paths(args.cache, args.archives)
    y = np.array([s["label"] for s in subjects])
    groups = np.array([s["subj"] for s in subjects])
    print(f"Subjects: {len(subjects)} (healthy={(y==0).sum()}, "
          f"pathological={(y==1).sum()})")

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

        test_auc = train_one_fold(
            subjects, train_idx, val_idx, test_idx,
            args, device, fold_idx,
        )
        fold_aucs.append(test_auc)

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    print(f"\n{'='*55}")
    print(f"  HuBERT baseline: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  MVP reference:   0.958")
    print(f"{'='*55}")

    output = {
        "model": args.model_name,
        "fusion": args.fusion,
        "protocol": "HuBERT fine-tune, raw waveform, 10-fold speaker-independent CV",
        "n_subjects": len(subjects),
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
