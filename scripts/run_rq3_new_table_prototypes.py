#!/usr/bin/env python3
"""Prototype zero-shot baselines for the new Table 5 target tasks.

For non-LLM audio encoders, target datasets have no task-specific training
split. We use nearest-centroid prototypes learned from the closest seen source
task in the same label space:
  T1-T3 Bridge2AI subtypes <- S7 Bridge2AI voice pathology/control
  T4 Coswara covid breath  <- S3 Coswara covid exhale (or cough)
  T5 Coswara smoker breath <- S5 Coswara smoker cough
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "opera_src"))

from respvoice.htsat_encoder import build_htsat_encoder


SR = 16000
WAV_LEN = SR * 8


TASKS = {
    "b2ai_voice_pathology": {"mel_root": "data/mel_cache/b2ai_voice_pathology", "name": "S7 Bridge2AI"},
    "b2ai_laryngeal_cancer": {"mel_root": "data/mel_cache/b2ai_laryngeal_cancer", "name": "T1 Cancer"},
    "b2ai_benign_lesions": {"mel_root": "data/mel_cache/b2ai_benign_lesions", "name": "T2 Benign"},
    "b2ai_laryngeal_dystonia": {"mel_root": "data/mel_cache/b2ai_laryngeal_dystonia", "name": "T3 Dystonia"},
    "coswara_covid_exhale": {"mel_root": "data/mel_cache/coswara_covid_exhale", "name": "S3 Exhale"},
    "coswara_covid_cough": {"mel_root": "data/mel_cache/coswara_covid_cough", "name": "S4 Cough"},
    "coswara_covid_breathing": {"mel_root": "data/mel_cache/coswara_covid_breathing", "name": "T4 Breath"},
    "coswara_smoker_cough": {"mel_root": "data/mel_cache/coswara_smoker_cough", "name": "S5 Smoker Cough"},
    "coswara_smoker_breathing": {"mel_root": "data/mel_cache/coswara_smoker_breathing", "name": "T5 Smoker Breath"},
    "uk_covid_cough": {"mel_root": "data/mel_cache/uk_covid_cough", "name": "T6 UK COVID Cough"},
}

TARGETS = {
    "T1_laryngeal_cancer": ("b2ai_voice_pathology", "b2ai_laryngeal_cancer"),
    "T2_benign_lesions": ("b2ai_voice_pathology", "b2ai_benign_lesions"),
    "T3_laryngeal_dystonia": ("b2ai_voice_pathology", "b2ai_laryngeal_dystonia"),
    "T4_covid_breath": ("coswara_covid_exhale", "coswara_covid_breathing"),
    "T5_smoker_breath": ("coswara_smoker_cough", "coswara_smoker_breathing"),
    "T6_uk_covid_cough": ("coswara_covid_cough", "uk_covid_cough"),
}


class LabeledCachedMel(Dataset):
    def __init__(self, mel_root):
        self.root = ROOT / mel_root
        raw = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        samples = raw.get("samples", raw if isinstance(raw, list) else [])
        self.samples = [s for s in samples if "label" in s and (self.root / s["path"]).exists()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "mel": torch.load(str(self.root / s["path"]), map_location="cpu"),
            "label": int(s["label"]),
            "split": s.get("split", "train"),
        }


def collate(batch):
    return {
        "mel": torch.stack([b["mel"] for b in batch], dim=0),
        "label": np.array([b["label"] for b in batch], dtype=np.int64),
        "split": [b["split"] for b in batch],
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vast_like_encoder(ckpt_path, device, use_csaf=True):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("encoder.")
    }
    encoder = build_htsat_encoder(ckpt_path=None, freeze_backbone=True, use_csaf=use_csaf)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{ckpt_path} load mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    return encoder.to(device).eval()


def load_opera_ct(device):
    ckpt_path = ROOT / "opera_src/cks/model/encoder-operaCT.ckpt"
    return build_htsat_encoder(ckpt_path=str(ckpt_path), freeze_backbone=True, use_csaf=False).to(device).eval()


def load_efficientnet(device):
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
    model.classifier = nn.Identity()
    return model.to(device).eval()


def load_audiomae(device):
    import timm

    model = timm.create_model(
        "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k",
        pretrained=True,
        num_classes=0,
    )
    return model.to(device).eval()


def load_hubert(device):
    from transformers import AutoModel

    return AutoModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()


@torch.no_grad()
def extract_features(encoder, task_key, device, batch_size, num_workers):
    ds = LabeledCachedMel(TASKS[task_key]["mel_root"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    feats, labels, splits = [], [], []
    for batch in loader:
        z = encoder(batch["mel"].to(device, non_blocking=True))
        feats.append(z.mean(dim=1).cpu().numpy())
        labels.append(batch["label"])
        splits.extend(batch["split"])
    return {
        "features": np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "splits": np.array(splits),
    }


@torch.no_grad()
def extract_marvel_features(model, task_key, device, batch_size, num_workers):
    ds = LabeledCachedMel(TASKS[task_key]["mel_root"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    feats, labels, splits = [], [], []
    for batch in loader:
        mel = batch["mel"]
        x = mel.expand(-1, 3, -1, -1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(device)
        feats.append(model(x).cpu().numpy())
        labels.append(batch["label"])
        splits.extend(batch["split"])
    return {
        "features": np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "splits": np.array(splits),
    }


@torch.no_grad()
def extract_audiomae_features(model, task_key, device, batch_size, num_workers):
    ds = LabeledCachedMel(TASKS[task_key]["mel_root"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    feats, labels, splits = [], [], []
    for batch in loader:
        # Public AudioMAE AS2M checkpoint expects (B, 1, time=1024, mel=128).
        x = batch["mel"].transpose(2, 3)
        x = F.interpolate(x, size=(1024, 128), mode="bilinear", align_corners=False).to(device)
        feats.append(model(x).cpu().numpy())
        labels.append(batch["label"])
        splits.extend(batch["split"])
    return {
        "features": np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "splits": np.array(splits),
    }


@torch.no_grad()
def extract_hubert_mel_recon_features(model, task_key, device, batch_size, num_workers, n_iter=2):
    ds = LabeledCachedMel(TASKS[task_key]["mel_root"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    inv = torchaudio.transforms.InverseMelScale(
        n_stft=513, n_mels=64, sample_rate=SR, f_min=50, f_max=8000
    ).to(device)
    gl = torchaudio.transforms.GriffinLim(
        n_fft=1024, win_length=1024, hop_length=512, n_iter=n_iter, power=2.0
    ).to(device)
    feats, labels, splits = [], [], []
    for batch in loader:
        mel = torch.exp(batch["mel"].squeeze(1)).clamp_min(1e-6).to(device, non_blocking=True)
        wav = gl(inv(mel))
        wav = wav[:, :WAV_LEN]
        if wav.size(1) < WAV_LEN:
            wav = F.pad(wav, (0, WAV_LEN - wav.size(1)))
        wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-8)
        hidden = model(input_values=wav).last_hidden_state
        feat = hidden.mean(dim=1).cpu().numpy()
        feats.append(np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0))
        labels.append(batch["label"])
        splits.extend(batch["split"])
    return {
        "features": np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "splits": np.array(splits),
    }


def prototype_auc(source, target):
    src_mask = np.isin(source["splits"], ["train", "val"])
    tgt_mask = target["splits"] == "test"
    if not tgt_mask.any():
        tgt_mask = np.ones_like(target["splits"], dtype=bool)
    Xs = source["features"][src_mask]
    ys = source["labels"][src_mask]
    Xt = target["features"][tgt_mask]
    yt = target["labels"][tgt_mask]

    if len(np.unique(ys)) != 2 or len(np.unique(yt)) != 2:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)
    Xt = scaler.transform(Xt)
    c0 = Xs[ys == 0].mean(axis=0)
    c1 = Xs[ys == 1].mean(axis=0)
    d0 = np.linalg.norm(Xt - c0[None, :], axis=1)
    d1 = np.linalg.norm(Xt - c1[None, :], axis=1)
    score = d0 - d1
    auc = float(roc_auc_score(yt, score))
    acc = float(((score >= 0).astype(int) == yt).mean())
    return {"auroc": auc, "accuracy": acc, "n": int(len(yt))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="checkpoints/rq3_prototypes/new_table_prototypes.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--methods", nargs="+", default=None,
                   choices=[
                       "AudioMAE", "OPERA-COLA", "JEPA", "VAST (LP proto)",
                       "MARVEL / Unified", "MVP (HuBERT, mel→wav)"
                   ])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    methods = {
        "OPERA-COLA": lambda: load_opera_ct(device),
        "JEPA": lambda: load_vast_like_encoder(
            ROOT / "checkpoints/htsat_jepa_only_d768/htsat_lejepa_best.pt", device, use_csaf=True
        ),
        "VAST (LP proto)": lambda: load_vast_like_encoder(
            ROOT / "checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt", device, use_csaf=True
        ),
        "MARVEL / Unified": lambda: load_efficientnet(device),
        "AudioMAE": lambda: load_audiomae(device),
        "MVP (HuBERT, mel→wav)": lambda: load_hubert(device),
    }
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in set(args.methods)}

    needed = sorted(set([x for pair in TARGETS.values() for x in pair]))
    results = {}
    for method, loader in methods.items():
        print(f"\n{'=' * 70}\n{method}\n{'=' * 70}")
        encoder = loader()
        cache = {}
        for task_key in needed:
            print(f"  Extracting {task_key}")
            if method == "MARVEL / Unified":
                cache[task_key] = extract_marvel_features(
                    encoder, task_key, device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            elif method == "AudioMAE":
                cache[task_key] = extract_audiomae_features(
                    encoder, task_key, device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            elif method == "MVP (HuBERT, mel→wav)":
                cache[task_key] = extract_hubert_mel_recon_features(
                    encoder, task_key, device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
            else:
                cache[task_key] = extract_features(
                    encoder, task_key, device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
        row = {}
        for target_id, (src_key, tgt_key) in TARGETS.items():
            res = prototype_auc(cache[src_key], cache[tgt_key])
            row[target_id] = res
            if res:
                print(f"  {target_id}: AUROC={res['auroc']:.4f} Acc={res['accuracy']:.4f} n={res['n']}")
            else:
                print(f"  {target_id}: N/A")
        results[method] = row
        del encoder
        torch.cuda.empty_cache()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    header = f"{'Method':<18} {'T1':>8} {'T2':>8} {'T3':>8} {'T4':>8} {'T5':>8} {'Avg':>8}"
    print(header)
    print("-" * len(header))
    for method, row in results.items():
        vals = [row[k]["auroc"] if row.get(k) else None for k in TARGETS]
        avg = float(np.mean([v for v in vals if v is not None]))
        line = f"{method:<18}" + "".join(f"{v:>8.4f}" if v is not None else f"{'—':>8}" for v in vals)
        line += f"{avg:>8.4f}"
        print(line)


if __name__ == "__main__":
    main()
