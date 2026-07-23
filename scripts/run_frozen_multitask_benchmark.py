"""Frozen-encoder multi-task benchmark for RespVoice.

The protocol follows OPERA's evaluation principle: the acoustic encoder is
frozen and each downstream task trains only a linear classifier. Both the
Stage-4 representation and the complete TPA-CSAF representation are measured
under the same split, optimizer, and seeds.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "opera_src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset

from respvoice.htsat_encoder import build_htsat_encoder
from respvoice.dual_input_encoder import build_dual_input_encoder

WAV_TARGET_LEN = 16000 * 8

TASKS = {
    "icbhi_copd": {
        "name": "ICBHI Healthy-vs-COPD",
        "roots": ["data/mel_cache/opera_icbhi_disease"],
        "wav_roots": ["data/wav_cache/opera_icbhi_disease"],
        "n_classes": 2,
        "split": "icbhi",
    },
    "copd_severity": {
        "name": "Respiratory@TR COPD Severity",
        "roots": ["data/mel_cache/opera_copd"],
        "wav_roots": ["data/wav_cache/opera_copd"],
        "n_classes": 5,
        "split": "metadata",
    },
    "kauh_obstructive": {
        "name": "KAUH Obstructive Disease",
        "roots": ["data/mel_cache/opera_kauh"],
        "wav_roots": ["data/wav_cache/opera_kauh"],
        "n_classes": 2,
        "split": "metadata",
    },
    "svd_pathology": {
        "name": "SVD Voice Pathology",
        "roots": ["data/mel_cache/svd_full"],
        "wav_roots": ["data/wav_cache/svd_full"],
        "n_classes": 2,
        "split": "metadata",
        "note": "Complete SVD with fixed subject-independent splits.",
    },
}


class MultiCacheDataset(Dataset):
    def __init__(self, roots, wav_roots=None):
        self.samples = []
        wav_roots = wav_roots or [None] * len(roots)
        for root_str, wav_root_str in zip(roots, wav_roots):
            root = Path(root_str)
            raw = json.loads((root / "metadata.json").read_text())
            for item in raw.get("samples", raw if isinstance(raw, list) else []):
                if "label" not in item:
                    continue
                sample = dict(item)
                sample["_root"] = str(root)
                sample["_wav_root"] = wav_root_str
                sample["label"] = int(sample["label"])
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        mel = torch.load(
            Path(item["_root"]) / item["path"], map_location="cpu"
        )
        output = {
            "mel": mel,
            "label": torch.tensor(item["label"], dtype=torch.long),
        }
        if item.get("_wav_root"):
            wav_path = (
                Path(item["_wav_root"]) /
                item["path"].replace(".pt", ".npy")
            )
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing downstream waveform: {wav_path}")
            wav = np.load(wav_path).astype(np.float32)
            wav = (wav - wav.mean()) / (wav.std() + 1e-8)
            if len(wav) >= WAV_TARGET_LEN:
                wav = wav[:WAV_TARGET_LEN]
            else:
                wav = np.pad(wav, (0, WAV_TARGET_LEN - len(wav)))
            output["wav"] = torch.from_numpy(wav)
        return output


def split_indices(dataset, mode):
    if mode == "icbhi":
        trainval = [
            i for i, sample in enumerate(dataset.samples)
            if sample.get("split") != "test"
        ]
        test = [
            i for i, sample in enumerate(dataset.samples)
            if sample.get("split") == "test"
        ]
        labels = [dataset.samples[i]["label"] for i in trainval]
        train, val = train_test_split(
            trainval, test_size=0.2, random_state=1337, stratify=labels
        )
        return list(train), list(val), test

    split_map = {"train": [], "val": [], "test": []}
    for i, sample in enumerate(dataset.samples):
        split_map.setdefault(sample.get("split", "train"), []).append(i)
    if not split_map["val"]:
        train = split_map["train"]
        labels = [dataset.samples[i]["label"] for i in train]
        train, val = train_test_split(
            train, test_size=0.15, random_state=1337, stratify=labels
        )
        split_map["train"], split_map["val"] = list(train), list(val)
    return split_map["train"], split_map["val"], split_map["test"]


@torch.no_grad()
def encode_stages(encoder, mel, waveform=None):
    x = encoder._preprocess(mel)
    if waveform is not None and hasattr(encoder, "hubert_cnn"):
        cnn_out = encoder.hubert_cnn(waveform)
        wav_img = encoder.wav_adapter(cnn_out)
        x = encoder.fusion(x, wav_img)
    x = encoder.htsat.patch_embed(x)
    if encoder.htsat.ape:
        x = x + encoder.htsat.absolute_pos_embed
    x = encoder.htsat.pos_drop(x)
    x, _ = encoder.htsat.layers[0](x)
    e1 = encoder.pool1(x)
    x, _ = encoder.htsat.layers[1](x)
    e2 = encoder.pool2(x)
    x, _ = encoder.htsat.layers[2](x)
    e3 = x
    x, _ = encoder.htsat.layers[3](x)
    e4 = encoder.htsat.norm(x)
    return e1, e2, e3, e4


@torch.no_grad()
def extract_features(encoder, dataset, indices, variant, device, batch_size):
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices), batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    features, labels = [], []
    encoder.eval()
    for batch in loader:
        mel = batch["mel"].to(device, non_blocking=True)
        waveform = batch.get("wav")
        if waveform is not None:
            waveform = waveform.to(device, non_blocking=True)
        if variant == "tpa_csaf":
            z = encoder(mel, waveform) if hasattr(encoder, "hubert_cnn") else encoder(mel)
        else:
            stages = encode_stages(encoder, mel, waveform)
            z = stages[-1] if variant == "stage4" else torch.cat(stages, dim=-1)
        features.append(z.mean(dim=1).cpu())
        labels.append(batch["label"])
    return torch.cat(features), torch.cat(labels)


def metric(logits, labels, n_classes):
    probs = logits.softmax(dim=1).numpy()
    y = labels.numpy()
    pred = probs.argmax(axis=1)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y, probs[:, 1])
        else:
            auc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {
        "auroc": float(auc),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def train_linear_probe(features, labels, splits, n_classes, seed, epochs, lr):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_idx, val_idx, test_idx = splits
    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    head = nn.Linear(features.shape[1], n_classes)
    counts = torch.bincount(y_train, minlength=n_classes).float()
    weights = counts.sum() / (counts.clamp_min(1) * n_classes)
    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=32, shuffle=True,
        generator=generator,
    )

    best_auc, best_state = -float("inf"), None
    for _ in range(epochs):
        head.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(head(x), y, weight=weights)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_auc = metric(head(x_val), y_val, n_classes)["auroc"]
        score = val_auc if np.isfinite(val_auc) else -float("inf")
        if score > best_auc:
            best_auc = score
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        return metric(head(x_test), y_test, n_classes)


def load_encoder(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key.replace("encoder.", "", 1): value
        for key, value in ckpt["model_state"].items()
        if key.startswith("encoder.")
    }
    is_dual = any(key.startswith("encoder.hubert_cnn.") for key in ckpt["model_state"])
    encoder = (
        build_dual_input_encoder(ckpt_path=None, use_csaf=True)
        if is_dual else build_htsat_encoder(ckpt_path=None, use_csaf=True)
    )
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Encoder load: missing={len(missing)} unexpected={len(unexpected)}")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    print(f"Encoder type: {'dual-input' if is_dual else 'mel-only'}")
    return encoder.to(device).eval(), ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--variants", nargs="+", choices=("stage4", "concat", "tpa_csaf"),
        default=["stage4", "concat", "tpa_csaf"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="checkpoints/frozen_multitask/results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, ckpt = load_encoder(args.checkpoint, device)
    results = {
        "protocol": "frozen encoder + mean pooling + linear classifier",
        "checkpoint": args.checkpoint,
        "pretrain_epoch": ckpt.get("epoch"),
        "initialization": ckpt.get("initialization", "legacy/unknown"),
        "tasks": {},
    }

    for task_key in args.tasks:
        cfg = TASKS[task_key]
        dataset = MultiCacheDataset(
            cfg["roots"], cfg.get("wav_roots") if hasattr(encoder, "hubert_cnn") else None
        )
        splits = split_indices(dataset, cfg["split"])
        print(
            f"\n{cfg['name']}: n={len(dataset)} "
            f"train/val/test={tuple(map(len, splits))}"
        )
        task_result = {"note": cfg.get("note"), "variants": {}}
        for variant in args.variants:
            parts = [
                extract_features(
                    encoder, dataset, idx, variant, device, args.batch_size
                )
                for idx in splits
            ]
            features = torch.cat([part[0] for part in parts])
            labels = torch.cat([part[1] for part in parts])
            offsets = np.cumsum([0] + [len(part[1]) for part in parts])
            local_splits = tuple(
                list(range(offsets[i], offsets[i + 1])) for i in range(3)
            )
            seed_results = [
                train_linear_probe(
                    features, labels, local_splits, cfg["n_classes"], seed,
                    args.epochs, args.lr,
                )
                for seed in args.seeds
            ]
            aucs = [item["auroc"] for item in seed_results]
            task_result["variants"][variant] = {
                "auroc_mean": float(np.nanmean(aucs)),
                "auroc_std": float(np.nanstd(aucs)),
                "per_seed": seed_results,
            }
            print(
                f"  {variant}: AUROC={np.nanmean(aucs):.4f} "
                f"+/- {np.nanstd(aucs):.4f}"
            )
        results["tasks"][task_key] = task_result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
