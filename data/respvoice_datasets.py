"""
Dataset classes for RespVoice.

Supports:
  - RespVoiceDataset: generic folder-based dataset for pretraining (unlabeled)
  - LabeledDataset: labeled dataset for downstream tasks (Stage 3)
  - DummyDataset: synthetic data for smoke-testing without real audio files

Downstream datasets:
  SVD (Saarbrücken Voice Database), AVFAD, ICBHI, Coswara
  All follow the same interface.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Callable

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from respvoice.preprocessing import AudioPreprocessor


# ---------------------------------------------------------------------------
# Generic unlabeled dataset (Stage 1 & 2 pretraining)
# ---------------------------------------------------------------------------

class RespVoiceDataset(Dataset):
    """
    Scans a directory recursively for .wav / .flac / .mp3 files.
    Returns {"mel": Tensor(1, n_mels, T)} dicts.

    Args:
        root: path to audio directory
        preprocessor: AudioPreprocessor instance
        modality: "all" | "respiratory" | "voice" (filters by subdirectory name)
        max_samples: cap dataset size for quick experiments
    """

    AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}

    def __init__(
        self,
        root: str,
        preprocessor: AudioPreprocessor,
        modality: str = "all",
        max_samples: Optional[int] = None,
    ):
        self.preprocessor = preprocessor
        self.files = self._collect_files(root, modality, max_samples)
        print(f"[Dataset] Found {len(self.files)} audio files in '{root}' (modality={modality})")

    def _collect_files(self, root, modality, max_samples):
        root = Path(root)
        files = []
        for ext in self.AUDIO_EXTS:
            files.extend(root.rglob(f"*{ext}"))
        if modality != "all":
            files = [f for f in files if modality in str(f).lower()]
        files = sorted(files)
        if max_samples:
            files = files[:max_samples]
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            mel = self.preprocessor(str(path))
        except Exception as e:
            print(f"[Dataset] Warning: could not load {path}: {e}. Returning silence.")
            n_mels = self.preprocessor.n_mels
            T = int(self.preprocessor.target_len / self.preprocessor.sr * 1000 / 32)
            mel = torch.zeros(1, n_mels, T)
        return {"mel": mel, "path": str(path)}


# ---------------------------------------------------------------------------
# Labeled dataset (Stage 3 downstream)
# ---------------------------------------------------------------------------

class LabeledDataset(Dataset):
    """
    Expects a metadata JSON file with entries:
      {"path": "relative/path.wav", "label": 0, "score": 0.7}

    "score" (severity) is optional.

    Args:
        root: dataset root directory
        meta_file: path to JSON metadata file
        preprocessor: AudioPreprocessor instance
        label_map: optional dict to remap string labels to integers
    """

    def __init__(
        self,
        root: str,
        meta_file: str,
        preprocessor: AudioPreprocessor,
        label_map: Optional[dict] = None,
    ):
        self.root = Path(root)
        self.preprocessor = preprocessor
        self.label_map = label_map or {}

        with open(meta_file) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            self.samples = raw
        else:
            self.samples = raw.get("samples", raw.get("data", []))

        print(f"[LabeledDataset] {len(self.samples)} samples from {meta_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        path = self.root / item["path"]
        label = item["label"]
        if isinstance(label, str):
            label = self.label_map.get(label, 0)

        try:
            mel = self.preprocessor(str(path))
        except Exception as e:
            print(f"[LabeledDataset] Warning: {path}: {e}")
            mel = torch.zeros(1, self.preprocessor.n_mels, 250)

        out = {"mel": mel, "label": torch.tensor(label, dtype=torch.long)}
        if "score" in item:
            out["score"] = torch.tensor(item["score"], dtype=torch.float32)
        return out


class CachedMelDataset(Dataset):
    """
    Dataset backed by precomputed .pt tensors.

    Metadata JSON format:
      {"samples": [{"path": "xxx.pt", "label": 0, ...}, ...]}
    """

    def __init__(self, root: str, meta_file: str, include_labels: bool = False):
        self.root = Path(root)
        self.include_labels = include_labels
        with open(meta_file, encoding="utf-8") as f:
            raw = json.load(f)
        self.samples = raw.get("samples", raw if isinstance(raw, list) else [])
        print(f"[CachedMelDataset] {len(self.samples)} cached mels from {meta_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        mel = torch.load(self.root / item["path"], map_location="cpu")
        out = {"mel": mel, "path": str(self.root / item["path"])}
        if self.include_labels and "label" in item:
            out["label"] = torch.tensor(item["label"], dtype=torch.long)
        if self.include_labels and "score" in item:
            out["score"] = torch.tensor(item["score"], dtype=torch.float32)
        return out


# ---------------------------------------------------------------------------
# Dummy dataset for smoke-testing (no real audio needed)
# ---------------------------------------------------------------------------

class DummyDataset(Dataset):
    """
    Generates synthetic log-Mel spectrograms for smoke-testing.
    No audio files required — useful for verifying the pipeline on CPU.

    Args:
        n_samples: number of fake samples
        n_mels: mel bins
        T: time frames
        n_classes: number of output classes (for label generation)
    """

    def __init__(
        self,
        n_samples: int = 64,
        n_mels: int = 64,
        T: int = 250,
        n_classes: int = 2,
    ):
        self.n_samples = n_samples
        self.n_mels = n_mels
        self.T = T
        self.n_classes = n_classes
        print(f"[DummyDataset] {n_samples} synthetic samples ({n_mels}×{T} mel)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        mel = torch.randn(1, self.n_mels, self.T)
        label = idx % self.n_classes
        return {
            "mel": mel,
            "label": torch.tensor(label, dtype=torch.long),
            "score": torch.tensor(float(label) / self.n_classes, dtype=torch.float32),
            "path": f"dummy_{idx:05d}.wav",
        }


# ---------------------------------------------------------------------------
# Dataset factory helpers
# ---------------------------------------------------------------------------

DOWNSTREAM_CONFIGS = {
    "svd": {
        "description": "Saarbrücken Voice Database — voice pathology, binary or multi-class",
        "n_classes": 2,  # normal vs. pathological
        "url": "https://www.phoniatrics.uni-saarland.de/index.php?id=32&L=1",
    },
    "avfad": {
        "description": "Arabic Voice Frequency Analysis Database",
        "n_classes": 2,
        "url": "https://archive.ics.uci.edu/dataset/555/avfad",
    },
    "icbhi": {
        "description": "ICBHI 2017 respiratory sound benchmark, 4-class",
        "n_classes": 4,
        "url": "https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge",
    },
    "coswara": {
        "description": "Coswara — COVID-19 sound dataset (breathing/cough/voice)",
        "n_classes": 2,
        "url": "https://github.com/iiscleap/Coswara-Data",
    },
}


def list_datasets():
    """Print information about supported downstream datasets."""
    print("\nSupported downstream datasets:")
    for name, info in DOWNSTREAM_CONFIGS.items():
        print(f"  {name:10s}: {info['description']}")
        print(f"             URL: {info['url']}")
    print()
