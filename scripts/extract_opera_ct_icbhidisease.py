"""Extract OPERA-CT features for the official OPERA ICBHI disease task.

Run from the repository root:
  .venv/Scripts/python.exe scripts/extract_opera_ct_icbhidisease.py

This intentionally uses OPERA's own preprocessing and model definition from
opera_src so the feature file is directly compatible with
src.benchmark.linear_eval.linear_evaluation_icbhidisease(use_feature="operaCT").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
OPERA_ROOT = ROOT / "opera_src"
FEATURE_DIR = OPERA_ROOT / "feature" / "icbhidisease_eval"
CKPT_PATH = OPERA_ROOT / "cks" / "model" / "encoder-operaCT.ckpt"
OUT_PATH = FEATURE_DIR / "operaCT_feature.npy"


def main() -> None:
    sys.path.insert(0, str(OPERA_ROOT))

    from src.benchmark.model_util import initialize_pretrained_model
    from src.util import get_entire_signal_librosa

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing OPERA-CT checkpoint: {CKPT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device}")
    print(f"[extract] checkpoint={CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = initialize_pretrained_model("operaCT")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)

    sound_paths = np.load(FEATURE_DIR / "sound_dir_loc.npy", allow_pickle=True)
    features: list[list[float]] = []

    with torch.no_grad():
        for rel_path in tqdm(sound_paths, desc="OPERA-CT features"):
            # OPERA utility expects filename without .wav and relative to OPERA_ROOT.
            file_no_ext = str(Path(rel_path).with_suffix(""))
            spec = get_entire_signal_librosa(
                str(OPERA_ROOT) + "/",
                file_no_ext,
                spectrogram=True,
                input_sec=8,
                pad=True,
            )
            spec = np.asarray(spec, dtype=np.float32)
            x = torch.from_numpy(spec).unsqueeze(0).to(device)
            feat = model.extract_feature(x, dim=768).detach().cpu().numpy()[0]
            features.append(feat.tolist())

    x_data = np.asarray(features, dtype=np.float32)
    np.save(OUT_PATH, x_data)
    print(f"[extract] saved {OUT_PATH}")
    print(f"[extract] shape={x_data.shape}")


if __name__ == "__main__":
    main()
