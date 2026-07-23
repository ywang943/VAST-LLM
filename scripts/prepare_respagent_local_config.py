#!/usr/bin/env python
"""Create local Resp-Agent config files with paths valid in this workspace."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def write_generator_config(repo_dir: Path, out_path: Path) -> None:
    src = repo_dir / "Generator" / "config.yaml"
    config = yaml.safe_load(src.read_text(encoding="utf-8"))
    gen_dir = repo_dir / "Generator"
    config["paths"]["beats_tokenizer"] = str(
        gen_dir / "pretrained_models" / "Tokenizer_iter3_plus_AS2M.pt"
    )
    config["paths"]["beats_feature_extractor_checkpoint"] = str(
        gen_dir / "pretrained_models" / "BEATs_iter3_plus_AS2M.pt"
    )
    config["paths"]["checkpoint_dir"] = str(gen_dir / "checkpoints")
    config["logging"]["wandb"]["enabled"] = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def write_diagnoser_config(repo_dir: Path, out_path: Path, data_root: Path | None) -> None:
    src = repo_dir / "Diagnoser" / "config.yaml"
    config = yaml.safe_load(src.read_text(encoding="utf-8"))
    diag_dir = repo_dir / "Diagnoser"
    if data_root is not None:
        config["data"]["train_root"] = str(data_root / "train")
        config["data"]["val_root"] = str(data_root / "valid")
        config["data"]["test_root"] = str(data_root / "test")
    config["paths"]["beats_tokenizer"] = str(
        diag_dir / "pretrained_models" / "Tokenizer_iter3_plus_AS2M.pt"
    )
    config["paths"]["beats_feature_extractor_checkpoint"] = str(
        diag_dir / "pretrained_models" / "BEATs_iter3_plus_AS2M.pt"
    )
    config["paths"]["checkpoint_dir"] = str(diag_dir / "checkpoints")
    config["logging"]["wandb"]["enabled"] = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="external/Resp-Agent")
    parser.add_argument("--out-dir", default="checkpoints/respagent_waveform/configs")
    parser.add_argument(
        "--diagnoser-data-root",
        default=None,
        help="Optional dataset root containing train/valid/test for diagnoser training.",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    data_root = Path(args.diagnoser_data_root).resolve() if args.diagnoser_data_root else None

    write_generator_config(repo_dir, out_dir / "generator_config.yaml")
    write_diagnoser_config(repo_dir, out_dir / "diagnoser_config.yaml", data_root)
    print(f"Wrote local Resp-Agent configs to {out_dir}")


if __name__ == "__main__":
    main()
