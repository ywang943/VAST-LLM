#!/usr/bin/env python
"""Download Resp-Agent assets one file at a time.

This avoids snapshot-level failures when a large HF/Xet file stalls or leaves a
partial local file.  Files are copied into the official Resp-Agent directory
layout after each successful download.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


GENERATOR_FILES = [
    "Generator/audio_descriptions.jsonl",
    "Generator/pretrained_models/Tokenizer_iter3_plus_AS2M.pt",
    "Generator/pretrained_models/BEATs_iter3_plus_AS2M.pt",
    "Generator/checkpoints/llm/best_model_loss_1.2393_epoch_3.pth",
    "Generator/checkpoints/flow/best_ep5_val_loss_0.0638_step164485.pt",
]

DIAGNOSER_FILES = [
    "Diagnoser/audio_descriptions.jsonl",
    "Diagnoser/pretrained_models/Tokenizer_iter3_plus_AS2M.pt",
    "Diagnoser/pretrained_models/BEATs_iter3_plus_AS2M.pt",
    "Diagnoser/checkpoints/longformer/best_longformer_loss_0.3374_epoch_2.pth",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="AustinZhang/resp-agent-models")
    parser.add_argument("--local-dir", default="external/Resp-Agent")
    parser.add_argument("--include-diagnoser", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    files = list(GENERATOR_FILES)
    if args.include_diagnoser:
        files.extend(DIAGNOSER_FILES)

    for rel in files:
        dest = local_dir / rel
        if dest.exists() and dest.stat().st_size > 1024 and not args.force:
            print(f"[skip] {rel} ({dest.stat().st_size / 1e9:.2f} GB)")
            continue
        print(f"[download] {rel}", flush=True)
        cached = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="model",
            filename=rel,
            local_files_only=False,
            resume_download=True,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(cached).resolve() != dest.resolve():
            shutil.copy2(cached, dest)
        print(f"[done] {rel} -> {dest} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
