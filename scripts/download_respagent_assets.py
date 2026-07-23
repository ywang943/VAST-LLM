#!/usr/bin/env python
"""Download official Resp-Agent model assets into external/Resp-Agent.

The official repository expects checkpoints and BEATs assets under
Generator/ and Diagnoser/.  HuggingFace stores them with the same relative
layout, so snapshot_download can place them directly into the cloned repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="AustinZhang/resp-agent-models",
        help="HuggingFace model repo containing Resp-Agent checkpoints.",
    )
    parser.add_argument(
        "--local-dir",
        default="external/Resp-Agent",
        help="Local Resp-Agent clone directory.",
    )
    parser.add_argument(
        "--generator-only",
        action="store_true",
        help="Download only the generator assets needed for waveform synthesis.",
    )
    args = parser.parse_args()

    allow_patterns = [
        "Generator/audio_descriptions.jsonl",
        "Generator/checkpoints/flow/*.pt",
        "Generator/checkpoints/llm/*.pth",
        "Generator/pretrained_models/*.pt",
    ]
    if not args.generator_only:
        allow_patterns.extend(
            [
                "Diagnoser/audio_descriptions.jsonl",
                "Diagnoser/checkpoints/longformer/*.pth",
                "Diagnoser/pretrained_models/*.pt",
            ]
        )

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    print(f"Downloaded Resp-Agent assets to {local_dir.resolve()}")


if __name__ == "__main__":
    main()
