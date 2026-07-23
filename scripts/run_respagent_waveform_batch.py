#!/usr/bin/env python
"""Batch wrapper for the official Resp-Agent waveform generator."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="external/Resp-Agent")
    parser.add_argument(
        "--config",
        default="checkpoints/respagent_waveform/configs/generator_config.yaml",
        help="Local generator config with workspace paths.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ref-audios",
        nargs="+",
        default=["external/Resp-Agent/Generator/wav/reference_audio.wav"],
        help="Reference wav files whose style will condition synthesis.",
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=["COPD"],
        help="Disease/content prompts to synthesize.",
    )
    parser.add_argument("--out-dir", default="checkpoints/respagent_waveform/generated")
    parser.add_argument("--ref-ratio", type=float, default=0.30)
    parser.add_argument("--max-new-tokens-pad", type=int, default=8)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    generator_dir = repo_dir / "Generator"
    config = Path(args.config).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for disease in args.diseases:
        safe_disease = disease.lower().replace(" ", "_").replace("/", "_")
        for i, ref_audio in enumerate(args.ref_audios):
            ref_path = Path(ref_audio).resolve()
            run_dir = out_root / safe_disease / f"ref_{i:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python",
                "generator_pipeline.py",
                "--config",
                str(config),
                "--device",
                args.device,
                "--ref_audio",
                str(ref_path),
                "--out_dir",
                str(run_dir),
                "--disease",
                disease,
                "--ref_ratio",
                str(args.ref_ratio),
                "--max_new_tokens_pad",
                str(args.max_new_tokens_pad),
            ]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(generator_dir), check=True)
            wav_path = run_dir / "generated_from_llm_cfm.wav"
            manifest.append(
                {
                    "disease": disease,
                    "reference_audio": str(ref_path),
                    "generated_wav": str(wav_path),
                    "tokens_json": str(run_dir / "pred_beats_tokens.json"),
                }
            )

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
