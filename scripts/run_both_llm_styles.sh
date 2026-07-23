#!/bin/bash
set -euo pipefail

# Kept under the old filename because an existing queue may call it.
# It now runs only the SpeechGPT-style route:
#   RespVoice VQ IDs -> added LLM vocabulary tokens -> diagnosis.

ROOT=/hpc2hdd/home/ywang943/KDD_Audio
PYTHON=/hpc2hdd/home/ywang943/anaconda3/envs/pytorch/bin/python
export PYTHONPATH="$ROOT:$ROOT/opera_src"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$ROOT"

CKPT=${CKPT:-checkpoints/dual_lejepa_scratch/dual_lejepa_best.pt}
VQ_CKPT=${VQ_CKPT:-checkpoints/vq/speechgpt_vq.pt}
OUT=${OUT:-checkpoints/speechgpt_style/results.json}

mkdir -p logs checkpoints/speechgpt_style checkpoints/vq

echo "[$(date '+%F %T')] Starting SpeechGPT-style experiments"

$PYTHON -u scripts/run_speechgpt_style.py \
  --encoder-ckpt "$CKPT" \
  --vq-ckpt "$VQ_CKPT" \
  --llm-name "Qwen/Qwen2.5-0.5B-Instruct" \
  --tasks icbhi_copd svd_pathology kauh_obstructive copd_severity \
  --seeds 0 1 2 \
  --codebook-size 8192 \
  --vq-steps 5000 \
  --epochs 12 \
  --batch-size 4 \
  --out "$OUT" \
  2>&1 | tee -a logs/speechgpt_style.log

echo "[$(date '+%F %T')] SpeechGPT-style experiments complete"
