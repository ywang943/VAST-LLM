#!/bin/bash
# RQ3: Full cross-domain evaluation with all tasks + SVD + KAUH
# Uses correct VQ (no-L2 EMA) and correct encoder (v3_full)
# Runs audio_text, audio_only, text_only modes sequentially

set -e
cd /hpc2hdd/home/ywang943/KDD_Audio
PYTHON=/hpc2hdd/home/ywang943/anaconda3/envs/pytorch/bin/python

ENCODER=checkpoints/htsat_lejepa_v3_full/htsat_lejepa_best.pt
VQ=checkpoints/vq/mel_htsat_v3_full_vq_K512_ema.pt

# Full task set matching paper Tables 2 and 4
TRAIN_TASKS="icbhi_copd svd_pathology coswara_covid_cough coswara_smoker_cough b2ai_voice_pathology"
EVAL_TASKS="icbhi_copd svd_pathology coswara_covid_cough coswara_smoker_cough b2ai_voice_pathology kauh_obstructive copd_severity coswara_covid_breathing coswara_smoker_breathing b2ai_laryngeal_cancer b2ai_benign_lesions b2ai_laryngeal_dystonia"

echo "=== RQ3 Full Run ==="
echo "Encoder: $ENCODER"
echo "VQ: $VQ"
echo "Train tasks: $TRAIN_TASKS"
echo "Eval tasks: $EVAL_TASKS"

$PYTHON -u scripts/run_rq3_llm.py \
  --encoder-ckpt $ENCODER \
  --vq-ckpt $VQ \
  --llm Qwen/Qwen2.5-0.5B-Instruct \
  --train-tasks $TRAIN_TASKS \
  --eval-tasks $EVAL_TASKS \
  --run-all-modes \
  --epochs 5 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --lora-rank 16 \
  --max-length 512 \
  --seed 0 \
  --output checkpoints/rq3_llm/rq3_full_all_modes.json

echo "=== Done ==="
