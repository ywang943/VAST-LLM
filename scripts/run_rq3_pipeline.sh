#!/bin/bash
set -euo pipefail

ROOT=/hpc2hdd/home/ywang943/KDD_Audio
PYTHON=/hpc2hdd/home/ywang943/anaconda3/envs/pytorch/bin/python
export PYTHONPATH="$ROOT:$ROOT/opera_src"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=0

cd "$ROOT"
mkdir -p logs checkpoints/rq3_llm

CKPT=${CKPT:-checkpoints/htsat_lejepa_full/htsat_lejepa_best.pt}
VQ_CKPT=${VQ_CKPT:-checkpoints/vq/speechgpt_vq_K512.pt}

echo "============================================================"
echo "[$(date '+%F %T')] RQ3 Pipeline Start"
echo "============================================================"

# Step 1: Prepare Coswara mel caches
echo ""
echo "[$(date '+%F %T')] Step 1: Preparing Coswara mel caches..."
$PYTHON -u data/prepare_coswara_tasks_v2.py 2>&1 | tee -a logs/rq3_coswara_prep.log

# Step 2: Download OpenBioLLM-8B if not cached
echo ""
echo "[$(date '+%F %T')] Step 2: Ensuring LLM is downloaded..."
$PYTHON -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = 'aaditya/Llama3-OpenBioLLM-8B'
print(f'Downloading/verifying {model_name}...')
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'  Tokenizer OK (vocab={len(tok)})')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
print(f'  Model OK ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)')
del model
" 2>&1 | tee -a logs/rq3_model_download.log

# Step 3: Run RQ3 with all three input modes
echo ""
echo "[$(date '+%F %T')] Step 3: Running RQ3 LLM experiments (all modes)..."
$PYTHON -u scripts/run_rq3_llm.py \
    --encoder-ckpt "$CKPT" \
    --vq-ckpt "$VQ_CKPT" \
    --llm "aaditya/Llama3-OpenBioLLM-8B" \
    --train-tasks icbhi_copd svd_pathology coswara_covid_cough coswara_smoker_cough \
    --eval-tasks icbhi_copd svd_pathology coswara_covid_cough coswara_smoker_cough \
                 kauh_obstructive copd_severity coswara_covid_breathing coswara_smoker_breathing \
    --run-all-modes \
    --epochs 5 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --lora-rank 16 \
    --max-length 512 \
    --seed 0 \
    --output checkpoints/rq3_llm/results_openbio8b.json \
    2>&1 | tee -a logs/rq3_llm_openbio8b.log

echo ""
echo "============================================================"
echo "[$(date '+%F %T')] RQ3 Pipeline Complete"
echo "============================================================"
