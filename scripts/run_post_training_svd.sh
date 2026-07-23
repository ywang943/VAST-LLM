#!/bin/bash
set -euo pipefail

ROOT=/hpc2hdd/home/ywang943/KDD_Audio
PYTHON=/hpc2hdd/home/ywang943/anaconda3/envs/pytorch/bin/python
MAIN_PIPELINE_PID=${MAIN_PIPELINE_PID:-3615775}
SVD_PREP_PID=${SVD_PREP_PID:-3707145}
CHECKPOINT=checkpoints/htsat_lejepa_scratch_clean/htsat_lejepa_best.pt
LOG=logs/post_training_svd.log

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/opera_src"
export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }
wait_for_pid() {
    local pid=$1
    local name=$2
    while kill -0 "$pid" 2>/dev/null; do
        log "waiting for $name (PID=$pid)"
        sleep 300
    done
}

wait_for_pid "$SVD_PREP_PID" "complete SVD cache"
if [ ! -s data/mel_cache/svd_full/metadata.json ]; then
    log "SVD cache metadata missing; rebuilding"
    "$PYTHON" -u data/prepare_svd_full.py >> "$LOG" 2>&1
fi

wait_for_pid "$MAIN_PIPELINE_PID" "scratch pretraining and CSAF benchmarks"
if [ ! -s "$CHECKPOINT" ]; then
    log "missing scratch checkpoint: $CHECKPOINT"
    exit 1
fi

# The main pipeline normally creates these. Re-run only if an earlier stage
# failed before writing its final JSON result.
if [ ! -s checkpoints/frozen_multitask/scratch_clean_results.json ]; then
    log "running/recovering frozen multi-task benchmark"
    "$PYTHON" -u scripts/run_frozen_multitask_benchmark.py \
        --checkpoint "$CHECKPOINT" \
        --output checkpoints/frozen_multitask/scratch_clean_results.json \
        >> "$LOG" 2>&1
fi

if [ ! -s checkpoints/csaf_multitask_adapt/scratch_clean_results.json ]; then
    log "running/recovering task-adapted CSAF benchmark"
    "$PYTHON" -u scripts/run_multitask_csaf_adaptation.py \
        --checkpoint "$CHECKPOINT" \
        --output checkpoints/csaf_multitask_adapt/scratch_clean_results.json \
        >> "$LOG" 2>&1
fi

log "running complete-SVD multi-source evaluation with scratch LeJEPA"
"$PYTHON" -u scripts/run_svd_mvp_task.py \
    --encoder checkpoint --ckpt "$CHECKPOINT" \
    --out checkpoints/svd_mvp/svd_full_scratch_results.json \
    >> "$LOG" 2>&1

log "running complete-SVD multi-source OPERA-CT reference"
"$PYTHON" -u scripts/run_svd_mvp_task.py \
    --encoder opera_ct \
    --out checkpoints/svd_mvp/svd_full_opera_ct_results.json \
    >> "$LOG" 2>&1

log "running prototypical eval — scratch LeJEPA (stage4 + tpa_csaf)"
"$PYTHON" -u scripts/run_prototypical_eval.py \
    --encoder checkpoint --ckpt "$CHECKPOINT" \
    --out checkpoints/prototypical/scratch_results.json \
    >> "$LOG" 2>&1

log "running prototypical eval — OPERA-CT baseline (stage4 only)"
"$PYTHON" -u scripts/run_prototypical_eval.py \
    --encoder opera_ct \
    --out checkpoints/prototypical/opera_ct_results.json \
    >> "$LOG" 2>&1

log "all evaluations finished"
