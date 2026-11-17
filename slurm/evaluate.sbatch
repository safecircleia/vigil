#!/bin/bash
#SBATCH --job-name=vigil_eval
#SBATCH --partition=gpuUltra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
MODEL="${MODEL:?set MODEL via --export=ALL,MODEL=...}"
uv run python -m vigil.evaluate --model-dir "$MODEL" ${MODEL_REVISION:+--revision "$MODEL_REVISION"} --eval "${EVAL:-data/source/eval.jsonl}" \
  --out-dir "${OUT:-experiments/eval-${SLURM_JOB_ID}}" --max-rows "${MAX_ROWS:-5000}" --device cuda --design "${DESIGN:-noul7}" ${THRESHOLDS:+--thresholds-from "$THRESHOLDS"} ${MAX_FA:+--max-benign-fa "$MAX_FA"} ${RULE:+--threshold-rule "$RULE"}
