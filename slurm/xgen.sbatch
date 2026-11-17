#!/bin/bash
#SBATCH --job-name=vigil_xgen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
MODEL="${HF_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TAG="${TAG:-llama31-8b}"
uv run python -m vigil.xgen --backend hf --hf-model "$MODEL" --out "data/xgen/xgen-${TAG}.jsonl" \
  --per-category "${PER_CATEGORY:-100}" --benign "${BENIGN:-150}" \
  --batch-size "${BATCH:-32}" --seed "${SEED:-1}" --scenes "${SCENES:-basic}" \
  ${CATEGORIES:+--categories "${CATEGORIES//+/,}"}  # separate categories with + (sbatch --export splits on commas)
