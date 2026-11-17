#!/bin/bash
#SBATCH --job-name=vigil_train
#SBATCH --partition=gpuUltra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
ITEMS="${ITEMS:?set ITEMS via --export=ALL,ITEMS=...}"
OUT="${OUT:-experiments/vigil-${SLURM_JOB_ID}}"
BASE_REVISION="${BASE_REVISION:?set BASE_REVISION via --export=ALL,BASE_REVISION=<base model commit sha>}"
uv run python -m vigil.train --items "$ITEMS" --output-dir "$OUT" --base-revision "$BASE_REVISION" --epochs "${EPOCHS:-4}" ${SEED:+--seed "$SEED"}
