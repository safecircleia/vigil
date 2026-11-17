#!/bin/bash
#SBATCH --job-name=vigil_deploycal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
D="${DESIGN:?set DESIGN to noul7 or choice8}"
R="${RUN:-r4}"   # which trained run's saved scores to use
uv run python -m vigil.deploy_calib --val "experiments/$R-$D-greedy-val/rows.json" \
  --dep "llama=experiments/$R-$D-greedy-llama/rows.json" --dep "qwen3=experiments/$R-$D-greedy-qwen3/rows.json" \
  --n-cal 100 200 400 --weights 1 10 --repeats 10 --max-fa 0.02 --rule "${RULE:-greedy}"
