#!/bin/bash
#SBATCH --job-name=vigil_fetch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
REVISION="${REVISION:?set REVISION via --export=ALL,REVISION=<dataset commit sha>}"
uv run python -m vigil.fetch_data --revision "$REVISION"
wc -l data/source/train.jsonl data/source/eval.jsonl
