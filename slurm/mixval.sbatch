#!/bin/bash
#SBATCH --job-name=vigil_mixval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
GENERATED="${GENERATED:-data/xgen/xgen-mistral-val.jsonl+data/xgen/xgen-gemma-val.jsonl}"
OUT="${OUT:-data/xgen/mixed-val.jsonl}"
# shellcheck disable=SC2086
uv run python -m vigil.mixval --generated ${GENERATED//+/ } --out "$OUT" --n-horizon 900
