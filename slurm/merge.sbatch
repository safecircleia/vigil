#!/bin/bash
#SBATCH --job-name=vigil_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
FILES="${FILES:?set FILES to item files separated by + (sbatch --export splits on commas)}"
OUT="${OUT:?set OUT to the merged item file}"
# shellcheck disable=SC2086
uv run python -m vigil.merge_items ${FILES//+/ } --output "$OUT"
