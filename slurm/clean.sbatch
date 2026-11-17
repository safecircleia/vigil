#!/bin/bash
#SBATCH --job-name=vigil_clean
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
FILES="${FILES:?set FILES to generated files separated by +}"
# shellcheck disable=SC2086
uv run python -m vigil.clean_xgen ${FILES//+/ }
