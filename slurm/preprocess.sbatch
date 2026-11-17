#!/bin/bash
#SBATCH --job-name=vigil_preprocess
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
INPUT="${INPUT:-data/source/train.jsonl}"
TAG="${SOURCE_TAG:-h}"
N="${TRAIN_CONVERSATIONS:-5000}"
DESIGN="${DESIGN:-noul7}"
SUFFIX=""; [ "$DESIGN" = "noul7" ] || SUFFIX="-$DESIGN"
NAME="${NAME:-train-${N}${SUFFIX}}"
BASE_REVISION="${BASE_REVISION:?set BASE_REVISION via --export=ALL,BASE_REVISION=<base model commit sha>}"
uv run python -m vigil.preprocess \
  --input "$INPUT" \
  --output "data/items/${NAME}.pt" \
  --report "data/items/${NAME}.report.json" \
  --base-revision "$BASE_REVISION" \
  --train-conversations "$N" --design "$DESIGN" --source-tag "$TAG"
