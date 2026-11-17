#!/bin/bash
#SBATCH --job-name=vigil_assets
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
# Tokenizer file + golden vectors for the Android app. Usage: sbatch --export=ALL,MODEL=...,THRESHOLDS=... slurm/export_ondevice_assets.sbatch
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen
MODEL="${MODEL:?set MODEL}"; THRESHOLDS="${THRESHOLDS:?set THRESHOLDS}"
uv run python -W ignore scripts/export_ondevice_assets.py --model "$MODEL" --thresholds "$THRESHOLDS" --out-dir "${OUT:-$MODEL/litert}" --seq-len "${SEQ_LEN:-512}"
ls -la "${OUT:-$MODEL/litert}"
