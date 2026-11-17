#!/bin/bash
#SBATCH --job-name=vigil_litert
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
# Exports a trained choice8 checkpoint to LiteRT (.tflite) for the Android app and checks it against PyTorch.
# litert-torch pins its own torch/transformers, so it gets a throwaway node-local venv instead of the project one.
# Usage: sbatch --export=ALL,MODEL=experiments/c1-choice8,THRESHOLDS=experiments/c1-choice8-greedy-val/report.json slurm/export_litert.sbatch
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
MODEL="${MODEL:?set MODEL}"; THRESHOLDS="${THRESHOLDS:?set THRESHOLDS}"
WORK="$(mktemp -d /tmp/vigil-litert-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
uv venv "$WORK/venv" --python 3.12
uv pip install --python "$WORK/venv/bin/python" litert-torch
uv pip install --python "$WORK/venv/bin/python" --no-deps "laya==0.3.21"
# onnxruntime-style memory mapping failed on NFS before (SIGBUS), so build on node-local disk, then copy out
OUT="${OUT:-$MODEL/litert}"
PYTHONPATH=. "$WORK/venv/bin/python" -W ignore scripts/export_litert.py --model "$MODEL" --thresholds "$THRESHOLDS" \
  --out-dir "$WORK/out" --seq-len "${SEQ_LEN:-512}" --rows "${ROWS:-300}" --threads "${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$OUT"
cp "$WORK"/out/* "$OUT"/
du -h "$OUT"/*
