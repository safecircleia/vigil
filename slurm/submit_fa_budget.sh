#!/bin/bash
# Re-scores the mixed-trained models with thresholds picked by a benign false-alarm budget (validation only).
# Needs experiments/mix-{noul7,choice8} and data/xgen/mixed-val.jsonl from submit_mixed.sh.
set -euo pipefail
MAX_FA="${MAX_FA:-0.02}"
for d in noul7 choice8; do
  ev=$(sbatch --parsable -p gpu --export=ALL,MODEL=experiments/mix-$d,EVAL=data/xgen/mixed-val.jsonl,OUT=experiments/mix-$d-vfa,DESIGN=$d,MAX_FA=$MAX_FA slurm/evaluate.sbatch)
  ids="threshold-eval=$ev"
  for set in h:data/source/eval.jsonl llama:data/xgen/xgen-llama31-8b.jsonl qwen3:data/xgen/xgen-qwen3-test.jsonl; do
    tag=${set%%:*}; file=${set#*:}
    j=$(sbatch --parsable -p gpu --dependency=afterok:$ev --export=ALL,MODEL=experiments/mix-$d,EVAL=$file,THRESHOLDS=experiments/mix-$d-vfa/report.json,OUT=experiments/mix-$d-vfa-$tag,DESIGN=$d slurm/evaluate.sbatch)
    ids="$ids eval-$tag=$j"
  done
  echo "$d: $ids"
done
