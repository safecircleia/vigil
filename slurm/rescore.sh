#!/bin/bash
# Re-scores the round-2 models with thresholds picked on a validation file by one rule, then on the held-out sets.
# Usage: slurm/rescore.sh <tag> <precision|greedy> <validation.jsonl> [max_fa]
set -euo pipefail
tag=$1; rule=$2; val=$3; fa=${4:-0.02}
for d in noul7 choice8; do
  ev=$(sbatch --parsable -p gpu --export=ALL,MODEL=experiments/r2-$d,EVAL=$val,OUT=experiments/$tag-$d-$rule-val,DESIGN=$d,MAX_FA=$fa,RULE=$rule slurm/evaluate.sbatch)
  ids="val=$ev"
  for set in h:data/source/eval.jsonl llama:data/xgen/xgen-llama31-8b.jsonl qwen3:data/xgen/xgen-qwen3-test.jsonl; do
    t=${set%%:*}; file=${set#*:}
    j=$(sbatch --parsable -p gpu --dependency=afterok:$ev --export=ALL,MODEL=experiments/r2-$d,EVAL=$file,THRESHOLDS=experiments/$tag-$d-$rule-val/report.json,OUT=experiments/$tag-$d-$rule-$t,DESIGN=$d slurm/evaluate.sbatch)
    ids="$ids $t=$j"
  done
  echo "$d/$rule: $ids"
done
