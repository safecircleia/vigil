#!/bin/bash
# Trains both designs once per seed on data/items/<items-stem>-<design>.pt, then scores each run on the mixed validation
# file and the held-out sets with both threshold rules (2% budget).
# Usage: [DEP_noul7="--dependency=afterok:.."] [DEP_choice8="--dependency=afterok:.."] slurm/submit_seeds.sh <run-prefix> <items-stem> <seed>...
# DESIGNS="choice8" submits only that design (default: both). Runs are named experiments/<run-prefix><seed>-<design>[-<rule>-<set>].
set -euo pipefail
prefix=$1; stem=$2; shift 2
G=data/xgen
for d in ${DESIGNS:-noul7 choice8}; do
  depvar="DEP_$d"; dep="${!depvar:-}"
  for seed in "$@"; do
    run="$prefix$seed"
    # shellcheck disable=SC2086
    tr=$(sbatch --parsable -p gpu $dep --export=ALL,ITEMS=data/items/$stem-$d.pt,OUT=experiments/$run-$d,EPOCHS=2,SEED=$seed slurm/train.sbatch)
    for rule in precision greedy; do
      ev=$(sbatch --parsable -p gpu --dependency=afterok:$tr --export=ALL,MODEL=experiments/$run-$d,EVAL=$G/mixed-val-r4.jsonl,OUT=experiments/$run-$d-$rule-val,DESIGN=$d,MAX_FA=0.02,RULE=$rule slurm/evaluate.sbatch)
      ids="train=$tr val=$ev"
      for set in h:data/source/eval.jsonl llama:$G/xgen-llama31-8b.jsonl qwen3:$G/xgen-qwen3-test.jsonl; do
        t=${set%%:*}; file=${set#*:}
        j=$(sbatch --parsable -p gpu --dependency=afterok:$ev --export=ALL,MODEL=experiments/$run-$d,EVAL=$file,THRESHOLDS=experiments/$run-$d-$rule-val/report.json,OUT=experiments/$run-$d-$rule-$t,DESIGN=$d slurm/evaluate.sbatch)
        ids="$ids $t=$j"
      done
      echo "$run/$d/$rule: $ids"
    done
  done
done
