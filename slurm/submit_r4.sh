#!/bin/bash
# Round 4: same recipe as round 2/3 on data cleaned of empty-turn rows, retrained, both threshold rules.
# Needs the generated files from rounds 1-3 and data/items/train-5000{,-choice8}.pt (Horizon sample).
set -euo pipefail
G=data/xgen
TRAINSET="$G/xgen-mistral-train.jsonl+$G/xgen-gemma-train.jsonl+$G/xgen-mistral-safety.jsonl+$G/xgen-gemma-safety.jsonl"
VALSET="$G/xgen-mistral-val.jsonl+$G/xgen-gemma-val.jsonl+$G/xgen-phi-val.jsonl+$G/xgen-phi-val-safety.jsonl+$G/xgen-smol-val.jsonl+$G/xgen-smol-val-safety.jsonl"
cl=$(sbatch --parsable --export=ALL,FILES="$TRAINSET+$VALSET" slurm/clean.sbatch)
echo "clean: $cl"
mv=$(sbatch --parsable --dependency=afterok:$cl --export=ALL,GENERATED="$VALSET",OUT=$G/mixed-val-r4.jsonl slurm/mixval.sbatch)
for d in noul7 choice8; do
  sfx=""; [ "$d" = noul7 ] || sfx="-$d"
  p1=$(sbatch --parsable --dependency=afterok:$cl --export=ALL,INPUT=$G/xgen-mistral-train.jsonl,SOURCE_TAG=mi,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=c-mistral-$d slurm/preprocess.sbatch)
  p2=$(sbatch --parsable --dependency=afterok:$cl --export=ALL,INPUT=$G/xgen-gemma-train.jsonl,SOURCE_TAG=ge,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=c-gemma-$d slurm/preprocess.sbatch)
  p3=$(sbatch --parsable --dependency=afterok:$cl --export=ALL,INPUT=$G/xgen-mistral-safety.jsonl,SOURCE_TAG=ms,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=c-mistral-safety-$d slurm/preprocess.sbatch)
  p4=$(sbatch --parsable --dependency=afterok:$cl --export=ALL,INPUT=$G/xgen-gemma-safety.jsonl,SOURCE_TAG=gs,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=c-gemma-safety-$d slurm/preprocess.sbatch)
  mg=$(sbatch --parsable --dependency=afterok:$p1:$p2:$p3:$p4 --export=ALL,FILES="data/items/train-5000$sfx.pt+data/items/c-mistral-$d.pt+data/items/c-gemma-$d.pt+data/items/c-mistral-safety-$d.pt+data/items/c-gemma-safety-$d.pt",OUT=data/items/mix4-$d.pt slurm/merge.sbatch)
  tr=$(sbatch --parsable -p gpu --dependency=afterok:$mg --export=ALL,ITEMS=data/items/mix4-$d.pt,OUT=experiments/r4-$d,EPOCHS=2 slurm/train.sbatch)
  for rule in precision greedy; do
    ev=$(sbatch --parsable -p gpu --dependency=afterok:$tr:$mv --export=ALL,MODEL=experiments/r4-$d,EVAL=$G/mixed-val-r4.jsonl,OUT=experiments/r4-$d-$rule-val,DESIGN=$d,MAX_FA=0.02,RULE=$rule slurm/evaluate.sbatch)
    ids="train=$tr val=$ev"
    for set in h:data/source/eval.jsonl llama:$G/xgen-llama31-8b.jsonl qwen3:$G/xgen-qwen3-test.jsonl; do
      t=${set%%:*}; file=${set#*:}
      j=$(sbatch --parsable -p gpu --dependency=afterok:$ev --export=ALL,MODEL=experiments/r4-$d,EVAL=$file,THRESHOLDS=experiments/r4-$d-$rule-val/report.json,OUT=experiments/r4-$d-$rule-$t,DESIGN=$d slurm/evaluate.sbatch)
      ids="$ids $t=$j"
    done
    echo "$d/$rule: $ids"
  done
done
