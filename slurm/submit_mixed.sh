#!/bin/bash
# Submits the mixed-generator pipeline as one dependency chain. Run on the cluster from the repo root.
# Needs: data/items/train-5000{,-choice8}.pt (Horizon sample, tag h), data/xgen/xgen-{mistral,gemma}-{train,val}.jsonl,
#        data/xgen/xgen-{llama31-8b,qwen3-test}.jsonl (held-out generators).
set -euo pipefail
for f in data/items/train-5000.pt data/items/train-5000-choice8.pt data/xgen/xgen-mistral-train.jsonl \
         data/xgen/xgen-gemma-train.jsonl data/xgen/xgen-mistral-val.jsonl data/xgen/xgen-gemma-val.jsonl \
         data/xgen/xgen-llama31-8b.jsonl data/xgen/xgen-qwen3-test.jsonl; do
  test -f "$f" || { echo "missing $f" >&2; exit 1; }
done
MV=$(sbatch --parsable slurm/mixval.sbatch)
echo "mixval=$MV"
for d in noul7 choice8; do
  sfx=""; [ "$d" = noul7 ] || sfx="-$d"
  mi=$(sbatch --parsable --export=ALL,INPUT=data/xgen/xgen-mistral-train.jsonl,SOURCE_TAG=mi,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=mistral-$d slurm/preprocess.sbatch)
  ge=$(sbatch --parsable --export=ALL,INPUT=data/xgen/xgen-gemma-train.jsonl,SOURCE_TAG=ge,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=gemma-$d slurm/preprocess.sbatch)
  mg=$(sbatch --parsable --dependency=afterok:$mi:$ge --export=ALL,FILES="data/items/train-5000$sfx.pt+data/items/mistral-$d.pt+data/items/gemma-$d.pt",OUT=data/items/mix-$d.pt slurm/merge.sbatch)
  tr=$(sbatch --parsable -p gpu --dependency=afterok:$mg --export=ALL,ITEMS=data/items/mix-$d.pt,OUT=experiments/mix-$d,EPOCHS=2 slurm/train.sbatch)
  ev=$(sbatch --parsable -p gpu --dependency=afterok:$tr:$MV --export=ALL,MODEL=experiments/mix-$d,EVAL=data/xgen/mixed-val.jsonl,OUT=experiments/mix-$d-val,DESIGN=$d slurm/evaluate.sbatch)
  ids="preprocess=$mi,$ge merge=$mg train=$tr threshold-eval=$ev"
  for set in h:data/source/eval.jsonl llama:data/xgen/xgen-llama31-8b.jsonl qwen3:data/xgen/xgen-qwen3-test.jsonl; do
    tag=${set%%:*}; file=${set#*:}
    j=$(sbatch --parsable -p gpu --dependency=afterok:$ev --export=ALL,MODEL=experiments/mix-$d,EVAL=$file,THRESHOLDS=experiments/mix-$d-val/report.json,OUT=experiments/mix-$d-$tag,DESIGN=$d slurm/evaluate.sbatch)
    ids="$ids eval-$tag=$j"
  done
  echo "$d: $ids"
done
