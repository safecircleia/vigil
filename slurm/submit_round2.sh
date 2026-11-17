#!/bin/bash
# Round 2: more benign banter/safety-advice training data, and a validation-only generator (Falcon3).
# Reuses data/items/{train-5000,mistral,gemma}-*.pt and the held-out sets from round 1.
set -euo pipefail
for f in data/items/train-5000.pt data/items/train-5000-choice8.pt data/items/mistral-noul7.pt data/items/gemma-noul7.pt \
         data/items/mistral-choice8.pt data/items/gemma-choice8.pt data/xgen/xgen-mistral-val.jsonl data/xgen/xgen-gemma-val.jsonl \
         data/xgen/xgen-llama31-8b.jsonl data/xgen/xgen-qwen3-test.jsonl; do
  test -f "$f" || { echo "missing $f" >&2; exit 1; }
done
C=grooming+bullying+isolation+personal_info+platform_migration+threats
gm=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3,TAG=mistral-safety,SEED=31,PER_CATEGORY=0,BENIGN=1000,BATCH=64,SCENES=safety slurm/xgen.sbatch)
gg=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=google/gemma-2-9b-it,TAG=gemma-safety,SEED=32,PER_CATEGORY=0,BENIGN=1000,SCENES=safety slurm/xgen.sbatch)
# Falcon3 writes validation data only: never trained on. Mixed scenes (extended + safety benign) via two files.
ga=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=tiiuae/Falcon3-7B-Instruct,TAG=falcon-val,SEED=41,PER_CATEGORY=50,BENIGN=90,CATEGORIES=$C,SCENES=extended slurm/xgen.sbatch)
gs=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=tiiuae/Falcon3-7B-Instruct,TAG=falcon-val-safety,SEED=42,PER_CATEGORY=0,BENIGN=60,SCENES=safety slurm/xgen.sbatch)
echo "generation: mistral-safety=$gm gemma-safety=$gg falcon-val=$ga falcon-val-safety=$gs"
mv=$(sbatch --parsable --dependency=afterok:$ga:$gs --export=ALL,GENERATED="data/xgen/xgen-mistral-val.jsonl+data/xgen/xgen-gemma-val.jsonl+data/xgen/xgen-falcon-val.jsonl+data/xgen/xgen-falcon-val-safety.jsonl",OUT=data/xgen/mixed-val-r2.jsonl slurm/mixval.sbatch)
for d in noul7 choice8; do
  sfx=""; [ "$d" = noul7 ] || sfx="-$d"
  mb=$(sbatch --parsable --dependency=afterok:$gm --export=ALL,INPUT=data/xgen/xgen-mistral-safety.jsonl,SOURCE_TAG=ms,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=mistral-safety-$d slurm/preprocess.sbatch)
  gb=$(sbatch --parsable --dependency=afterok:$gg --export=ALL,INPUT=data/xgen/xgen-gemma-safety.jsonl,SOURCE_TAG=gs,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=gemma-safety-$d slurm/preprocess.sbatch)
  mg=$(sbatch --parsable --dependency=afterok:$mb:$gb --export=ALL,FILES="data/items/train-5000$sfx.pt+data/items/mistral-$d.pt+data/items/gemma-$d.pt+data/items/mistral-safety-$d.pt+data/items/gemma-safety-$d.pt",OUT=data/items/mix2-$d.pt slurm/merge.sbatch)
  tr=$(sbatch --parsable -p gpu --dependency=afterok:$mg --export=ALL,ITEMS=data/items/mix2-$d.pt,OUT=experiments/r2-$d,EPOCHS=2 slurm/train.sbatch)
  ev=$(sbatch --parsable -p gpu --dependency=afterok:$tr:$mv --export=ALL,MODEL=experiments/r2-$d,EVAL=data/xgen/mixed-val-r2.jsonl,OUT=experiments/r2-$d-vfa,DESIGN=$d,MAX_FA=0.02 slurm/evaluate.sbatch)
  ids="train=$tr threshold-eval=$ev"
  for set in h:data/source/eval.jsonl llama:data/xgen/xgen-llama31-8b.jsonl qwen3:data/xgen/xgen-qwen3-test.jsonl; do
    tag=${set%%:*}; file=${set#*:}
    j=$(sbatch --parsable -p gpu --dependency=afterok:$ev --export=ALL,MODEL=experiments/r2-$d,EVAL=$file,THRESHOLDS=experiments/r2-$d-vfa/report.json,OUT=experiments/r2-$d-vfa-$tag,DESIGN=$d slurm/evaluate.sbatch)
    ids="$ids eval-$tag=$j"
  done
  echo "$d: $ids"
done
