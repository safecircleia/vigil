#!/bin/bash
# Friendly-banter round: benign teasing conversations (Mistral, Gemma) merged into the hard-negative training items
# (mix5), then three fresh seeds per design (runs c1..c3) scored like every other run.
set -euo pipefail
G=data/xgen
gm=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3,TAG=mistral-banter,SEED=81,PER_CATEGORY=0,BENIGN=1000,BATCH=64,SCENES=banter slurm/xgen.sbatch)
gg=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=google/gemma-2-9b-it,TAG=gemma-banter,SEED=82,PER_CATEGORY=0,BENIGN=1000,SCENES=banter slurm/xgen.sbatch)
cl=$(sbatch --parsable --dependency=afterok:$gm:$gg --export=ALL,FILES="$G/xgen-mistral-banter.jsonl+$G/xgen-gemma-banter.jsonl" slurm/clean.sbatch)
echo "generation: $gm $gg  clean: $cl"
for d in noul7 choice8; do
  p1=$(sbatch --parsable --dependency=afterok:$cl --export=ALL,INPUT=$G/xgen-mistral-banter.jsonl,SOURCE_TAG=t1,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=c-mistral-banter-$d slurm/preprocess.sbatch)
  p2=$(sbatch --parsable --dependency=afterok:$cl --export=ALL,INPUT=$G/xgen-gemma-banter.jsonl,SOURCE_TAG=t2,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=c-gemma-banter-$d slurm/preprocess.sbatch)
  mg=$(sbatch --parsable --dependency=afterok:$p1:$p2 --export=ALL,FILES="data/items/mix5-$d.pt+data/items/c-mistral-banter-$d.pt+data/items/c-gemma-banter-$d.pt",OUT=data/items/mix6-$d.pt slurm/merge.sbatch)
  export "DEP_$d=--dependency=afterok:$mg"
done
./slurm/submit_seeds.sh c mix6 1 2 3
