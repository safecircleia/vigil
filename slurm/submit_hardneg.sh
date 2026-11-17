#!/bin/bash
# Hard-negative round: benign conversations that talk about risky topics educationally (Mistral, Gemma), merged into the
# round-4 training items, then three fresh seeds per design (runs b1..b3) scored like every other run.
set -euo pipefail
G=data/xgen
gm=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3,TAG=mistral-edu,SEED=71,PER_CATEGORY=0,BENIGN=1000,BATCH=64,SCENES=educational slurm/xgen.sbatch)
gg=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=google/gemma-2-9b-it,TAG=gemma-edu,SEED=72,PER_CATEGORY=0,BENIGN=1000,SCENES=educational slurm/xgen.sbatch)
cl=$(sbatch --parsable --dependency=afterok:$gm:$gg --export=ALL,FILES="$G/xgen-mistral-edu.jsonl+$G/xgen-gemma-edu.jsonl" slurm/clean.sbatch)
echo "generation: $gm $gg  clean: $cl"
for d in noul7 choice8; do
  p1=$(sbatch --parsable --dependency=afterok:$cl --export=ALL,INPUT=$G/xgen-mistral-edu.jsonl,SOURCE_TAG=e1,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=c-mistral-edu-$d slurm/preprocess.sbatch)
  p2=$(sbatch --parsable --dependency=afterok:$cl --export=ALL,INPUT=$G/xgen-gemma-edu.jsonl,SOURCE_TAG=e2,TRAIN_CONVERSATIONS=0,DESIGN=$d,NAME=c-gemma-edu-$d slurm/preprocess.sbatch)
  mg=$(sbatch --parsable --dependency=afterok:$p1:$p2 --export=ALL,FILES="data/items/mix4-$d.pt+data/items/c-mistral-edu-$d.pt+data/items/c-gemma-edu-$d.pt",OUT=data/items/mix5-$d.pt slurm/merge.sbatch)
  export "DEP_$d=--dependency=afterok:$mg"
done
./slurm/submit_seeds.sh b mix5 1 2 3
