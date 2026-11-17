#!/bin/bash
# Round 3: no retraining. Validation-only data from two unseen generators that write risky chats (Phi-3.5, SmolLM3),
# then two threshold rules (precision-target budget vs greedy per-category) on the same held-out sets.
# Needs experiments/r2-{noul7,choice8} and the round 1/2 data.
set -euo pipefail
C=grooming+bullying+isolation+personal_info+platform_migration+threats
g1=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=microsoft/Phi-3.5-mini-instruct,TAG=phi-val,SEED=61,PER_CATEGORY=50,BENIGN=150,BATCH=48,CATEGORIES=$C,SCENES=extended slurm/xgen.sbatch)
g2=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=microsoft/Phi-3.5-mini-instruct,TAG=phi-val-safety,SEED=63,PER_CATEGORY=0,BENIGN=60,BATCH=48,SCENES=safety slurm/xgen.sbatch)
g3=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=HuggingFaceTB/SmolLM3-3B,TAG=smol-val,SEED=62,PER_CATEGORY=50,BENIGN=150,BATCH=48,CATEGORIES=$C,SCENES=extended slurm/xgen.sbatch)
g4=$(sbatch --parsable -p gpu --export=ALL,HF_MODEL=HuggingFaceTB/SmolLM3-3B,TAG=smol-val-safety,SEED=64,PER_CATEGORY=0,BENIGN=60,BATCH=48,SCENES=safety slurm/xgen.sbatch)
echo "generation: $g1 $g2 $g3 $g4"
mv=$(sbatch --parsable --dependency=afterok:$g1:$g2:$g3:$g4 --export=ALL,GENERATED="data/xgen/xgen-mistral-val.jsonl+data/xgen/xgen-gemma-val.jsonl+data/xgen/xgen-phi-val.jsonl+data/xgen/xgen-phi-val-safety.jsonl+data/xgen/xgen-smol-val.jsonl+data/xgen/xgen-smol-val-safety.jsonl",OUT=data/xgen/mixed-val-r3.jsonl slurm/mixval.sbatch)
echo "mixval: $mv"
for d in noul7 choice8; do
  for rule in precision greedy; do
    ev=$(sbatch --parsable -p gpu --dependency=afterok:$mv --export=ALL,MODEL=experiments/r2-$d,EVAL=data/xgen/mixed-val-r3.jsonl,OUT=experiments/r3-$d-$rule-val,DESIGN=$d,MAX_FA=0.02,RULE=$rule slurm/evaluate.sbatch)
    ids="val=$ev"
    for set in h:data/source/eval.jsonl llama:data/xgen/xgen-llama31-8b.jsonl qwen3:data/xgen/xgen-qwen3-test.jsonl; do
      tag=${set%%:*}; file=${set#*:}
      j=$(sbatch --parsable -p gpu --dependency=afterok:$ev --export=ALL,MODEL=experiments/r2-$d,EVAL=$file,THRESHOLDS=experiments/r3-$d-$rule-val/report.json,OUT=experiments/r3-$d-$rule-$tag,DESIGN=$d slurm/evaluate.sbatch)
      ids="$ids $tag=$j"
    done
    echo "$d/$rule: $ids"
  done
done
