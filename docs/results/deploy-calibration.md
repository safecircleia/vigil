# Per-deployment calibration (2026-09-30)

**Question:** Vigil ranks well (AUC 0.97-1.00) but its thresholds do not transfer to text from a new
source (`round4.md`, `openjev-zero-shot.md`). If the thresholds also see a small sample of the target
traffic, do the false alarms come down?

**Short answer:** not enough. A few hundred labelled rows of the target traffic cut false alarms by
0-8 points at best, and every real reduction costs about as much recall. No setting comes near the 2%
budget. The reason is in the scores, not the thresholds: some benign chats simply score like risky ones.

## How it was tested (offline, on saved scores; no model was run)

The round-4 models (`r4-noul7`, `r4-choice8`) already scored the validation file and the two unseen
generators. For each generator in turn, treated as "the deployment":

- baseline thresholds: greedy rule, 2% benign false-alarm budget, picked on all validation rows;
- calibrated thresholds: the same rule on validation plus `n_cal` random deployment rows (100, 200 or 400)
  repeated `weight` times (1 = plain pooling, 10 = the sample counts ten times as much);
- both measured on the deployment rows that were not sampled; 10 random splits per setting.

Code: `vigil/deploy_calib.py`, run by `slurm/deploy_calib.sbatch`.

## Results (mean over 10 splits; paired change against the baseline on the same held-out part)

Benign false alarms / catch rate for the correct category.

| Model, deployment | Baseline (validation only) | 400 rows, weight 1 | 400 rows, weight 10 |
|---|---|---|---|
| 8-question, Llama | 10.8% / 83.7% | 8.8% (-2.4) / 83.5% (-0.5) | 7.4% (-3.9) / 82.8% (-1.2) |
| 8-question, Qwen3 | 14.2% / 84.4% | 12.6% (-1.4) / 82.9% (-1.9) | 8.5% (-5.6) / 78.2% (-6.7) |
| 2-pass, Llama | 12.0% / 85.9% | 13.3% (+0.0) / 84.7% (-1.5) | 12.0% (-1.3) / 82.9% (-3.3) |
| 2-pass, Qwen3 | 16.2% / 87.8% | 16.3% (-0.1) / 87.5% (-0.9) | 8.1% (-8.3) / 79.3% (-9.0) |

Smaller samples (100 and 200 rows) do less: false-alarm changes of -0.2 to -4.7 points. Split-to-split
spread is 2-5 points, so most of these changes are within noise.

## What this shows

- **Plain pooling does almost nothing.** Adding 100-400 deployment rows to 2,700 validation rows barely
  moves the thresholds.
- **Upweighting helps a little, then trades recall one for one.** The best case (2-pass, Qwen3, 400 rows,
  weight 10) halves the false alarms (16.2% to 8.1%) and loses 9 points of catch.
- **The floor is in the scores.** The earlier audit (`false-alarm-audit.md`) found that 3-11% of benign
  chats get a *confident* flag (top probability of at least 0.5), typically insulting banter or safety
  advice full of risk words. A threshold cannot remove those without also removing true positives that
  score similarly, so no amount of calibration data gets to 2%.
- So the threshold-transfer problem is real but it is not the whole story: after thresholds, the
  remaining false alarms need a better model or a second opinion.

## What it does not show

- Behaviour with a much larger calibration sample, or with a real deployment (both "deployments" here are
  LLM-written text and the calibration rows come from the same finite set as the test rows).
- Anything about other threshold rules; only the greedy rule was tried.
- Repeated splits reuse the same rows, so the spread understates true uncertainty.

## Suggested next steps

1. Attack the confident false alarms from the model side: more benign hard negatives of exactly the kinds
   the audit found (insulting-but-friendly banter, safety advice), and a decision on whether such banter
   should count as bullying at all (a product policy choice).
2. Measure training noise first (3 seeds per design), since round 4 showed it is as large as most of the
   differences between configurations.
3. A second-stage confirmer would cut false alarms (the OpenJev cascade did), but that route was dropped
   for licence and cost reasons; any other confirmer would need to be permissively licensed.
4. Real or human-written conversations remain the deciding test.
