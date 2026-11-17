# Round 4: clean the data, retrain (2026-09-30)

**Question:** two bugs were found in round 3: the generation parser accepted content-less turns, so
some generated conversations contain a blank turn, and the first greedy rule could not clear benign
rows flagged by two categories. Both are fixed in code. Do the results change once the affected data is
cleaned and both designs are retrained?

**Short answer:** the conclusions do not change, and the size of the changes between round 3 and round
4 shows how much run-to-run noise these comparisons carry. Differences of a few points between
configurations in rounds 2-3 should be treated as inside that noise.

## What was done

- `vigil/clean_xgen.py` drops generated rows with an empty turn (originals kept as `<file>.orig`):
  40 training rows (1 Mistral, 32 Gemma, 7 Gemma safety-scene; 0.5% of the 7,396 generated training
  conversations) and 8 Gemma validation rows. Both held-out sets (Llama, Qwen3) had none.
- Both designs retrained from scratch on the cleaned data with the same recipe (2 epochs), and the
  validation file rebuilt from the cleaned validation files (2,757 rows; 405 benign in the validation
  half).
- Both threshold rules (precision-target budget; the fixed greedy rule), 2% budget, applied once to
  the same held-out sets.

## Results: round 3 (old data) to round 4 (clean data)

Benign false alarms / catch rate for a correct category.

| Config | Horizon (in-dist) | Llama | Qwen3 |
|---|---|---|---|
| noul7, precision rule | 0.3% / 95.7% to 0.3% / 94.9% | 3.4% / 81.4% to 4.1% / 78.2% | 12.7% / 82.7% to 6.7% / 77.6% |
| noul7, greedy | 0.9% / 97.6% to 1.2% / 97.7% | 6.8% / 84.4% to 12.2% / 83.9% | 24.0% / 88.6% to 20.7% / 86.6% |
| choice8, precision rule | 0.9% / 96.8% to 0.8% / 94.8% | 12.2% / 84.4% to 8.8% / 80.7% | 12.7% / 88.9% to 10.0% / 79.4% |
| choice8, greedy | 1.1% / 98.2% to 0.9% / 97.4% | 12.9% / 87.9% to 11.6% / 85.1% | 18.0% / 90.4% to 16.7% / 88.3% |

Validation false alarms: noul7 precision 3/405 (0.7%, target 0.95), greedy 8/405 (2.0%); choice8
precision 5/405 (1.2%, target 0.95, was 0.9), greedy 8/405 (2.0%).

## What this shows

- **No consistent improvement from cleaning.** Held-out false alarms move by between -6.0 and +5.4
  points (noul7 precision on Qwen3 12.7% to 6.7%; noul7 greedy on Llama 6.8% to 12.2%). The catch rate
  for the correct category is lower in 11 of 12 cells (by 0.5 to 9.5 points; choice8 precision on
  Qwen3 88.9% to 79.4%) and unchanged in one.
- **These swings are the finding.** Removing 0.5% of the generated training rows and retraining moved
  results by as much as most of the differences between configurations in `round2.md` and
  `round3.md`. Those differences (a few points either way) are therefore not reliable.
- I cannot say whether the consistent drop in catch rate comes from cleaning (the removed rows could
  have been useful) or from training variance: each design was retrained once, so there are only two
  independent draws.
- The fixed code and cleaned data are the correct base going forward; the earlier reported numbers
  came from the older data and code and stay valid as history only.

## What it does not show

- Which design or rule is best. With this much variance, none of the four configurations is separated
  from the others by the data collected so far.
- Anything about real conversations.

## Suggested next steps

1. **Measure the variance directly:** train each design with 3 seeds on the same data and report the
   spread of every headline number. This is cheap (about 30 minutes per noul7 run, 5 per choice8)
   and it decides how much of any future difference can be believed.
2. Then decide on the operating point (false alarms versus missed risky chats) with confidence
   intervals that include training variance.
3. Real or human-written conversations remain the deciding test.
