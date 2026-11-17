# Round 3: validation with unseen risky chats, and per-category thresholds (2026-09-30)

**Questions.** `round2.md` left two problems: the validation generator (Falcon3) refused most risky
requests, so thresholds were set almost only on benign evidence and cost recall; and one global
precision target treats every category alike. (a) Does validation data from unseen generators that
*do* write risky chats fix the recall loss? (b) Does choosing thresholds per category do better than a
single precision target?

**Short answer:** (a) yes, recall comes back. (b) it buys more recall but at a higher false-alarm
rate, so it is a different operating point, not a free improvement. And the main lesson: **a 2%
false-alarm budget on validation does not control the false-alarm rate on a new generator** (the same
1.5-1.8% validation budget gave 3% to 24% on the held-out sets).

## What was done

No retraining: the round-2 models (`r2-noul7`, `r2-choice8`) are unchanged.

- **New validation-only generators:** Phi-3.5-mini-instruct (510 of 510 requests kept) and
  SmolLM3-3B (459 of 510 kept). Both wrote every category, in both languages. I first probed OLMo-2,
  SmolLM3 and Phi-3.5 on 48 requests each; OLMo's output was unusable (empty turns), so it was dropped.
- **Validation set:** 900 Horizon eval rows that the test sample never uses, plus Mistral (450),
  Gemma (450), Phi (510) and SmolLM3 (459) validation rows: 2,765 rows. Thresholds are picked on half
  of it (398 benign rows in that half).
- **Two threshold rules**, both with a 2% benign false-alarm budget, applied once to the held-out sets
  (Horizon sample, Llama 831, Qwen3 850):
  - *precision rule* (round 1/2): smallest precision target in {0.8, 0.9, 0.95, 0.98} that fits.
    It picked 0.95 for noul7 and 0.9 for choice8 (validation false alarms 6/398 = 1.5%).
  - *greedy rule* (new, `pick_thresholds_greedy`): per-category thresholds that keep as much
    validation recall as possible within the budget, giving up the cheapest recall first.
    Validation false alarms 7/398 = 1.8% for both designs.

Two bugs found and fixed on the way:

- The conversation parser accepted turns that were only markdown noise (OLMo's empty turns). Fixed.
  About 1% of the Gemma-written training data has such a turn (32 of 2,700 train, 7 of 1,000 safety, 8
  of 450 validation) and a few Falcon rows; both held-out sets have none. I did not retrain over it.
- The first greedy rule could not clear benign rows flagged by two categories, so on choice8 it stopped
  at 3.0% validation false alarms ("budget not met"). Fixed with a fractional-credit rule; only the
  fixed results are reported.

## Results (benign false alarms / catch rate for a correct category)

| Config | Horizon (in-dist) | Llama | Qwen3 |
|---|---|---|---|
| noul7, precision rule, round 2 (Falcon validation) | 0.2% / 93.4% | 2.0% / 77.0% | 1.3% / 76.3% |
| **noul7, precision rule** | 0.3% (2/654) / 95.7% | 3.4% (5/147) / 81.4% | 12.7% (19/150) / 82.7% |
| **noul7, greedy** | 0.9% (6/654) / 97.6% | 6.8% (10/147) / 84.4% | 24.0% (36/150) / 88.6% |
| choice8, precision rule, round 2 | 0.3% / 96.0% | 10.2% / 82.2% | 11.3% / 86.1% |
| **choice8, precision rule** | 0.9% (6/654) / 96.8% | 12.2% (18/147) / 84.4% | 12.7% (19/150) / 88.9% |
| **choice8, greedy** | 1.1% (7/654) / 98.2% | 12.9% (19/147) / 87.9% | 18.0% (27/150) / 90.4% |

Recall of the hardest categories on unseen sets (noul7, precision rule, round 2 to round 3): bullying
on Qwen3 58% to 87%, threats on Llama 65% to 81%, grooming on Qwen3 89% to 93%.

## What this shows

- **Validation positives fix the recall collapse.** With Phi and SmolLM3 in validation, noul7's catch
  rate on unseen sets rises from about 77% to about 82%, and the bullying/threats losses mostly reverse.
- **The false-alarm budget does not transfer reliably.** Same validation budget (1.5-1.8%), but
  held-out false alarms of 3.4% and 12.7% (noul7, precision rule) and up to 24% (noul7, greedy, Qwen3).
  Qwen3 is consistently the harder set. A validation budget is a floor, not a promise.
- **Greedy trades false alarms for recall.** At the same validation false-alarm rate it gains about
  2-6 points of catch rate (bullying and threats recall are noticeably better) but the held-out false
  alarms are higher (for noul7 roughly double). It fits validation more tightly, which transfers worse.
- **There is no dominant configuration.** noul7 with the precision rule has the lowest false alarms
  (0.3% / 3.4% / 12.7%) at about 82% catch on unseen sets. choice8 with greedy has the highest catch
  (98.2% / 87.9% / 90.4%) at 12.9-18% false alarms. choice8 is also about 3x faster on CPU.

## What it does not show

- Which point to choose: that depends on how many false alarms per real flag are tolerable, against
  how many risky chats may be missed.
- Anything about real conversations; both held-out sets are LLM-written, one training run per design.
- Confidence intervals are wide (147-150 benign rows per held-out set): the 3.4% vs 6.8% Llama gap is
  5 vs 10 rows.

## Suggested next steps

1. A product decision on the operating point (or two points: a high-recall alert and a
   high-precision one).
2. If false alarms must be bounded on unseen text, set the budget much lower than the target (the
   validation-to-held-out gap here is 2-16x), or add a third held-out generator to report a range.
3. Real or human-written conversations remain the deciding test.
