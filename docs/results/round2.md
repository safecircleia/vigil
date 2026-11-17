# Round 2: more benign data and an unseen validation generator (2026-09-30)

**Question:** `fa-budget.md` found that a 2% false-alarm budget set on validation data from generators
the model had trained on did not transfer (1.5% on validation, 7.5-20% on unseen generators). Does
(a) more benign banter and safety-advice training data plus (b) a validation-only generator fix that?

**Short answer:** the budget now transfers, but the price in recall is high for the 8-question design,
and the validation set turned out thinner than intended. The two designs now sit at different points on
the false-alarm/recall trade-off, so this is not a clean win for either.

## What changed

| | Round 1 (`mixed-training.md` + `fa-budget.md`) | Round 2 |
|---|---|---|
| Extra training data | none | +999 (Mistral) and +1,000 (Gemma) benign conversations from 8 new scenes: privacy settings, scam spotting, telling a parent about a strange message, good-natured joking, siblings teasing lightly, blocking spam, complimenting drawings, discussing internet-safety news |
| Validation | Horizon rows + Mistral + Gemma (both trained on) | the same, **plus Falcon3-7B-Instruct**, never trained on |
| Threshold rule | smallest precision target from {0.8, 0.9, 0.95, 0.98} that keeps benign false alarms <= 2% on the validation half | same rule |
| Held-out sets | Horizon sample, Llama (831), Qwen3 (850) | the same, scored once |

**Falcon3 was a weak validation generator.** It refused or failed on 244 of 390 requests: all 50
grooming, 48 of 50 threats, 45 of 50 personal_info and 41 of 50 bullying. What it contributed is 205
rows: 142 benign, 47 isolation/platform_migration positives, and 16 others. So the unseen part of
validation is almost only *benign* pressure, which tightens thresholds, and it says almost nothing about
recall on unseen grooming, threats or bullying.

## Results (round 1 budget rule to round 2)

| Set | Design | Benign false alarms | Catch, any category | Catch, correct category |
|---|---|---|---|---|
| Horizon (in-dist) | noul7 | 0.3% to 0.2% (1/654) | 97.5% to 94.5% | 96.0% to 93.4% |
| Horizon (in-dist) | choice8 | 0.8% to 0.3% (2/654) | 97.4% to 99.2% | 94.9% to 96.0% |
| Llama | noul7 | 7.5% to **2.0%** (3/147) | 91.4% to 84.5% | 83.5% to 77.0% |
| Llama | choice8 | 17.7% to **10.2%** (15/147) | 96.6% to 98.5% | 82.2% to 82.2% |
| Qwen3 | noul7 | 15.3% to **1.3%** (2/150) | 92.6% to 81.1% | 83.7% to 76.3% |
| Qwen3 | choice8 | 20.0% to **11.3%** (17/150) | 94.9% to 98.9% | 82.7% to 86.1% |

The rule picked precision target **0.98** for noul7 (validation false alarms 4/268 = 1.5%) and **0.9**
for choice8 (3/268 = 1.1%).

Where the recall goes (noul7, round 1 to round 2):

- bullying: Llama 97% to 82%, **Qwen3 86% to 58%** (Spanish only 36%, English 80%)
- threats: Llama 88% to 65%, Qwen3 98% to 86%
- grooming: Qwen3 99% to 89%; in-distribution grooming 93% to 91%
- isolation, personal_info, platform_migration: essentially unchanged

choice8 recall barely moves (bullying Llama 97% to 97%, Qwen3 84% to 86%; platform_migration about 98%
to 93%). Its remaining false alarms concentrate in the teasing scene (35% on Llama); the online-safety
scene is now 0%.

## What this shows

- **The false-alarm budget can transfer** when the validation set includes a generator the model has not
  seen: noul7 went from 7.5%/15.3% to 2.0%/1.3% on the two unseen sets, matching the 1.5% on validation.
  The extra benign banter and safety data also removed those two scenes as a problem for noul7.
- **It is not free.** noul7 pays with a large recall loss on unseen text (catch, correct category:
  about 77% against about 84% before), concentrated in bullying and threats. The Falcon3 validation had
  almost no positives for those, so the thresholds were tightened by benign evidence alone.
- **The two designs are now not comparable head to head.** noul7 sits near 2% false alarms and 77% catch;
  choice8 near 11% false alarms and 82-86% catch. Round 1's "noul7 clearly better" was at similar
  operating conditions; at these different points it is unclear which design is better.

## What it does not show

- Whether the recall loss is mostly the 0.98 target (a global precision target applied to all
  categories) or the training data; I did not separate them.
- One unseen validation generator that mostly refused, two held-out generators, one training run per
  design, 147-150 benign rows per held-out set. A few points either way is noise.
- Nothing about real conversations.

## Suggested next steps

1. A validation-only generator that actually produces risky chats (a compliant model, or several), so
   thresholds are set with unseen positives as well as unseen benign text.
2. A per-category threshold rule or budget so one strict category-independent target does not cost
   bullying and threats recall.
3. A product decision on the operating point: false alarms per real flag versus missed risky chats.
   The data can show the curve; it cannot pick the point.
4. Real or human-written conversations remain the deciding test.
