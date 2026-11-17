# Thresholds by a false-alarm budget (2026-09-30)

**Question:** `false-alarm-audit.md` found that many benign false alarms are barely-over-threshold
flags, so the threshold policy matters. If thresholds are chosen so the benign false-alarm rate on the
mixed validation set stays within 2%, what happens on the held-out sets?

**Short answer:** it roughly halves false alarms at a modest recall cost, but it does **not** reach 2%
on unseen generators. The budget is met on validation (1.5%), and the same thresholds give 7.5-20% on
the unseen Llama and Qwen3 sets.

## Rule and setup

No retraining. Same mixed-trained models as `mixed-training.md`, same held-out sets. Only the
threshold policy changed, and it was fixed in advance:

> Try precision targets 0.8, 0.9, 0.95, 0.98 in order; take the first whose thresholds keep the
> benign false-alarm rate on the **mixed validation half** at or below 2% (`vigil/metrics.py`,
> `pick_thresholds_with_fa_budget`). Then apply those thresholds once, unchanged, to the held-out sets.

Both designs picked **precision target 0.95** (validation false alarms 3 of 202 = 1.5%). Held-out sets
were scored once; nothing was tuned on them.

## Results (before = precision >= 0.8 rule; after = 2% budget rule)

| Set | Design | Benign false alarms before | after | Catch, any category, before | after | Catch, correct category, before | after |
|---|---|---|---|---|---|---|---|
| Horizon (in-dist, 5,000) | noul7 | 0.8% (5/654) | **0.3%** (2/654) | 99.4% | 97.5% | 97.8% | 96.0% |
| Horizon (in-dist, 5,000) | choice8 | 2.0% (13/654) | **0.8%** (5/654) | 99.5% | 97.4% | 97.5% | 94.9% |
| Llama (831) | noul7 | 15.0% (22/147) | **7.5%** (11/147) | 96.8% | 91.4% | 84.8% | 83.5% |
| Llama (831) | choice8 | 37.4% (55/147) | **17.7%** (26/147) | 99.1% | 96.6% | 83.9% | 82.2% |
| Qwen3 (850) | noul7 | 23.3% (35/150) | **15.3%** (23/150) | 97.0% | 92.6% | 87.0% | 83.7% |
| Qwen3 (850) | choice8 | 40.0% (60/150) | **20.0%** (30/150) | 99.3% | 94.9% | 88.9% | 82.7% |

Recall per category is mostly unchanged; the costs concentrate in a few places. Grooming recall on the
unseen sets moves 95% to 93% (Llama, noul7) and 100% to 99% (Qwen3, noul7). **Bullying on Qwen3 drops
from 96% to 86% (noul7) and 97% to 84% (choice8)**, the largest single cost. Threats, isolation,
personal_info and platform_migration are essentially unchanged.

## What this shows

- The policy change is worth having: about half the false alarms are removed everywhere, including
  in-distribution, for a cost of about 2-5 points of "any category" catch rate and 1-6 points of
  "correct category" catch rate.
- With the budget rule, choice8's false alarms on the unseen sets (17.7% / 20.0%) are about where
  noul7 was with the old rule (15.0% / 23.3%). noul7 is still clearly better (7.5% / 15.3%).
- **The 2% budget does not transfer.** Validation false alarms are 1.5% because the validation set is
  built from generators the model was trained on (Mistral, Gemma) and from Horizon. On unseen
  generators the same thresholds give 7.5-20%. So a validation set of seen generators underestimates
  false alarms on new text by roughly 5-10x.

## What it does not show

- It does not show what a stricter budget would do; I did not sweep budgets on the held-out sets, on
  purpose, because that would be tuning on the test.
- Two unseen generators, both LLM-synthetic; one training run per design; the bullying drop on Qwen3
  is one set and may be partly specific to it.
- False alarms of 7.5-15% on noul7 are still too many for real traffic, where benign chats dominate.

## Suggested next steps

1. A validation set that better represents unseen text: hold one more generator out of training and
   include it only in validation, so the budget is set on something the model has not seen.
2. More diverse training data (more generators, more benign banter and safety-advice chats), then
   re-run this rule.
3. A product decision on the trade-off: how many false alarms per real flag are acceptable, and how much
   recall (especially bullying) may be given up for it. That decides the budget, and I cannot pick it
   from the data.
4. Real or human-written conversations remain the deciding test.
