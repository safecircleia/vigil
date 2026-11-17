# False-alarm audit (2026-09-30)

> **Correction (banter round, `banter.md`):** a blind audit of the teasing scene found 43/46 rows friendly, 0 cruel, so the flags there are mostly real model errors, not label noise as suggested below.

**Question:** after mixed-generator training (`mixed-training.md`), most remaining benign false alarms
sit in two scenes: friends teasing each other and a child asking an adult about online safety. Are
those real model errors, ambiguous labels, or something else?

**Short answer:** mostly neither. A large share are **barely-over-threshold flags**: the model gives
the benign chat a probability of 0.01-0.09, and the operating threshold (0.007-0.03 for most
categories) is lower than that. The confident false alarms (probability >= 0.5) are a much smaller
group, and they are the boundary cases: insulting banter and safety advice that talks about strangers
and risky topics.

## How confident are the false alarms?

Top flagged probability of each flagged benign row, at the mixed-validation thresholds (held-out
generators, benign rows only: 147 Llama, 150 Qwen3).

| Model / set | Benign rows flagged | top prob < 0.1 | 0.1 - 0.5 | >= 0.5 |
|---|---|---|---|---|
| noul7 / Llama | 22 (15.0%) | 16 (10.9%) | 2 (1.4%) | 4 (2.7%) |
| noul7 / Qwen3 | 35 (23.3%) | 16 (10.7%) | 3 (2.0%) | 16 (10.7%) |
| choice8 / Llama | 55 (37.4%) | 23 (15.6%) | 21 (14.3%) | 11 (7.5%) |
| choice8 / Qwen3 | 60 (40.0%) | 26 (17.3%) | 12 (8.0%) | 22 (14.7%) |

For noul7, 73% (Llama) and 46% (Qwen3) of false alarms are low-confidence flags. The rate of
*confident* false alarms is 2.7% and 10.7% of benign chats, against 15% and 23% overall.

## What I read (one reader, a Claude model; small samples)

**Teasing, noul7 on Qwen3 (7 of the 22 flagged rows):**

- 3 flagged at bullying 0.99. Each contains a real insult in an otherwise friendly exchange
  ("You're such a failure", "just give up and let me do it for you", "you're still single"). The
  "benign" label is arguable; these are boundary cases, not clear model errors.
- 4 flagged at bullying 0.01-0.09 on plainly harmless banter (Spanish ice-cream jokes, homework
  chat). These are threshold artifacts, not content.

**Online-safety advice, noul7 on Llama (6 of the 7 flagged rows):**

- 4 flagged at platform_migration 0.008-0.009, just over a 0.007 threshold, on entirely harmless
  advice chats. Threshold artifacts.
- 2 are real confident-ish errors: platform_migration 0.76 (a child asking about suspicious links
  and messaging apps) and threats 0.37 (a child asking how to spot strangers). The topic vocabulary
  overlaps with the risky categories.
- One row labelled English has the adult answering in Spanish, a generator quirk.

## What this means

- The dominant cause of the remaining false alarms is the **threshold policy**, not the content. The
  rule "max recall at precision >= 0.8 on validation" places thresholds right at the edge of the
  negative score mass (0.007-0.03). On an unseen generator the negative mass shifts up a little, and
  a large part of it crosses.
- The genuinely confident errors are a smaller set of true boundary cases. Some of them (insulting
  banter) may not be errors at all, depending on how strict the product wants "bullying" to be.
- This audit is small (13 rows read, 2 scenes, noul7 only) and I did not read choice8 rows. The
  choice8 pattern (more mass in the 0.1-0.5 and >= 0.5 buckets) suggests its problem is broader.

## Suggested next step

Change how thresholds are chosen, using **validation data only**: for example, pick the smallest
precision target (0.8, 0.9, 0.95, 0.98) whose benign false-alarm rate on the mixed validation set is
at most 2%, then apply it once to the held-out sets. That is decided a priori, so it does not tune on
the test sets. It will cost some recall, which the report should show alongside the false-alarm drop.
