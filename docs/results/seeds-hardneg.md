# Training noise and a hard-negative round (2026-09-30)

> **Correction (banter round, `banter.md`):** a blind audit of the teasing scene found 43/46 rows friendly, 0 cruel, so the flags there are mostly real model errors, not label noise as suggested below.

**Questions.** (1) How much do results move between training runs (seeds)? Round 4 suggested a lot.
(2) Do more benign chats that *talk about* risky topics (the kind the false-alarm audit found) reduce
false alarms on unseen text, once that noise is accounted for?

**Short answer.** (1) Noise is real: over 3 seeds, false alarms on unseen text vary by 1-3 points
(sd) and, with the precision-target rule, catch rate by up to 5 points. The greedy rule is much more
stable. (2) The hard negatives did **not** reduce overall false alarms for either design. They did lift
the 2-pass design's catch rate by 4-6 points at unchanged false alarms. And almost half of all remaining
false alarms come from a single scene: friends teasing each other.

## Setup

- **Group A (current data, 3 seeds):** round 4's model (seed 20260922) plus seeds 1 and 2.
- **Group B (hard negatives, 3 new seeds 1-3):** the same training data plus 999 + 1,000 benign
  conversations (Mistral, Gemma) from 10 new "educational" scenes: a parent explaining why not to share an
  address online, a teacher explaining grooming, asking a counsellor how to report a bully, asking a sibling
  what to do if a stranger asks for photos, family app rules, a class chat about an online-safety
  assembly, asking whether to move a chat to another app, tips on blocking and reporting, and so on.
- Same recipe (2 epochs), same mixed validation file, thresholds by the 2% budget rule (greedy and
  precision-target), applied unchanged to the Horizon sample, Llama and Qwen3 sets. Both designs.
- Code: `--seed` for training, `slurm/submit_seeds.sh`, `slurm/submit_hardneg.sh`,
  `vigil/seed_spread.py`, `scripts/scene_spread.py`.

**Caveat on independence:** I chose the new scenes after auditing the held-out Llama and Qwen3 sets, so
improvement on the matching held-out scene (online-safety advice) is partly by construction and is not an
independent test.

## 1. Noise across seeds (group A, three runs; mean +/- sd [range])

Greedy rule (8-question design / 2-pass design):

| Set | False alarms, 8-question | Catch (correct), 8-question | False alarms, 2-pass | Catch (correct), 2-pass |
|---|---|---|---|---|
| Horizon | 0.9% +/- 0.4 | 97.8% +/- 0.4 | 1.3% +/- 0.5 | 97.2% +/- 0.8 |
| Llama | 8.8% +/- 3.1 [6.1-12.2] | 84.3% +/- 0.7 | 11.6% +/- 2.0 [9.5-13.6] | 85.7% +/- 0.9 |
| Qwen3 | 17.3% +/- 2.9 [15.3-20.7] | 88.1% +/- 1.3 | 15.3% +/- 1.3 [14.0-16.7] | 87.4% +/- 1.1 |

The precision-target rule is noisier on catch rate: sd 3.9-5.0 points for the 8-question design on the
three sets (for example Qwen3 77.7% +/- 5.0, range 72.7-82.7), against 0.4-1.3 with the greedy rule. So
differences of a few points between configurations in rounds 2-4 should be treated as noise, and the
greedy rule is the better default.

## 2. Hard negatives (group B) against group A, greedy rule

| Set | Design | False alarms A to B | Catch (correct) A to B |
|---|---|---|---|
| Horizon | 8-question | 0.9% to 0.9% | 97.8% to 98.3% |
| Llama | 8-question | 8.8% to 7.7% | 84.3% to 84.9% |
| Qwen3 | 8-question | 17.3% to 18.4% | 88.1% to 86.7% |
| Horizon | 2-pass | 1.3% to 0.8% | 97.2% to 98.3% |
| Llama | 2-pass | 11.6% to 11.6% | 85.7% to **89.5%** |
| Qwen3 | 2-pass | 15.3% to 17.1% | 87.4% to **93.0%** |

- False alarms on unseen text do not go down for either design; all changes are inside the seed noise.
- **8-question design:** nothing distinguishable.
- **2-pass design:** catch rate rises 3.8 points (Llama) and 5.6 points (Qwen3), several times the seed
  sd (about 1), at unchanged false alarms. With the precision rule the gain is similar (Qwen3 81.6% to
  89.4%).
- After this round, the 2-pass design matches or beats the 8-question design on the unseen sets at the
  greedy rule (catch 89.5% / 93.0% against 84.9% / 86.7%; false alarms 11.6% / 17.1% against 7.7% / 18.4%)
  and needs about a third of the compute.

## 3. Where the false alarms are (benign rate by scene, both unseen generators pooled, mean over runs)

| Scene | 8-question A | 8-question B | 2-pass A | 2-pass B |
|---|---|---|---|---|
| **friends joking and teasing each other** (46 rows) | **49%** | **52%** | **49%** | **50%** |
| child asking an adult about staying safe online (27) | 16% | 19% | 14% | **1%** |
| classmate asking about homework (43) | 5% | 5% | 6% | 11% |
| parent or relative arranging pick-up and dinner (53) | 7% | 4% | 8% | 13% |
| teacher or coach talking normally (44) | 5% | 3% | 9% | 8% |
| birthday party (36) | 5% | 3% | 2% | 3% |
| video-game chat between classmates (48) | 5% | 6% | 5% | 8% |

- **One scene carries about half or more of all false alarms**, and it does not move with more data or
  either design. Excluding it, overall false alarms are roughly 6-8% (estimated from the per-scene rates),
  against 15-18% with it.
- The online-safety scene is fixed for the 2-pass design (14% to 1%, partly by construction, see the
  caveat) but not for the 8-question design.
- The earlier audit found that in this teasing scene, many flagged chats contain real insults ("you're
  such a failure") that the generator was asked to write as harmless banter. So part of this 50% may be
  correct flags on rows whose "benign" label is arguable, not model errors.

## What this shows and does not show

- Noise is measured now: sd of 1-3 points on false alarms, and the greedy rule is the stable choice.
- More benign data of this kind is not a lever for false alarms; it is a modest lever for the 2-pass
  design's recall.
- The remaining problem is concentrated and partly a labelling question, which is a product policy
  decision (below).
- Three seeds per group give crude spread estimates; two held-out generators; all LLM-written text.

## Suggested next steps

1. **Decide the policy** on friendly banter that contains an insult: is it bullying or not? Then audit and
   relabel (or drop) the ambiguous rows in the test sets and add correctly labelled banter data. That
   single scene is about half of the measured false alarms.
2. Consider the 2-pass design with the greedy rule as the main candidate: equal or better on unseen
   sets, about 3x cheaper on CPU.
3. Real or human-written conversations remain the deciding test.
