# Friendly-banter round (2026-09-30)

**Question.** Policy decided: friendly teasing with a mild insult is *benign*. Did a blind audit and ~2,000
extra friendly-banter conversations cut the false alarms on the "friends joking and teasing" scene?

**Short answer.** No. The scene stays at about 50% flagged for the 2-pass design and 41-52% for the
8-question design, whatever the data. Overall false alarms on unseen generators are unchanged within noise.

## Correction to earlier write-ups

`false-alarm-audit.md` and `seeds-hardneg.md` suggested that much of the teasing-scene false alarms were
label noise (insults the generator was asked to write as banter). The blind audit
(`docs/audit/teasing-verdicts.json`, 46 rows) says otherwise: **43 friendly, 3 unclear, 0 cruel**. So the
flags are mostly real model errors, not bad labels. Treat those earlier statements as superseded.

## Setup

- Group A: current data (3 seeds). B: + educational hard negatives (3 seeds). C: B + 2,000 friendly-banter
  conversations (Mistral, Gemma; 1,000 each, `banter` scene set), 3 seeds. Greedy rule, 2% budget, both designs.
- Code: `vigil/xgen.py` banter scenes, `slurm/submit_banter.sh`, `slurm/submit_seeds.sh` (`DESIGNS=` subset).

## Results (greedy rule; mean +/- sd over 3 seeds)

| Set | Design | False alarms A / B / C | Catch (correct) A / B / C |
|---|---|---|---|
| Horizon | 8-question | 0.9 / 0.9 / 1.2% | 97.8 / 98.3 / 98.3% |
| Llama | 8-question | 8.8 / 7.7 / 8.6% | 84.3 / 84.9 / 84.0% |
| Qwen3 | 8-question | 17.3 / 18.4 / 17.6% | 88.1 / 86.7 / 89.4% |
| Horizon | 2-pass | 1.3 / 0.8 / 0.7% | 97.2 / 98.3 / 99.1% |
| Llama | 2-pass | 11.6 / 11.6 / 12.0% | 85.7 / 89.5 / **93.5%** |
| Qwen3 | 2-pass | 15.3 / 17.1 / 19.3% | 87.4 / 93.0 / **95.9%** |

Teasing-scene false alarms: 8-question 49 / 52 / 41%; 2-pass 49 / 50 / 49%.
Other scenes (2-pass, A / B / C): homework 6 / 11 / 12%, pick-up and dinner 8 / 13 / 16%,
safety advice 14 / 1 / 6%.

## What this shows

- Friendly-banter data does not fix the teasing scene. A plausible reason (hypothesis, untested): the
  bullying class in the source dataset is largely insulting banter, so the boundary is intrinsically subtle.
- The 2-pass design keeps gaining catch (85.7 -> 93.5% Llama, 87.4 -> 95.9% Qwen3, A to C, several times the
  seed sd), but false alarms on Qwen3 drift up (15.3 -> 19.3%) and several benign scenes creep up. Catch and
  false alarms trade off; it is not a clean win. At best a modest recall lever.
- 2-pass remains the main candidate (equal or better than the 8-question design at a third of the compute).

## Not shown

Only LLM-written text, two held-out generators, 3 seeds (sd 1-6 points on false alarms). The teasing audit
is 46 rows, one reviewer. Real human conversations remain the deciding test.

## Next steps

1. Get real or human-written conversations for the teasing scene; generators may not match real banter.
2. Decide the operating point (catch vs false alarms) as a product choice; 2-pass C has highest recall.
3. A second-stage confirmer is the only thing so far that cut false alarms (OpenJev cascade, dropped).
