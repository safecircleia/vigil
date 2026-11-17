# Vigil pilot run (2026-09-29)

> **Update:** a cross-generator test (`xgen.md`) found that the near-perfect numbers below partly reflect
> generator similarity: ranking transfers but the thresholds and false-alarm rates do not (benign false
> alarms 38% for noul7 and 86% for choice8 on a set written by a different model). Read this file with
> that in mind.

A small first run to answer three questions: does fine-tuning Laya-multilingual on Horizon's dataset
work at all, what does it cost, and does a cheaper question design lose accuracy? It is **not** a
candidate to replace Horizon (see "What this does not show").

The first pilot (jobs 18004 onward) was reviewed and had defects (calibration slice leaked whole
conversations into training, evaluation silently truncated long conversations, the base comparison hid
a threshold fallback). Everything below is the **rerun after those fixes**.

## Setup

| | |
|---|---|
| Dataset | `safecircleai/horizon-training-data@0cb2a780c2743ebaf0142fd543f376e53ac4776b`: 1,584,000 train / 160,000 eval rows (144,000 train rows are Horizon's identity-refusal examples and are skipped) |
| Overlap audit | 0 exact and 0 whitespace/case-normalised duplicates of any of the 160,000 eval conversations in the train file (`scripts/check_overlap.py`). Near-duplicates (paraphrased templates) were **not** tested |
| Pilot sample | 5,000 conversations, stratified: 625 each for 7 categories + benign. 0 malformed, 0 inconsistent labels, 0 overflow. Same conversations and seed for both designs |
| Base | `convaiinnovations/laya-multilingual` (322M) |
| Training | full fine-tune, cross-entropy + RLCD term, 2 epochs (the script default is 4), micro-batch 8 x grad-accum 4, one L40S on partition `gpu` (the script default is `gpuUltra`; the H100 queue was about 12 h away). Whole conversations held out for calibration (up to 1,600 items) |
| Evaluation | 5,000 random eval rows (seed 20260922), 0 excluded for length. Split in two halves: thresholds picked on the validation half (max recall subject to precision >= 0.8), scored on the disjoint test half (2,500 rows: 331 benign, 2,169 risky) |

Two designs, same data, one run each:

- **noul7** (spec design): 7 yes/no questions + severity = **8 encoder passes** per conversation (job 18038).
- **choice8**: one 8-way choice (benign + 7 categories) + severity = **2 encoder passes** (job 18044).
  Several gold categories would share the target mass; this data has at most one category per
  conversation, so that path is untested.

## Results on the test half

Precision is at the validation-picked threshold. Every category met the 0.8 precision target on the
validation half (none fell back to 0.5), but it is a single observed score, so test precision wobbles.

| Category | noul7 recall (95% CI) | noul7 precision | choice8 recall (95% CI) | choice8 precision |
|---|---|---|---|---|
| grooming | 95.81% (92.96-97.53) | 80.7% | 97.42% (94.99-98.69) | 83.4% |
| bullying | 99.37% (97.74-99.83) | 86.8% | 98.43% (96.38-99.33) | 90.2% |
| sexual_content | 97.71% (95.35-98.89) | 84.5% | 97.71% (95.35-98.89) | 81.7% |
| isolation | 98.29% (96.07-99.27) | 90.3% | 98.98% (97.03-99.65) | 87.1% |
| personal_info | 99.08% (97.34-99.69) | 90.3% | 98.78% (96.90-99.52) | 98.8% |
| platform_migration | 98.94% (96.94-99.64) | 100.0% | 99.65% (98.03-99.94) | 99.3% |
| threats | 98.79% (96.93-99.53) | 95.0% | 99.39% (97.82-99.83) | 87.9% |

| | noul7 | choice8 | Untouched base (noul7) |
|---|---|---|---|
| Catch rate: any category flagged, among risky | 99.45% (99.04-99.68) | 100.00% (99.82-100.00) | 42.92% |
| Catch rate: a *correct* category flagged | 98.29% (97.66-98.76) | 98.62% (98.03-99.03) | not computed |
| Benign false alarms | 0.9% (3/331; CI 0.3-2.6%) | 1.8% (6/331; CI 0.8-3.9%) | 3.9% (13/331) |
| Severity macro-F1 (single run, no interval) | 0.628 | 0.671 | not computed |
| AUC per category | 0.993-1.000 | 0.995-1.000 | 0.44-0.88 (grooming 0.50) |
| Fitted temperatures (choice, score, noul) | 1.2 (unused), **6.06**, 4.07 | 1.68, 1.56, 1.2 (unused) | n/a |
| GPU latency per conversation, p50 / p95 | 13.0 / 17.9 ms | 10.6 / 19.9 ms | n/a |

Reading the two designs: on this data they are **not distinguishable on recall** (the intervals
overlap for every category). choice8 has a higher point estimate for benign false alarms (1.8% vs 0.9%,
intervals overlap) and a higher severity F1 (0.671 vs 0.628, no interval, one run each). The untouched
base is weak on every category (grooming AUC 0.50, i.e. no signal), so most of the gain comes from
fine-tuning.

Horizon's recorded numbers, for reference only: grooming recall 90.36%, catch rate 12.90% (its regex
rules, not a model), 2.1 s, 1017 MB. They are **not comparable**: a different eval set, procedure and
operating point.

## Precision at realistic prevalence

The test half is 87% risky. Real traffic is mostly benign, so a flag means less than the tables above
suggest. Computed from the catch rate and benign false-alarm rate above (any category flagged; the
point value and the upper 95% bound of the false-alarm rate), by the share of conversations that are
actually risky:

| Design | False-alarm rate | 10% risky | 1% risky | 0.1% risky |
|---|---|---|---|---|
| noul7 | 0.9% (point) | 92% | 53% | 9.9% |
| noul7 | 2.6% (upper bound) | 81% | 28% | 3.7% |
| choice8 | 1.8% (point) | 86% | 36% | 5.2% |
| choice8 | 3.9% (upper bound) | 74% | 21% | 2.5% |

At 1% prevalence or lower, most flags would be false alarms. The false-alarm rate rests on only 331
benign rows.

## Export (real pilot checkpoints, cluster CPU node)

The job requested 8 CPUs but `os.cpu_count()` reported 30, so ONNX Runtime may have used more threads
than allocated: treat CPU latency as a server-CPU figure, not an on-device number.

| | noul7 (job 18041) | choice8 (job 18055) |
|---|---|---|
| fp32 ONNX vs PyTorch, 25 conversations | max abs prob diff 0.0000, 0 flips | max abs prob diff 0.0000, 0 flips |
| fp32 ONNX latency per conversation, p50 / p95 | 948 / 1,510 ms | **306 / 438 ms** |
| fp32 size | 1.2 GB | 1.2 GB |
| int8 (dynamic) size / p50 latency | 883 MB / 1,067 ms | 883 MB / 276 ms |
| int8 vs fp32 | max abs diff 0.325, 0/175 flips | max abs diff 0.656, 3/175 flips |

Decision "flips" are counted at 0.5, but the operating thresholds are far lower (0.008-0.015 for noul7),
so flip counts say little about shipped behaviour; the maximum absolute probability difference is the
informative number. int8 dynamic quantization is not worth using: no meaningful speed-up, 883 MB, and
probabilities move by up to 0.66.

## What this shows

- Fine-tuning works on this dataset: a few minutes of training takes catch rate from 43% to 99-100%
  and grooming recall from 0% to 96-97%.
- The 2-pass choice8 design costs no measurable recall here and cuts CPU latency about 3x (306 ms vs
  948 ms), which meets the 500 ms target on a server CPU. GPU latency barely changes (about 11 vs 13 ms).
- fp32 ONNX export is faithful to PyTorch for both designs.

## What this does not show

1. **Not real-world performance.** The data is synthetic and both models separate it almost perfectly
   (AUC 0.993-1.000, scores near 0.999 vs 0.002). That looks like a generator artifact as much as a
   strong model. Nothing was tested on real conversations.
2. **Prevalence.** See the table above; precision at realistic prevalence is far lower.
3. **Targets not met or unproven.** Grooming recall 97.42% (choice8) clears 97% only at the point
   estimate: the interval 94.99-98.69% does not. Disk is 883 MB to 1.2 GB against the 50 MB goal. CPU
   latency was measured on a server, not a phone. Severity macro-F1 is 0.63-0.67, which is weak.
4. **Small sample, one run per design.** Each design was trained once on 5,000 conversations for 2
   epochs; there is no seed-to-seed variance estimate. Differences under about a point on recall
   should be treated as noise.
5. **Thresholds are far from 0.5** (0.008-0.015 for six noul7 categories; 0.03-0.36 for choice8). They
   follow from "max recall at precision >= 0.8" on a bimodal score distribution, so they are a policy
   choice and not calibrated probabilities. Recall at 0.5 is lower for several categories (see AUC and
   "Recall @0.5" columns in the `report.md` files).
6. **Calibration.** The noul7 score temperature was fitted at 6.06, above the range Laya honours at
   inference (0.5-5), so the runtime used 5. The fit is now clamped to that range
   (`TEMP_MIN/TEMP_MAX`); the checkpoints here predate the clamp but behave identically. The much
   higher temperatures than in the first, leaky pilot (score 4.67 to 6.06, noul 3.21 to 4.07) show the
   item-level split had understated overconfidence.
7. **Single-label data.** Every conversation has at most one category, so multi-label behaviour is not
   exercised by either design; the choice8 soft-target path is untested on real data.
8. **Overlap audit is exact-match only.** It found no duplicates but cannot rule out templated
   near-duplicates between train and eval.

## Decision this raises

choice8 looks like the better base for a larger run: same accuracy on this data, about 3x less CPU
compute, better-behaved temperatures. The cost is native multi-label output, which this dataset cannot
test. Before scaling, the open questions are how real conversations differ (single vs multi category)
and whether the near-perfect separation survives contact with non-synthetic data.
