# Cross-generator test (2026-09-29)

> **Follow-up:** `mixed-training.md` retrains on several generators and re-runs this test. It roughly halves
> the false alarms below; the 8-question design is still at 15-23% on unseen generators.

**Question:** the pilot models (`docs/results/pilot.md`) separated Horizon's synthetic eval data almost
perfectly. Is that because the eval data came from the same generator and prompts as the training
data?

**Short answer:** partly. Ranking quality transfers reasonably (AUC 0.94-0.995 on well-labelled
categories, against 0.993-1.000 in-distribution), but the **operating points do not transfer**: at the
thresholds chosen in-distribution, precision for grooming and bullying roughly halves (about 81-87%
to 42-54%), and 38% (noul7) to 86% (choice8) of benign conversations are flagged. The pilot's headline numbers were inflated by generator similarity.

## The test set

| | |
|---|---|
| Generator | `meta-llama/Llama-3.1-8B-Instruct` on the SLURM cluster (job 18093, 12 min), fresh prompts written for this test (`vigil/xgen.py`). Horizon's main data came from Qwen2.5-7B, and some of its augmentation from Claude via Bedrock, so this is a different model family and a different prompt template. It is still LLM-written synthetic text, not real conversations |
| Size | 850 requested, **831 kept** (19 refused or unparseable, 10 of them `sexual_content`): 684 risky + 147 benign, 419 English / 412 Spanish |
| Benign | includes hard negatives: teacher or coach chatting, parent arranging pick-up, friends teasing, a child asking an adult about online safety |
| Labels | by construction: the requested category and severity |
| Scoring | both pilot checkpoints, **thresholds fixed** from their in-distribution evaluations and applied unchanged to all 831 rows; nothing tuned on this set |

## Label quality (my reading of 34 random samples)

I read 34 conversations (4 per risky category, 6 benign). One reader, and it is a Claude model
reviewing Llama output, so this is a sanity check and not independent annotation.

| Category | Clear agree | Ambiguous | Wrong category |
|---|---|---|---|
| grooming | 2 | 2 | 0 |
| bullying | 4 | 0 | 0 |
| **sexual_content** | **0** | 3 | 1 |
| isolation | 3 | 1 | 0 |
| personal_info | 3 | 1 | 0 |
| platform_migration | 4 | 0 | 0 |
| threats | 4 | 0 | 0 |
| benign | 6 | 0 | 0 |

26 of 34 clearly right, 33 of 34 counting the ambiguous ones. Consequences:

- **`sexual_content` labels are not usable.** Llama softened every request into grooming-style
  probing ("are you alone right now?", "do you have a more recent photo?"), and refused 10 more. Read
  the `sexual_content` rows as weak labels: the per-category tables below leave the category out, but the
  catch-rate rows count all risky conversations, including these 90.
- **Severity labels are unreliable** (mild bullying tagged severity 4, blackmail tagged 1). The severity
  F1 on this set (0.25 noul7, 0.27 choice8) says nothing about the models.
- The 6 benign samples were clean, so the benign false alarms below are unlikely to be label noise, but
  6 is a small sample.

## Results (fixed thresholds; 831 rows; categories other than sexual_content)

Recall and precision here use each design's in-distribution thresholds. In-distribution figures (from
`pilot.md`) are in brackets.

| Category | noul7 recall | noul7 precision | noul7 AUC | choice8 recall | choice8 precision | choice8 AUC |
|---|---|---|---|---|---|---|
| grooming | 90.8% (95.8) | 42.2% (80.7) | 0.940 (0.993) | 91.8% (97.4) | 41.5% (83.4) | 0.936 (0.997) |
| bullying | 91.9% (99.4) | 49.7% (86.8) | 0.962 (0.999) | 89.9% (98.4) | 53.6% (90.2) | 0.954 (0.998) |
| isolation | 88.0% (98.3) | 83.0% (90.3) | 0.980 (0.997) | 100.0% (99.0) | 46.7% (87.1) | 0.995 (0.999) |
| personal_info | 100.0% (99.1) | 75.4% (90.3) | 0.995 (1.000) | 96.9% (98.8) | 76.6% (98.8) | 0.994 (1.000) |
| platform_migration | 53.5% (98.9) | 98.1% (100.0) | 0.960 (1.000) | 80.8% (99.7) | 89.9% (99.3) | 0.981 (1.000) |
| threats | 50.0% (98.8) | 68.5% (95.0) | 0.886 (0.999) | 82.0% (99.4) | 46.9% (87.9) | 0.931 (0.999) |

| | noul7 | choice8 |
|---|---|---|
| Benign false alarms (any category flagged) | **38.1%** (56/147; CI 30.6-46.2%) [0.9%] | **86.4%** (127/147; CI 79.9-91.0%) [1.8%] |
| Catch rate, any category flagged, among risky | 88.5% (605/684) [99.5%] | 99.3% (679/684) [100%] |
| Catch rate, a *correct* category flagged | 71.5% (489/684) [98.3%] | 85.7% (586/684) [98.6%] |

Grooming recall of 90.8% / 91.8% has 95% intervals of 83.5-95.1% and 84.7-95.8%.

## Where it breaks

- **Benign hard negatives.** noul7 false-alarm rate by scene: birthday-party planning 56%, friends
  teasing 52%, teacher or coach 50%, child asking about online safety 46%, parent arranging pick-up
  35%, video-game chat 29%, homework question 5%. choice8 flagged **100%** of the teacher/coach,
  parent/relative, online-safety and birthday scenes, and **100% of all 75 Spanish benign
  conversations** (72% of English ones). Adult-child benign interactions and Spanish are the weak spot,
  and Horizon's benign data very likely does not cover them.
- **Threats and platform_migration recall** fall sharply for noul7 (English threats 32%, migration
  47%), less so for choice8 (72%, 86%).
- Thresholds do not carry over: they were tuned for the in-distribution score scale (0.01-0.36), and
  benign conversations from another generator score above them much of the time.

## Reading against the bands I set beforehand

Before running I said: grooming recall >= 90% and catch rate >= 95% would mean the pilot numbers mostly
hold; grooming recall 75-90% would mean partly artifact; under 75% would mean untrustworthy. By that
rule alone choice8 lands in the "mostly holds" band (grooming 91.8%, catch 99.3%) and noul7 misses it
on catch rate (88.5%).

**That rule was too weak, and I am not going to lean on it.** It ignored false alarms. choice8's 99.3%
catch rate comes largely from flagging 86% of benign conversations, which makes it unusable at these
thresholds. Taking recall, precision and false alarms together the honest verdict is: **the pilot's
near-perfect numbers partly reflect generator similarity.** The models learned real signal (AUC stays
high) but are miscalibrated on a different distribution, and noul7 transfers noticeably better than
choice8 at fixed thresholds.

## What this does and does not show

- It is one other LLM (8B, Llama), with noisy labels, weak severity and `sexual_content` labels, and
  147 benign rows. Differences of a few points are noise.
- It does not show real-world behaviour: this is still synthetic text.
- The noul7 vs choice8 comparison from the pilot changes: choice8 was about 3x cheaper on CPU and equal
  in-distribution, but it transfers much worse at fixed thresholds. That may be fixable with
  recalibration on mixed data, but it is not free.
- Re-tuning thresholds on this set would likely restore precision, but that would be tuning on the
  test; a mixed validation set is the right way (see below).

## Suggested next steps

1. Generate more independent training data with other open models (Mistral, Gemma) on the cluster,
   including benign hard negatives and Spanish, and mix it into training. Keep this Llama set (and a
   set from a third model) strictly held out.
2. Pick thresholds on a validation set that mixes generators, and report transfer to the held-out sets.
3. Then re-compare noul7 and choice8 before choosing a design to scale.
