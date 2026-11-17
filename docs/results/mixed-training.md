# Mixed-generator training (2026-09-29)

**Question:** `xgen.md` showed the pilot models did not transfer to conversations written by a
different model (benign false alarms 38% / 86%). Does training on data from more generators, and
choosing thresholds on a mixed-generator validation set, fix that?

**Short answer:** it helps a lot without hurting in-distribution results, but it is not solved.
On two unseen generators the 8-question design (`noul7`) now flags **15% and 23%** of benign chats
(was 38% and 46%), and recovers recall. The 2-pass design (`choice8`) still flags **37% and 40%**
(was 86% and 95%), so its 3x speed-up does not pay off yet.

## What changed

| | Before (`pilot.md`) | Now |
|---|---|---|
| Training data | 5,000 Horizon conversations | the same 5,000 **plus** 2,697 written by Mistral-7B-Instruct-v0.3 and 2,700 by Gemma-2-9B-it (300 per category for grooming, bullying, isolation, personal_info, platform_migration, threats; 900 benign each, from 15 benign scenes including more adult-child and family chats; both English and Spanish) |
| Not used from generated data | | `sexual_content` (Llama/Qwen3 could not write it reliably) and severity (unreliable when generated), so generated rows train only the category questions |
| Threshold selection | in-distribution eval half | a **mixed validation** file: 900 Horizon eval rows that the test sample never uses + 450 new Mistral + 450 new Gemma rows; thresholds picked on half of it (max recall at precision >= 0.8) |
| Training | 2 epochs, calibration slice split by item | 2 epochs, calibration slice split by whole conversation |
| Items | noul7 40,000; choice8 10,000 | noul7 about 77,800; choice8 about 15,400 |

Held out (never trained on, thresholds fixed and applied unchanged): the Horizon eval sample (5,000
rows, in-distribution), the **Llama-3.1-8B** set (831 rows, from `xgen.md`) and a new **Qwen3-8B**
set (850 rows, same prompts and specs as the Llama set). Qwen3 is a newer model than the Qwen2.5-7B
that wrote most of Horizon's data, so it is not fully independent of Horizon's generator family.

A bug caught on the way: the first generation jobs silently produced grooming plus benign only
(`sbatch --export` split the category list on commas). I noticed because the requested counts were
1,200 instead of 2,700, deleted those files, fixed the script, and regenerated.

## Results

"Before" numbers are the pilot models with their original thresholds; "after" the mixed-trained models
with mixed-validation thresholds. All are fixed thresholds on held-out rows, nothing tuned on them.
The original models were also scored on the Qwen3 set for a like-for-like comparison.

### In-distribution (Horizon test sample, 5,000 rows) is unchanged

| | Grooming recall | Benign false alarms | Catch rate, correct category |
|---|---|---|---|
| noul7 before / after | 95.8% / 96.3% | 0.9% / 0.8% (5/654) | 98.3% / 97.8% |
| choice8 before / after | 97.4% / 96.3% | 1.8% / 2.0% (13/654) | 98.6% / 97.5% |

### Unseen generators: benign false alarms and catch rate

| Set | Design | Benign false alarms before | after | Catch rate, correct category, before | after |
|---|---|---|---|---|---|
| Llama (831) | noul7 | 38.1% | **15.0%** (22/147; CI 10.1-21.6) | 71.5% | 84.8% |
| Llama (831) | choice8 | 86.4% | **37.4%** (55/147; CI 30.0-45.5) | 85.7% | 83.9% |
| Qwen3 (850) | noul7 | 46.0% | **23.3%** (35/150; CI 17.3-30.7) | 70.0% | 87.0% |
| Qwen3 (850) | choice8 | 94.7% | **40.0%** (60/150; CI 32.5-48.0) | 83.4% | 88.9% |

### Unseen generators: recall / precision per category (noul7, before to after)

| Category | Llama recall | Llama precision | Qwen3 recall | Qwen3 precision |
|---|---|---|---|---|
| grooming | 90.8 to 94.9 | 42.2 to 57.1 | 77.0 to 100.0 | 44.0 to 63.3 |
| bullying | 91.9 to 100.0 | 49.7 to 52.9 | 80.0 to 96.0 | 58.4 to 56.5 |
| isolation | 88.0 to 100.0 | 83.0 to 89.3 | 69.0 to 100.0 | 62.7 to 86.2 |
| personal_info | 100.0 to 100.0 | 75.4 to 96.1 | 99.0 to 93.0 | 73.9 to 98.9 |
| platform_migration | 53.5 to 96.0 | 98.1 to 92.2 | 14.0 to 97.0 | 70.0 to 87.4 |
| threats | 50.0 to 88.0 | 68.5 to 88.0 | 89.0 to 98.0 | 64.5 to 65.8 |

AUC on the unseen sets rose as well (for example Llama grooming 0.940 to 0.975, threats 0.886 to
0.974). `sexual_content` is left out here as before: its labels in the Llama and Qwen3 sets are weak
(recall 8-38%, uninterpretable).

## Where the remaining false alarms are

noul7 benign false alarms by scene (Llama / Qwen3):

- **friends joking and teasing each other:** 43% / **96%**
- **a child asking an adult about staying safe online:** 54% / 43%
- video-game chat 8% / 17%, teacher or coach 9% / 9%, birthday party 6% / 0%, parent arranging
  pick-up 0% / 4%, homework 0% / 0%

So the teacher/coach, parent pick-up and birthday scenes that were the pilot's worst case (35-56%
before) are now 0-9%, and Spanish is no longer the outlier (Llama 17% vs 12% for English, Qwen3 19% vs
28%). What is left is playful teasing, which is close to bullying by nature, and safety advice, which
talks about strangers and risky topics. Some of those rows may be legitimately ambiguous rather than model errors;
I have not audited them.

choice8 still flags 19-70% of *every* scene on Llama, so its problem is general, not scene-specific.

## What this shows and does not show

- Training on several generators plus mixed-generator thresholds roughly halves benign false alarms
  on unseen generators and restores recall, at no cost in-distribution.
- **noul7 transfers clearly better than choice8** here. choice8's low thresholds (about 0.016-0.02 for
  grooming, bullying and isolation) make it flag much benign text; its speed advantage (about 3x on CPU) would need a different
  operating rule or more training before it is usable.
- **Still not good enough to ship:** 15-23% benign false alarms and grooming precision 57-63% on other
  generators, against about 1% and 79% in-distribution. At real prevalence (mostly benign) that
  would mean mostly false alarms per flag.
- Two unseen generators, both LLM-synthetic text. One training run per design, no seed-to-seed
  variance. The teasing scene and `sexual_content` labels are unreliable. Nothing here is real
  conversations.

## Suggested next steps

1. Audit the teasing and online-safety benign rows, and add more benign diversity for banter and
   safety questions.
2. Real (or human-written) conversations are still the deciding test.
3. Retrain severity on labels that are reliable (the generated ones are not).
4. Only then revisit speed (`choice8`) and size work.
