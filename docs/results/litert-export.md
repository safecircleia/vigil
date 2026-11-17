# LiteRT export for the Android app (2026-10-02)

Checkpoint: `c1-choice8` (2-pass design, banter round), thresholds from `c1-choice8-greedy-val`.
Everything was run on the cluster (CPU node, job 19924 for the model, 19926 for the tokenizer).

## What the app gets

| File | Size | Role |
|---|---|---|
| `vigil_s512_fp32.tflite` | 1.29 GB | encoder + head, input `[2, 512]` (category row, severity row), output marker logits `[2, 8]` |
| `vigil_spec.json` | 3 KB | question prefixes, marker positions, temperatures, per-category thresholds |
| `vigil_tokenizer.bin` | 9 MB | compact BPE tokenizer (single-char ids, merges, added tokens) |

Softmax with the checkpoint temperature, thresholds and the tokenizer live outside the graph
(`vigil/ondevice.py` is the reference, the Kotlin port is in the app under `runtime/vigil`).
A conversation longer than 512 tokens keeps its most recent tokens.

## Fidelity (300 eval conversations that fit the window, 2,100 category decisions)

| Variant | File | Resident after load | Max abs prob diff vs PyTorch | Decision flips at the shipped thresholds |
|---|---|---|---|---|
| fp32 | 1,289 MB | 702 MB | 0.00006 | 4 / 2,100 |
| int8 weight-only, everything | 329 MB | 831 MB | 0.37 | 12 / 2,100 |
| int8 embedding table only | 702 MB | 664 MB | 0.17 | 5 / 2,100 |

fp32 is the one to ship. The thresholds are tiny (0.004 to 0.12), so even small probability drift moves
decisions; the 4 fp32 flips are rows sitting on a threshold. int8 weight-only saves disk but not memory
(the weights are dequantized at load: 831 MB resident), and fp16 weight casting is not supported by the
quantizer for these ops. Latency on a server CPU (8 threads) is about 140 ms per conversation, not a
phone figure.

## Tokenizer

`Bpe` (vigil/bpe.py) matches the Hugging Face tokenizer on 3,037 of 3,037 strings (3,000 eval
conversations plus edge cases: emoji, CJK, Arabic, `<eos>` typed into the text, runs of spaces).
The Kotlin port passes the same golden vectors in a JVM unit test.

## Not verified

The model has not been run on a phone. The Kotlin app compiles and its unit tests pass on the cluster
(tokenizer golden vectors, input-id parity with Python, decode logic), but the `.tflite` was only run with
the Python LiteRT interpreter. Op support of the on-device LiteRT 2.2.0 runtime, latency and memory on a real
device are untested.

## Reproduce

```
sbatch --export=ALL,MODEL=experiments/c1-choice8,THRESHOLDS=experiments/c1-choice8-greedy-val/report.json slurm/export_litert.sbatch
sbatch --export=ALL,MODEL=experiments/c1-choice8,THRESHOLDS=experiments/c1-choice8-greedy-val/report.json slurm/export_ondevice_assets.sbatch
```
