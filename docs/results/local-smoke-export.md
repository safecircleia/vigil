# ONNX export check on a local smoke checkpoint

**This is not a pilot result.** The checkpoint was trained for 20 CPU micro-steps on 4,000 items
(500 conversations), so its accuracy is meaningless. It exists only to verify the export mechanics
and to take first size and latency measurements. Re-run on the real pilot checkpoint once the
cluster run (plan Task 8) is unblocked.

Machine: 16-core CPU workstation, onnxruntime CPU provider. Data: 25 real eval conversations.

| Check | Result |
|---|---|
| fp32 ONNX vs PyTorch, 8 questions x 25 conversations | max abs probability diff 0.0000, 0 decision flips (outputs vary: 143 distinct values in 0.03-0.15) |
| int8 (dynamic, weight-only) vs fp32 ONNX | max abs diff 0.4693, 11 of 175 noul decisions flip |
| fp32 file size | 1.2 GB (2.7 MB graph + 1.2 GB data) |
| int8 file size | 884 MB |
| Latency per conversation (8 questions), fp32 ONNX | p50 9.8 s, p95 15.6 s |
| Latency per conversation (8 questions), int8 ONNX | p50 8.0 s, p95 13.4 s |
| Latency per conversation, PyTorch Agent (base model, 40 rows) | p50 3.0 s, p95 4.5 s |

Findings:

1. The upstream export script traces with batch size 1, which bakes a `[1,1,S,S]` attention mask into
   the graph: it runs for batch 1 at any sequence length and fails for any batch > 1. Vigil's copy
   traces with batch 2 (see `NOTICE`). The parity test is the regression test.
2. Dynamic int8 quantizes only MatMul weights, so the large multilingual embedding table stays fp32:
   884 MB, not the ~330 MB the spec estimated. It also flipped 11 of 175 decisions on this checkpoint;
   int8 must be re-measured on a trained checkpoint before it is used.
3. Eight questions per conversation means eight encoder passes over the full conversation. CPU
   latency is several seconds, far from the 500 ms target. See the design question in the run summary.
