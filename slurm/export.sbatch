#!/bin/bash
#SBATCH --job-name=vigil_export
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${VIGIL_DIR:-$SLURM_SUBMIT_DIR}"
uv sync --frozen --extra export
SRC="${CHECKPOINT:?set CHECKPOINT via --export=ALL,CHECKPOINT=...}"
# Work on node-local disk: onnxruntime memory-maps the 1.2 GB external-data file, and the job died
# with SIGBUS (exit 135) when that file lived on the NFS-backed /slurm filesystem.
WORK="$(mktemp -d /tmp/vigil-export-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
CK="$WORK/ckpt"
cp -r "$SRC" "$CK"
rm -rf "$CK/checkpoint_latest"
uv run python scripts/export_onnx.py --model "$CK" --output "$CK/vigil.onnx" --quantize
du -h "$CK"/vigil*.onnx*
VIGIL_CHECKPOINT="$CK" VIGIL_EVAL=data/source/eval.jsonl VIGIL_DESIGN="${DESIGN:-noul7}" uv run pytest tests/test_export_parity.py -q -s -W ignore
DESIGN="${DESIGN:-noul7}" uv run python -W ignore - "$CK" <<'PY'
import json, os, sys, time
from laya.onnx_agent import ONNXAgent
from vigil.questions import build_questions
from vigil.rows import split_messages
from vigil.evaluate import _category_probs
from vigil.labels import CATEGORIES

ck = sys.argv[1]
states = []
for line in open("data/source/eval.jsonl", encoding="utf-8"):
    states.append(split_messages(json.loads(line)["messages"])[0])
    if len(states) == 25:
        break
DESIGN = os.environ.get("DESIGN", "noul7")
q = build_questions(DESIGN)
out = {}
for name in ("vigil.onnx", "vigil.int8.onnx"):
    agent = ONNXAgent(ck, onnx_path=os.path.join(ck, name))
    agent.predict(states[0], q)  # warm-up
    lat, probs = [], []
    for s in states:
        t0 = time.perf_counter()
        a = agent.predict(s, q)["answers"]
        lat.append((time.perf_counter() - t0) * 1000)
        p = _category_probs(a, DESIGN)
        probs.append([p[c] for c in CATEGORIES])
    lat.sort()
    out[name] = {"p50_ms": round(lat[len(lat) // 2]), "p95_ms": round(lat[int(len(lat) * 0.95)]), "probs": probs}
fp, i8 = out["vigil.onnx"]["probs"], out["vigil.int8.onnx"]["probs"]
drift = max(abs(a - b) for r1, r2 in zip(fp, i8) for a, b in zip(r1, r2))
flips = sum((a >= .5) != (b >= .5) for r1, r2 in zip(fp, i8) for a, b in zip(r1, r2))
print(json.dumps({"cpus": os.cpu_count(),
                  "fp32_latency_ms": {k: v for k, v in out["vigil.onnx"].items() if k != "probs"},
                  "int8_latency_ms": {k: v for k, v in out["vigil.int8.onnx"].items() if k != "probs"},
                  "int8_vs_fp32_max_abs_diff": round(drift, 4), "int8_decision_flips": f"{flips}/{sum(len(r) for r in fp)}"}))
PY
cp "$CK"/vigil.onnx "$CK"/vigil.onnx.data "$CK"/vigil.int8.onnx "$SRC"/
