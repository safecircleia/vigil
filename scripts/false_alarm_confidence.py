"""How confident are the false alarms? Buckets the top flagged probability of each flagged benign row.

Usage: python scripts/false_alarm_confidence.py <eval-output-dir> <generated.jsonl>
"""
import json
import sys

from vigil.metrics import flagged

out_dir, data = sys.argv[1], sys.argv[2]
meta = [json.loads(line)["meta"] for line in open(data, encoding="utf-8")]
rows = json.load(open(f"{out_dir}/rows.json"))
th = json.load(open(f"{out_dir}/report.json"))["thresholds"]
assert len(rows) == len(meta)
tops = []
for r, m in zip(rows, meta):
    if m["category"] != "benign":
        continue
    fl = flagged(r["probs"], th)
    if fl:
        tops.append(max(r["probs"][c] for c in fl))
n_benign = sum(m["category"] == "benign" for m in meta)
buckets = [("< 0.1", lambda p: p < 0.1), ("0.1 - 0.5", lambda p: 0.1 <= p < 0.5), (">= 0.5", lambda p: p >= 0.5)]
print(f"{out_dir}: {len(tops)}/{n_benign} benign rows flagged")
for name, f in buckets:
    k = sum(f(p) for p in tops)
    print(f"   top flagged prob {name:9s}: {k:3d} rows ({k / n_benign:5.1%} of benign)")
