"""Break down errors on a generated test set: benign false alarms by scene/language, recall by category/language.

Usage: python scripts/breakdown_xgen.py <eval-output-dir> <generated.jsonl>
The eval output dir holds rows.json and report.json; row order must match the generated file (all rows scored).
"""
import json
import sys
from collections import defaultdict

from vigil.labels import CATEGORIES
from vigil.metrics import flagged

out_dir, data = sys.argv[1], sys.argv[2]
meta = [json.loads(line)["meta"] for line in open(data, encoding="utf-8")]
rows = json.load(open(f"{out_dir}/rows.json"))
th = json.load(open(f"{out_dir}/report.json"))["thresholds"]
assert len(rows) == len(meta), "rows.json must cover the whole generated file in order"
fa, rec = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
for r, m in zip(rows, meta):
    fl = flagged(r["probs"], th)
    if m["category"] == "benign":
        for k in (m["scene"][:46], "[" + m["language"] + "]"):
            fa[k][1] += 1
            fa[k][0] += bool(fl)
    else:
        rec[(m["category"], m["language"][:2])][1] += 1
        rec[(m["category"], m["language"][:2])][0] += m["category"] in fl
print(f"benign false alarms by scene / language ({out_dir})")
for k, (a, n) in sorted(fa.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
    print(f"  {a / n:5.0%} ({a:3d}/{n:3d})  {k}")
print("recall by category and language (En / Es)")
for c in CATEGORIES:
    e, s = rec[(c, "En")], rec[(c, "Sp")]
    print(f"  {c:19s} En {e[0] / max(1, e[1]):5.0%} ({e[1]})  Es {s[0] / max(1, s[1]):5.0%} ({s[1]})")
