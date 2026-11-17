"""Print benign conversations of one scene that the model flagged, to judge by eye.

Usage: python scripts/audit_false_alarms.py <eval-output-dir> <generated.jsonl> <scene-substring> <n> [seed]
Row order in rows.json must match the generated file (all rows scored).
"""
import json
import random
import sys

from vigil.metrics import flagged

out_dir, data, scene, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
rows_meta = [json.loads(line) for line in open(data, encoding="utf-8")]
scored = json.load(open(f"{out_dir}/rows.json"))
th = json.load(open(f"{out_dir}/report.json"))["thresholds"]
assert len(rows_meta) == len(scored)
hits = []
for row, sc in zip(rows_meta, scored):
    m = row["meta"]
    if m["category"] == "benign" and scene in m["scene"]:
        fl = flagged(sc["probs"], th)
        if fl:
            hits.append((m, row, {c: round(sc["probs"][c], 3) for c in fl}))
print(f"{len(hits)} flagged benign rows in scenes matching {scene!r} ({out_dir}); showing {min(n, len(hits))}")
for m, row, fl in random.Random(seed).sample(hits, min(n, len(hits))):
    lines = row["messages"][0]["content"].split("\n")[1:]
    print(f"\n=== {m['id']} | {m['language'][:2]} | flagged {fl} | {len(lines)} turns")
    for ln in lines[:10]:
        print("   ", ln[:150])
    if len(lines) > 10:
        print("    ...")
