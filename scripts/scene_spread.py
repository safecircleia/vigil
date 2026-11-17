"""Benign false-alarm rate per scene, averaged over several runs of a group (both unseen generators pooled).

Usage: python scripts/scene_spread.py <design> <rule> <GROUP=run1,run2,...> [<GROUP2=...>]
Reads experiments/<run>-<design>-<rule>-{llama,qwen3}/{rows,report}.json and data/xgen/ files for the scene of each row.
"""
import json
import statistics
import sys
from collections import defaultdict

from vigil.metrics import flagged

design, rule = sys.argv[1], sys.argv[2]
DATA = {"llama": "data/xgen/xgen-llama31-8b.jsonl", "qwen3": "data/xgen/xgen-qwen3-test.jsonl"}
meta = {k: [json.loads(line)["meta"] for line in open(p, encoding="utf-8")] for k, p in DATA.items()}
groups = {}
for arg in sys.argv[3:]:
    name, runs = arg.split("=", 1)
    per_run = []
    for run in runs.split(","):
        fa = defaultdict(lambda: [0, 0])
        for gen in DATA:
            d = f"experiments/{run}-{design}-{rule}-{gen}"
            rows, th = json.load(open(f"{d}/rows.json")), json.load(open(f"{d}/report.json"))["thresholds"]
            for r, m in zip(rows, meta[gen]):
                if m["category"] == "benign":
                    fa[m["scene"][:44]][1] += 1
                    fa[m["scene"][:44]][0] += bool(flagged(r["probs"], th))
        per_run.append(fa)
    groups[name] = per_run
scenes = sorted({s for runs in groups.values() for fa in runs for s in fa})
print(f"{design}/{rule}: benign false-alarm rate by scene, mean over runs (both unseen generators pooled)")
print(f"{'scene':46s} " + " ".join(f"{g:>12s}" for g in groups))
for s in scenes:
    cells = []
    for runs in groups.values():
        rates = [fa[s][0] / fa[s][1] for fa in runs if fa[s][1]]
        cells.append(f"{statistics.mean(rates):11.0%} " if rates else f"{'-':>12s}")
    n = max(fa[s][1] for runs in groups.values() for fa in runs)
    print(f"{s:46s} " + " ".join(cells) + f"   (n={n} rows per run)")
