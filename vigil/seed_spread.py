"""How much do results move from one training run to the next? Mean, sd and range over several report.json files.

Usage: python -m vigil.seed_spread label=dirA,dirB,dirC [label2=dirs ...]   (each dir holds a report.json)
"""
from __future__ import annotations

import json
import statistics
import sys


def _sd(xs: list) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def spread(reports: list) -> dict:
    fa = [r["benign_false_alarms"]["hits"] / r["benign_false_alarms"]["support"] for r in reports]
    ch = [r["catch_rate_category"]["hits"] / r["catch_rate_category"]["support"] for r in reports]
    return {"n": len(reports), "fa_mean": statistics.mean(fa), "fa_sd": _sd(fa), "fa_min": min(fa), "fa_max": max(fa),
            "catch_mean": statistics.mean(ch), "catch_sd": _sd(ch), "catch_min": min(ch), "catch_max": max(ch)}


def main() -> None:
    for arg in sys.argv[1:]:
        label, dirs = arg.split("=", 1)
        s = spread([json.load(open(f"{d}/report.json")) for d in dirs.split(",")])
        print(f"{label:34s} n={s['n']}  false alarms {s['fa_mean']:5.1%} +/- {s['fa_sd']:4.1%} [{s['fa_min']:5.1%}-{s['fa_max']:5.1%}]"
              f"  catch(correct) {s['catch_mean']:5.1%} +/- {s['catch_sd']:4.1%} [{s['catch_min']:5.1%}-{s['catch_max']:5.1%}]")


if __name__ == "__main__":
    main()
