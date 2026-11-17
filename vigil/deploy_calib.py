"""Per-deployment calibration, offline: what if thresholds also see a small sample of the target traffic?

Pretend one scored set is "the deployment". Baseline thresholds come from the validation rows only; the
calibrated ones from validation plus a random sample of the deployment (optionally upweighted). Both are then
measured on the deployment rows that were NOT sampled. Works on rows.json files, so no model is run.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics

from vigil.metrics import (benign_false_alarms, category_catch_rate, pick_thresholds_greedy,
                           pick_thresholds_with_fa_budget)


def _load(path: str) -> list:
    with open(path) as f:
        rows = json.load(f)
    for r in rows:
        r["gold_categories"] = frozenset(r["gold_categories"])
    return rows


def _pick(rows: list, max_fa: float, rule: str) -> dict:
    return (pick_thresholds_greedy if rule == "greedy" else pick_thresholds_with_fa_budget)(rows, max_fa)["thresholds"]


def _metrics(rows: list, thresholds: dict) -> dict:
    hits, n = benign_false_alarms(rows, thresholds)
    ch, cn = category_catch_rate(rows, thresholds)
    return {"fa": hits / n if n else 0.0, "catch": ch / cn if cn else 0.0, "thresholds": thresholds}


def calibrate_on_deployment(val_rows: list, dep_rows: list, *, n_cal: int, weight: int, seed: int,
                            max_fa: float, rule: str = "greedy") -> dict:
    idx = list(range(len(dep_rows)))
    random.Random(seed).shuffle(idx)
    cal_idx, test_idx = sorted(idx[:n_cal]), sorted(idx[n_cal:])
    cal = [dep_rows[i] for i in cal_idx]
    test = [dep_rows[i] for i in test_idx]
    return {
        "n_cal": len(cal), "n_test": len(test), "cal_idx": cal_idx, "test_idx": test_idx,
        "baseline": _metrics(test, _pick(val_rows, max_fa, rule)),
        "calibrated": _metrics(test, _pick(val_rows + cal * weight, max_fa, rule)),
    }


def _mean_sd(xs: list) -> tuple:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def run_experiment(val_rows: list, dep_rows: list, *, n_cals: tuple, weights: tuple, repeats: int,
                   max_fa: float, rule: str = "greedy") -> dict:
    table, base_fa, base_catch = [], [], []
    for n_cal in n_cals:
        for weight in weights:
            runs = [calibrate_on_deployment(val_rows, dep_rows, n_cal=n_cal, weight=weight, seed=s,
                                            max_fa=max_fa, rule=rule) for s in range(repeats)]
            fa = [r["calibrated"]["fa"] for r in runs]
            catch = [r["calibrated"]["catch"] for r in runs]
            if not base_fa:  # baseline thresholds never change; only the held-out part differs per split
                base_fa = [r["baseline"]["fa"] for r in runs]
                base_catch = [r["baseline"]["catch"] for r in runs]
            table.append({
                "n_cal": n_cal, "weight": weight, "repeats": repeats,
                "fa_mean": _mean_sd(fa)[0], "fa_sd": _mean_sd(fa)[1],
                "catch_mean": _mean_sd(catch)[0], "catch_sd": _mean_sd(catch)[1],
                "fa_delta_mean": statistics.mean(c - b["baseline"]["fa"] for c, b in zip(fa, runs)),
                "catch_delta_mean": statistics.mean(c - b["baseline"]["catch"] for c, b in zip(catch, runs)),
            })
    return {"baseline": {"fa_mean": statistics.mean(base_fa), "catch_mean": statistics.mean(base_catch)},
            "rows": table}


def main() -> None:
    p = argparse.ArgumentParser(description="Per-deployment calibration experiment on saved scores")
    p.add_argument("--val", required=True, help="rows.json of the validation file")
    p.add_argument("--dep", action="append", required=True, metavar="NAME=ROWS.JSON")
    p.add_argument("--n-cal", type=int, nargs="+", default=[100, 200, 400])
    p.add_argument("--weights", type=int, nargs="+", default=[1, 10])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--max-fa", type=float, default=0.02)
    p.add_argument("--rule", choices=("precision", "greedy"), default="greedy")
    a = p.parse_args()
    val = _load(a.val)
    for name, path in (d.split("=", 1) for d in a.dep):
        t = run_experiment(val, _load(path), n_cals=tuple(a.n_cal), weights=tuple(a.weights),
                           repeats=a.repeats, max_fa=a.max_fa, rule=a.rule)
        print(f"\n== {name}: baseline (validation only) benign FA {t['baseline']['fa_mean']:.1%}, "
              f"catch {t['baseline']['catch_mean']:.1%}  [{a.repeats} random splits]")
        for r in t["rows"]:
            print(f"   n_cal={r['n_cal']:4d} weight={r['weight']:2d}: FA {r['fa_mean']:5.1%} +/- {r['fa_sd']:4.1%} "
                  f"(paired change {r['fa_delta_mean']:+5.1%}) | catch {r['catch_mean']:5.1%} +/- {r['catch_sd']:4.1%} "
                  f"(paired change {r['catch_delta_mean']:+5.1%})")


if __name__ == "__main__":
    main()
