"""Mixed-generator validation file for choosing thresholds.

Takes Horizon eval rows that the evaluator's test sample does NOT use (so thresholds are never picked
on test rows) and appends validation rows written by other generators.
"""
from __future__ import annotations

import argparse
import json
import random

from vigil.evaluate import sample_indices


def build(horizon_eval: str, generated: list, out: str, *, n_horizon: int, seed: int,
          exclude_max_rows: int = 5000, exclude_seed: int = 20260922) -> dict:
    with open(horizon_eval, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    excluded = set(sample_indices(len(lines), exclude_max_rows, exclude_seed))
    candidates = [i for i in range(len(lines)) if i not in excluded]
    chosen = sorted(random.Random(seed).sample(candidates, min(n_horizon, len(candidates))))
    rows = [lines[i].rstrip("\n") for i in chosen]
    n_generated = 0
    for path in generated:
        with open(path, encoding="utf-8") as f:
            extra = [line.rstrip("\n") for line in f if line.strip()]
        rows += extra
        n_generated += len(extra)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(r + "\n" for r in rows)
    return {"horizon": len(chosen), "generated": n_generated, "total": len(rows)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build a mixed-generator validation file")
    p.add_argument("--horizon-eval", default="data/source/eval.jsonl")
    p.add_argument("--generated", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-horizon", type=int, default=900)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    print(json.dumps(build(a.horizon_eval, a.generated, a.out, n_horizon=a.n_horizon, seed=a.seed)))


if __name__ == "__main__":
    main()
