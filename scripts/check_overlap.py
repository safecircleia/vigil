"""Audit: how many eval conversations also appear in the training file?

Exact = identical conversation text. Normalised = same after lowercasing and collapsing whitespace.
Reports the whole eval file and the 5,000-row sample the evaluator uses (same seed and sampling).
"""
import argparse
import json
import random
import re

from vigil.rows import RowError, split_messages


def _states(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield split_messages(json.loads(line)["messages"])[0]
                except (KeyError, RowError):
                    yield None


def _norm(state):
    return re.sub(r"\s+", " ", state.lower()).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/source/train.jsonl")
    p.add_argument("--eval", default="data/source/eval.jsonl")
    p.add_argument("--sample", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260922)
    a = p.parse_args()

    exact, normal = set(), set()
    for s in _states(a.train):
        if s is not None:
            exact.add(hash(s))
            normal.add(hash(_norm(s)))
    eval_states = list(_states(a.eval))
    picked = set(random.Random(a.seed).sample(range(len(eval_states)), min(a.sample, len(eval_states))))
    for name, idx in (("whole eval", range(len(eval_states))), (f"evaluator sample ({a.sample})", sorted(picked))):
        idx = [i for i in idx if eval_states[i] is not None]
        e = sum(hash(eval_states[i]) in exact for i in idx)
        n = sum(hash(_norm(eval_states[i])) in normal for i in idx)
        print(f"{name}: {len(idx)} rows | exact duplicates of a train conversation: {e} ({e / len(idx):.2%}) "
              f"| after normalising: {n} ({n / len(idx):.2%})")
    print(f"distinct train conversations: {len(exact)}")


if __name__ == "__main__":
    main()
