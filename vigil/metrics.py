from __future__ import annotations

import bisect
import math
import random

from scipy.stats import binomtest

from vigil.labels import CATEGORIES


def flagged(probs: dict, thresholds: dict) -> frozenset:
    return frozenset(c for c, p in probs.items() if p >= thresholds.get(c, 0.5))


def per_category_recall(rows: list, thresholds: dict) -> dict:
    out = {}
    for cat in CATEGORIES:
        support = [r for r in rows if cat in r["gold_categories"]]
        hits = sum(cat in flagged(r["probs"], thresholds) for r in support)
        out[cat] = (hits, len(support))
    return out


def precision(rows: list, thresholds: dict, category: str) -> tuple:
    """(true positives, flagged) for one category."""
    flagged_rows = [r for r in rows if category in flagged(r["probs"], thresholds)]
    return sum(category in r["gold_categories"] for r in flagged_rows), len(flagged_rows)


def benign_false_alarms(rows: list, thresholds: dict) -> tuple:
    """(benign rows with any category flagged, benign rows)."""
    benign = [r for r in rows if r["gold_severity"] == 0]
    return sum(bool(flagged(r["probs"], thresholds)) for r in benign), len(benign)


def catch_rate(rows: list, thresholds: dict) -> tuple:
    risky = [r for r in rows if r["gold_severity"] > 0]
    caught = sum(bool(flagged(r["probs"], thresholds)) for r in risky)
    return caught, len(risky)


def category_catch_rate(rows: list, thresholds: dict) -> tuple:
    """(rows where at least one *gold* category is flagged, rows with a gold category)."""
    with_gold = [r for r in rows if r["gold_categories"]]
    hits = sum(bool(r["gold_categories"] & flagged(r["probs"], thresholds)) for r in with_gold)
    return hits, len(with_gold)


def auc(scores: list, golds: list):
    """Threshold-free ranking quality (Mann-Whitney AUC, ties count half); None if a class is missing."""
    pos = [s for s, g in zip(scores, golds) if g]
    neg = [s for s, g in zip(scores, golds) if not g]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def severity_macro_f1(gold: list, pred: list) -> float:
    classes = sorted(set(gold) | set(pred))
    scores = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def pick_threshold(scores: list, golds: list, min_precision: float) -> tuple:
    total_pos = sum(golds)
    best = None  # (recall, threshold)
    for thr in sorted(set(scores)):
        predicted = [s >= thr for s in scores]
        tp = sum(p and g for p, g in zip(predicted, golds))
        n_pred = sum(predicted)
        if n_pred == 0 or total_pos == 0 or tp / n_pred < min_precision:
            continue
        candidate = (tp / total_pos, thr)
        if best is None or candidate >= best:
            best = candidate
    return (best[1], True) if best else (0.5, False)


PRECISION_TARGETS = (0.8, 0.9, 0.95, 0.98)


def pick_thresholds_with_fa_budget(rows: list, max_fa: float, targets: tuple = PRECISION_TARGETS) -> dict:
    """Smallest precision target whose thresholds keep the benign false-alarm rate within `max_fa`.

    Decided on the rows given (validation only); if no target fits, the strictest one is returned
    with met=False.
    """
    for i, target in enumerate(targets):
        thresholds = {
            cat: pick_threshold([r["probs"][cat] for r in rows], [cat in r["gold_categories"] for r in rows], target)[0]
            for cat in CATEGORIES
        }
        hits, n = benign_false_alarms(rows, thresholds)
        fits = n > 0 and hits / n <= max_fa
        if fits or i == len(targets) - 1:
            return {"thresholds": thresholds, "precision_target": target,
                    "benign_false_alarms": (hits, n), "met": bool(fits)}


def pick_thresholds_greedy(rows: list, max_fa: float) -> dict:
    """Per-category thresholds that keep as much validation recall as possible within a benign
    false-alarm budget (share of benign rows with any category flagged).

    Starts every threshold at the lowest positive score (full recall), then repeatedly raises the
    threshold of whichever category frees the most flagged benign rows per unit of recall it gives up,
    looking ahead over any jump length, until the budget is met. Decided on the rows given.
    """
    benign = [r for r in rows if r["gold_severity"] == 0]
    if not benign:
        raise ValueError("need benign rows to budget false alarms")
    allowed = max_fa * len(benign)
    thresholds, active, n_pos = {}, [], {}
    for cat in CATEGORIES:
        pos = [r["probs"][cat] for r in rows if cat in r["gold_categories"]]
        n_pos[cat] = len(pos)
        thresholds[cat] = min(pos) if pos else 0.5  # a category without validation positives cannot be tuned
        if pos:
            active.append(cat)
    counts = [sum(r["probs"][c] >= thresholds[c] for c in CATEGORIES) for r in benign]
    fa = sum(c > 0 for c in counts)
    while fa > allowed + 1e-9:
        best = None  # (ratio, -recall_lost, category, new_threshold)
        for cat in active:
            t0 = thresholds[cat]
            pos_sorted = sorted(r["probs"][cat] for r in rows if cat in r["gold_categories"] and r["probs"][cat] >= t0)
            # a benign row flagged by k categories is cleared by k raises, so each raise earns 1/k of it
            flagged_now = sorted((b["probs"][cat], 1.0 / k) for b, k in zip(benign, counts) if b["probs"][cat] >= t0 and k > 0)
            scores_sorted = [p for p, _ in flagged_now]
            cumulative = [0.0]
            for _, w in flagged_now:
                cumulative.append(cumulative[-1] + w)
            candidates = sorted({r["probs"][cat] for r in rows if r["probs"][cat] > t0}) + [2.0]
            for new_t in candidates:
                removed = cumulative[bisect.bisect_left(scores_sorted, new_t)]
                if removed <= 0:
                    continue
                lost = bisect.bisect_left(pos_sorted, new_t) / n_pos[cat]
                key = (removed / (lost + 1e-9), -lost)
                if best is None or key > best[:2]:
                    best = (*key, cat, new_t)
        if best is None:
            break
        _, _, cat, new_t = best
        old_t = thresholds[cat]
        for i, b in enumerate(benign):
            if old_t <= b["probs"][cat] < new_t:
                counts[i] -= 1
        thresholds[cat] = new_t
        fa = sum(c > 0 for c in counts)
    return {"thresholds": thresholds, "benign_false_alarms": (fa, len(benign)), "met": fa <= allowed + 1e-9}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def mcnemar_exact(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def split_val_test(n: int, seed: int) -> tuple:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    half = n // 2
    return sorted(order[:half]), sorted(order[half:])
