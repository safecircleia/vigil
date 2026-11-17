import pytest

from vigil.metrics import (pick_thresholds_greedy, pick_thresholds_with_fa_budget, auc, benign_false_alarms, catch_rate, category_catch_rate, flagged, mcnemar_exact, per_category_recall,
                           pick_threshold, precision, severity_macro_f1, split_val_test, wilson_interval)


def _row(gold, sev, **probs):
    return {"gold_categories": frozenset(gold), "gold_severity": sev, "probs": probs,
            "severity_probs": [0.2] * 5}


def test_flagged_uses_per_category_thresholds():
    probs = {"grooming": 0.6, "threats": 0.6}
    assert flagged(probs, {"grooming": 0.5, "threats": 0.7}) == frozenset({"grooming"})


def test_per_category_recall_counts_hits_over_support():
    rows = [_row({"grooming"}, 3, grooming=p) for p in (0.9, 0.8, 0.4, 0.1)]
    rows.append(_row(set(), 0, grooming=0.99))  # not grooming: does not count as support
    assert per_category_recall(rows, {"grooming": 0.5})["grooming"] == (2, 4)


def test_catch_rate_ignores_benign_rows():
    rows = [
        _row({"threats"}, 2, threats=0.9),
        _row({"grooming"}, 3, grooming=0.9),
        _row({"bullying"}, 1, bullying=0.1),
        _row(set(), 0, threats=0.99),
    ]
    th = {"threats": 0.5, "grooming": 0.5, "bullying": 0.5}
    assert catch_rate(rows, th) == (2, 3)


def test_catch_rate_with_no_risky_rows():
    assert catch_rate([_row(set(), 0, threats=0.9)], {"threats": 0.5}) == (0, 0)


def test_severity_macro_f1_hand_computed():
    assert severity_macro_f1([0, 1, 2, 2], [0, 1, 2, 1]) == pytest.approx(0.7778, abs=1e-3)


def test_pick_threshold_prefers_high_recall_then_high_threshold():
    scores = [0.9, 0.8, 0.7, 0.3, 0.2]
    golds = [True, True, False, True, False]
    assert pick_threshold(scores, golds, 0.6) == (0.3, True)


def test_pick_threshold_falls_back_when_precision_unreachable():
    assert pick_threshold([0.9, 0.8], [False, False], 0.5) == (0.5, False)


def test_wilson_interval():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_exact():
    assert mcnemar_exact(10, 0) == pytest.approx(0.001953125)
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert mcnemar_exact(0, 0) == 1.0


def test_split_val_test_is_disjoint_and_deterministic():
    val, test = split_val_test(101, seed=3)
    assert set(val).isdisjoint(test) and sorted(val + test) == list(range(101))
    assert (val, test) == split_val_test(101, seed=3)


def test_precision_counts_true_over_flagged():
    rows = [_row({"grooming"}, 3, grooming=0.9), _row(set(), 0, grooming=0.9), _row(set(), 0, grooming=0.1)]
    assert precision(rows, {"grooming": 0.5}, "grooming") == (1, 2)


def test_precision_with_nothing_flagged():
    assert precision([_row(set(), 0, grooming=0.1)], {"grooming": 0.5}, "grooming") == (0, 0)


def test_benign_false_alarms_count_only_benign_rows():
    rows = [_row(set(), 0, threats=0.9), _row(set(), 0, threats=0.1), _row({"threats"}, 2, threats=0.9)]
    assert benign_false_alarms(rows, {"threats": 0.5}) == (1, 2)


def test_auc_perfect_reversed_ties_and_missing_class():
    assert auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auc([0.5, 0.5], [True, False]) == 0.5
    assert auc([0.5, 0.6], [True, True]) is None


def test_category_catch_rate_requires_the_right_category():
    rows = [
        _row({"grooming"}, 3, grooming=0.1, bullying=0.9),   # flagged, but the wrong category
        _row({"grooming"}, 3, grooming=0.9, bullying=0.1),   # right category
        _row(set(), 0, grooming=0.9),                        # benign: ignored
    ]
    th = {"grooming": 0.5, "bullying": 0.5}
    assert catch_rate(rows, th) == (2, 2)          # any flag counts, even the wrong category
    assert category_catch_rate(rows, th) == (1, 2)


def _full_row(gold, sev, grooming):
    """A row with a score for every category, like real evaluation rows (only grooming varies)."""
    from vigil.labels import CATEGORIES

    return _row(gold, sev, **{c: (grooming if c == "grooming" else 0.0) for c in CATEGORIES})


def _budget_rows(benign_scores):
    rows = [_full_row({"grooming"}, 3, 0.9) for _ in range(5)]
    rows += [_full_row({"grooming"}, 3, 0.5) for _ in range(5)]
    rows += [_full_row(set(), 0, s) for s in benign_scores]
    return rows


def test_fa_budget_picks_the_smallest_precision_target_that_fits_the_budget():
    rows = _budget_rows([0.5, 0.5] + [0.0] * 48)  # precision 0.8 lets both benign 0.5 rows through: 4% > 2%
    policy = pick_thresholds_with_fa_budget(rows, max_fa=0.02)
    assert policy["precision_target"] == 0.9 and policy["met"] is True
    assert policy["thresholds"]["grooming"] == 0.9
    assert policy["benign_false_alarms"] == (0, 50)


def test_fa_budget_keeps_the_first_target_when_it_already_fits():
    policy = pick_thresholds_with_fa_budget(_budget_rows([0.0] * 50), max_fa=0.02)
    assert policy["precision_target"] == 0.8 and policy["met"] is True


def test_fa_budget_falls_back_to_the_strictest_target_and_says_it_missed():
    rows = [_full_row({"grooming"}, 3, 0.9) for _ in range(5)] + [_full_row(set(), 0, 0.9) for _ in range(5)]
    policy = pick_thresholds_with_fa_budget(rows, max_fa=0.02)
    assert policy["met"] is False and policy["precision_target"] == 0.98


def _scored_row(gold, sev, **scores):
    from vigil.labels import CATEGORIES

    return _row(gold, sev, **{c: scores.get(c, 0.0) for c in CATEGORIES})


def _two_category_rows():
    rows = [_scored_row({"grooming"}, 3, grooming=0.9) for _ in range(10)]
    rows += [_scored_row({"bullying"}, 2, bullying=0.9) for _ in range(10)]
    rows += [_scored_row(set(), 0, grooming=0.5) for _ in range(10)]   # already below every grooming positive
    rows += [_scored_row(set(), 0, bullying=0.95) for _ in range(10)]  # above every bullying positive
    return rows


def test_greedy_keeps_full_recall_when_the_budget_already_fits():
    policy = pick_thresholds_greedy(_two_category_rows(), max_fa=0.5)
    assert policy["thresholds"]["grooming"] == 0.9 and policy["thresholds"]["bullying"] == 0.9
    assert policy["met"] is True and policy["benign_false_alarms"] == (10, 20)


def test_greedy_raises_only_the_category_that_must_give_up_recall():
    policy = pick_thresholds_greedy(_two_category_rows(), max_fa=0.0)
    assert policy["thresholds"]["bullying"] > 0.95 and policy["thresholds"]["grooming"] == 0.9
    assert policy["met"] is True and policy["benign_false_alarms"] == (0, 20)


def test_greedy_gives_up_the_cheapest_recall_first():
    rows = [_scored_row({"grooming"}, 3, grooming=0.9) for _ in range(9)] + [_scored_row({"grooming"}, 3, grooming=0.6)]
    rows += [_scored_row({"bullying"}, 2, bullying=0.9) for _ in range(5)] + [_scored_row({"bullying"}, 2, bullying=0.7) for _ in range(5)]
    rows += [_scored_row(set(), 0, grooming=0.7) for _ in range(5)] + [_scored_row(set(), 0, bullying=0.8) for _ in range(5)]
    policy = pick_thresholds_greedy(rows, max_fa=0.5)  # 10 benign rows, all flagged at first; must free 5
    assert policy["thresholds"]["grooming"] > 0.7      # costs 1 of 10 grooming positives
    assert policy["thresholds"]["bullying"] == 0.7     # would have cost 5 of 10 bullying positives
    assert policy["met"] is True


def test_greedy_needs_benign_rows_to_budget_false_alarms():
    with pytest.raises(ValueError):
        pick_thresholds_greedy([_scored_row({"grooming"}, 3, grooming=0.9)], max_fa=0.02)


def test_greedy_can_clear_benign_rows_flagged_by_several_categories():
    rows = [_scored_row({"grooming"}, 3, grooming=0.9) for _ in range(10)]
    rows += [_scored_row({"bullying"}, 2, bullying=0.9) for _ in range(10)]
    rows += [_scored_row(set(), 0, grooming=0.95, bullying=0.95) for _ in range(10)]  # each flagged twice
    policy = pick_thresholds_greedy(rows, max_fa=0.0)
    assert policy["met"] is True and policy["benign_false_alarms"] == (0, 10)
