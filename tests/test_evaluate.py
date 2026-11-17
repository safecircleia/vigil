import json

import pytest

from vigil.evaluate import HORIZON_REFERENCE, load_rows, render_markdown, summarize
from vigil.labels import CATEGORIES


def _rows():
    rows = []
    for i in range(20):
        risky = i % 2 == 0
        rows.append({
            "gold_categories": frozenset({"grooming"}) if risky else frozenset(),
            "gold_severity": 3 if risky else 0,
            "probs": {c: (0.9 if (c == "grooming" and risky) else 0.05) for c in CATEGORIES},
            "severity_probs": [0.0, 0.0, 0.0, 1.0, 0.0] if risky else [1.0, 0.0, 0.0, 0.0, 0.0],
            "latency_ms": 10.0 + i,
        })
    return rows


def test_summarize_picks_thresholds_on_val_and_scores_on_test():
    summary = summarize(_rows(), min_precision=0.8, seed=1)
    assert set(summary["thresholds"]) == set(CATEGORIES)
    assert summary["n_val"] + summary["n_test"] == 20
    hits, support = summary["per_category"]["grooming"]["recall"]
    assert hits == support > 0
    assert summary["catch_rate"]["value"] == 1.0
    assert summary["severity_macro_f1"] == 1.0
    assert summary["latency_ms"]["p50"] > 0


def test_markdown_states_horizon_reference_and_caveats():
    md = render_markdown(summarize(_rows(), min_precision=0.8, seed=1), model="x", device="cpu")
    assert "90.36" in md and "regex" in md
    assert "calibration" in md.lower() and "validation half" in md
    json.dumps(HORIZON_REFERENCE)


def test_summarize_includes_precision_and_benign_false_alarms():
    summary = summarize(_rows(), min_precision=0.8, seed=1)
    tp, flagged_n = summary["per_category"]["grooming"]["precision"]
    assert 0 <= tp <= flagged_n
    fa = summary["benign_false_alarms"]
    assert 0 <= fa["hits"] <= fa["support"]
    assert 0.0 <= summary["risky_share_test"] <= 1.0
    md = render_markdown(summary, model="x", device="cpu")
    assert "Precision" in md and "benign" in md.lower() and "prevalence" in md.lower()


def test_summarize_reports_auc_recall_at_half_and_category_catch_rate():
    summary = summarize(_rows(), min_precision=0.8, seed=1)
    g = summary["per_category"]["grooming"]
    assert g["auc"] == 1.0
    hits, support = g["recall_at_half"]
    assert 0 <= hits <= support
    cc = summary["catch_rate_category"]
    assert 0 <= cc["hits"] <= cc["support"]
    md = render_markdown(summary, model="x", device="cpu")
    assert "AUC" in md and "correct category" in md.lower()


def test_markdown_states_excluded_overflow_rows():
    summary = summarize(_rows(), min_precision=0.8, seed=1)
    summary["sampling"] = {"sampled": 100, "overflow_excluded": 7}
    md = render_markdown(summary, model="x", device="cpu")
    assert "7 of 100" in md and "not scored" in md


class _FakeAgent:
    def predict(self, state, questions):
        answers = {c: {"noul": 0.1} for c in CATEGORIES}
        answers["severity"] = {"probabilities": {"0": 1.0}}
        return {"answers": answers}


def _line(state, target):
    return json.dumps({"messages": [
        {"role": "user", "content": f"Analyze this conversation:\n{state}"},
        {"role": "assistant", "content": json.dumps(target)},
    ]})


@pytest.mark.integration
def test_load_rows_excludes_and_counts_conversations_longer_than_the_window(tmp_path, tok):
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join([
        _line("Child: hi\nOther: hey", {"risk_level": "none", "categories": []}),
        _line("Child: hello there\nOther: hi\n" * 400, {"risk_level": "none", "categories": []}),
        _line("Child: hi\nOther: how old are you?", {"risk_level": "high", "categories": ["grooming"]}),
    ]) + "\n")
    rows, stats = load_rows(str(path), _FakeAgent(), max_rows=10, seed=0, tok=tok)
    assert len(rows) == 2
    assert stats == {"sampled": 3, "overflow_excluded": 1}


class _FakeChoiceAgent:
    def predict(self, state, questions):
        assert set(questions) == {"category", "severity"}
        probs = {"benign": 0.3, **{c: 0.1 for c in CATEGORIES}}
        return {"answers": {"category": {"probabilities": probs},
                            "severity": {"probabilities": {"0": 1.0}}}}


@pytest.mark.integration
def test_load_rows_reads_category_probabilities_for_the_choice_design(tmp_path, tok):
    path = tmp_path / "eval.jsonl"
    path.write_text(_line("Child: hi\nOther: hey", {"risk_level": "none", "categories": []}) + "\n")
    rows, _ = load_rows(str(path), _FakeChoiceAgent(), max_rows=5, seed=0, tok=tok, design="choice8")
    assert set(rows[0]["probs"]) == set(CATEGORIES) and rows[0]["probs"]["grooming"] == 0.1


def test_summarize_with_fixed_thresholds_scores_every_row_and_does_not_retune():
    fixed = {c: 0.5 for c in CATEGORIES}
    fixed["grooming"] = 0.99  # deliberately above every grooming score in the fixture
    summary = summarize(_rows(), min_precision=0.8, seed=1, fixed_thresholds=fixed)
    assert summary["n_test"] == 20 and summary["n_val"] == 0
    assert summary["thresholds"] == fixed
    assert summary["per_category"]["grooming"]["recall"] == (0, 10)  # the fixed threshold is honoured
    md = render_markdown(summary, model="x", device="cpu")
    assert "fixed" in md.lower() and "validation half" not in md


def test_summarize_with_a_false_alarm_budget_reports_the_policy():
    summary = summarize(_rows(), min_precision=0.8, seed=1, max_benign_fa=0.02)
    policy = summary["threshold_policy"]
    assert policy["max_benign_fa"] == 0.02 and policy["precision_target"] in (0.8, 0.9, 0.95, 0.98)
    assert isinstance(policy["met"], bool) and set(summary["thresholds"]) == set(CATEGORIES)
    assert "false-alarm budget" in render_markdown(summary, model="x", device="cpu").lower()


def test_summarize_with_the_greedy_rule_reports_it():
    summary = summarize(_rows(), min_precision=0.8, seed=1, max_benign_fa=0.02, threshold_rule="greedy")
    policy = summary["threshold_policy"]
    assert policy["rule"] == "greedy" and isinstance(policy["met"], bool)
    assert set(summary["thresholds"]) == set(CATEGORIES)
    assert "per-category" in render_markdown(summary, model="x", device="cpu").lower()
