import json
from collections import Counter

import pytest
import torch

from vigil.labels import Labels
from vigil.preprocess import run, stratified_indices, stratum


def test_stratum_names():
    assert stratum(Labels(frozenset(), 0)) == "benign"
    assert stratum(Labels(frozenset({"threats", "grooming"}), 3)) == "grooming+threats"


def test_stratified_indices_balances_and_caps():
    strata = ["a"] * 100 + ["b"] * 10 + ["c"] * 2
    picked = stratified_indices(strata, 30, seed=1)
    counts = Counter(strata[i] for i in picked)
    assert counts == {"a": 18, "b": 10, "c": 2}
    assert picked == sorted(picked)


def test_stratified_indices_is_deterministic():
    strata = ["a"] * 50 + ["b"] * 50
    assert stratified_indices(strata, 20, seed=7) == stratified_indices(strata, 20, seed=7)
    assert stratified_indices(strata, 20, seed=7) != stratified_indices(strata, 20, seed=8)


def test_stratified_indices_keeps_every_rare_stratum():
    strata = ["common"] * 1000 + ["rare"] * 3
    picked = stratified_indices(strata, 50, seed=0)
    assert Counter(strata[i] for i in picked)["rare"] == 3


@pytest.mark.parametrize("n", [0, -1, 5, 99])
def test_stratified_indices_returns_all_when_n_covers_everything(n):
    assert stratified_indices(["a", "b", "a", "b", "a"], n, seed=0) == [0, 1, 2, 3, 4]


def _row(state, target):
    return json.dumps({"messages": [
        {"role": "user", "content": f"Analyze this conversation:\n{state}"},
        {"role": "assistant", "content": json.dumps(target)},
    ]})


@pytest.fixture
def jsonl(tmp_path):
    rows = [
        _row("Child: hi\nOther: how old are you?", {"risk_level": "high", "categories": ["grooming"]}),
        _row("Child: hi\nOther: lol you suck", {"risk_level": "low", "categories": ["bullying"]}),
        _row("Child: hi\nOther: hey", {"risk_level": "none", "categories": []}),
        _row("Child: hi\nOther: hey", {"risk_level": "none", "categories": ["threats"]}),  # inconsistent
        _row("Child: hi\nOther: " + "blah " * 6000, {"risk_level": "none", "categories": []}),  # overflow
    ]
    path = tmp_path / "in.jsonl"
    path.write_text("\n".join(rows) + "\n")
    return path


@pytest.mark.integration
def test_run_reports_counts(tmp_path, jsonl, tok):
    report = run(str(jsonl), str(tmp_path / "items.pt"), str(tmp_path / "report.json"), tok)
    assert report["rows"] == 5 and report["malformed"] == 0
    assert report["inconsistent"] == 1
    assert report["overflowed"] == 1 and report["dropped_overflow"] == 1
    assert report["kept_conversations"] == 4 and report["items"] == 32
    assert len(torch.load(tmp_path / "items.pt", weights_only=True)) == 32
    assert json.loads((tmp_path / "report.json").read_text())["items"] == 32


@pytest.mark.integration
def test_run_fails_loudly_on_malformed_row(tmp_path, tok):
    path = tmp_path / "bad.jsonl"
    path.write_text(_row("Child: hi", {"risk_level": "extreme", "categories": []}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        run(str(path), str(tmp_path / "o.pt"), str(tmp_path / "r.json"), tok)


@pytest.mark.integration
def test_run_tolerates_malformed_up_to_limit(tmp_path, jsonl, tok):
    with jsonl.open("a") as f:
        f.write(_row("Child: hi", {"risk_level": "extreme", "categories": []}) + "\n")
    report = run(str(jsonl), str(tmp_path / "o.pt"), str(tmp_path / "r.json"), tok, max_malformed=1)
    assert report["malformed"] == 1 and report["rows"] == 6


@pytest.mark.integration
def test_run_skips_refusal_rows_without_calling_them_malformed(tmp_path, tok):
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join([
        _row("Who are you?", {"error": "I only analyze conversations for child safety risks."}),
        _row("Child: hi\nOther: hey", {"risk_level": "none", "categories": []}),
    ]) + "\n")
    report = run(str(path), str(tmp_path / "o.pt"), str(tmp_path / "r.json"), tok)
    assert report["rows"] == 2 and report["refusals_skipped"] == 1
    assert report["malformed"] == 0 and report["kept_conversations"] == 1


@pytest.mark.integration
def test_run_tags_every_item_with_its_conversation(tmp_path, jsonl, tok):
    from collections import Counter

    run(str(jsonl), str(tmp_path / "items.pt"), str(tmp_path / "report.json"), tok)
    items = torch.load(tmp_path / "items.pt", weights_only=True)
    per_conv = Counter(it["conv"] for it in items)
    assert len(per_conv) == 4 and set(per_conv.values()) == {8}


@pytest.mark.integration
def test_run_with_choice_design_makes_two_items_per_conversation(tmp_path, jsonl, tok):
    report = run(str(jsonl), str(tmp_path / "items.pt"), str(tmp_path / "report.json"), tok,
                 design="choice8")
    assert report["design"] == "choice8" and report["kept_conversations"] == 4 and report["items"] == 8


@pytest.mark.integration
def test_run_tags_sources_and_skips_severity_for_generated_rows(tmp_path, tok):
    horizon_like = json.loads(_row("Child: hi\nOther: hey", {"risk_level": "none", "categories": []}))
    generated = json.loads(_row("Child: hi\nOther: how old are you?", {"risk_level": "high", "categories": ["grooming"]}))
    generated["meta"] = {"id": "xg-grooming-000", "category": "grooming"}
    path = tmp_path / "mixed.jsonl"
    path.write_text(json.dumps(horizon_like) + "\n" + json.dumps(generated) + "\n")
    run(str(path), str(tmp_path / "o.pt"), str(tmp_path / "r.json"), tok, source_tag="mi")
    items = torch.load(tmp_path / "o.pt", weights_only=True)
    per_conv = Counter(it["conv"] for it in items)
    assert set(per_conv) == {"mi:0", "mi:1"}
    assert per_conv["mi:0"] == 8 and per_conv["mi:1"] == 7  # generated rows carry no severity item
