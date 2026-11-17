import json

import pytest

from vigil.labels import CATEGORIES, SEVERITY_LEVELS, LabelError, is_refusal, parse_target


def test_parses_source_target_keys():
    raw = json.dumps({
        "risk_level": "high", "categories": ["grooming", "isolation"],
        "confidence": 0.9, "reasoning": "x", "matched_terms": [],
    })
    labels = parse_target(raw)
    assert labels.categories == frozenset({"grooming", "isolation"})
    assert labels.severity == SEVERITY_LEVELS.index("high")


def test_parses_prompt_schema_keys():
    labels = parse_target({"category": "threats", "severity": "low"})
    assert labels.categories == frozenset({"threats"})
    assert labels.severity == 1


def test_benign_means_no_categories():
    assert parse_target({"category": "benign", "severity": "none"}).categories == frozenset()
    assert parse_target({"categories": [], "risk_level": "none"}).categories == frozenset()


@pytest.mark.parametrize("raw", [
    "not json",
    "{}",
    "[]",
    '{"risk_level": "extreme", "categories": []}',
    '{"risk_level": "low", "categories": ["hugs"]}',
    '{"risk_level": "low"}',
    '{"categories": ["threats"]}',
    '{"risk_level": "low", "categories": [1]}',
])
def test_malformed_targets_raise(raw):
    with pytest.raises(LabelError):
        parse_target(raw)


def test_inconsistent_flag():
    assert parse_target({"risk_level": "none", "categories": ["threats"]}).inconsistent
    assert parse_target({"risk_level": "low", "categories": []}).inconsistent
    assert not parse_target({"risk_level": "low", "categories": ["threats"]}).inconsistent
    assert not parse_target({"risk_level": "none", "categories": []}).inconsistent


def test_seven_categories():
    assert len(CATEGORIES) == 7 and len(set(CATEGORIES)) == 7


def test_refusal_targets_are_recognised():
    assert is_refusal('{"error": "I only analyze conversations for child safety risks."}')
    assert is_refusal({"error": "x"})
    assert not is_refusal('{"risk_level": "none", "categories": []}')
    assert not is_refusal('{"error": "x", "risk_level": "none"}')
    assert not is_refusal("not json")
