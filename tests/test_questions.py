import pytest
from laya.common import QTYPES, render_options

from vigil.labels import CATEGORIES, SEVERITY_LEVELS, Labels
from vigil.questions import QUESTION_IDS, build_questions, question_ids, target_for, to_sequence_question


def test_eight_questions_in_order():
    assert QUESTION_IDS == CATEGORIES + ("severity",)
    assert list(build_questions()) == list(QUESTION_IDS)


def test_option_counts_match_laya_rendering():
    for qid, q in build_questions().items():
        n = len(render_options(to_sequence_question(q)))
        assert n == (5 if qid == "severity" else 2), qid


def test_noul_criteria_are_keyed_true_false():
    for qid in CATEGORIES:
        q = build_questions()[qid]
        assert q["type"] == "noul"
        assert set(q["criteria"]) == {"true", "false"}


def test_score_has_one_level_per_severity():
    q = build_questions()["severity"]
    assert q["type"] == "score" and len(q["criteria"]) == len(SEVERITY_LEVELS)


def test_target_for_noul():
    labels = Labels(frozenset({"grooming"}), 2)
    assert target_for("grooming", labels) == (QTYPES["noul"], [0.0, 1.0], 1)
    assert target_for("threats", labels) == (QTYPES["noul"], [1.0, 0.0], 0)


def test_target_for_score():
    labels = Labels(frozenset({"grooming"}), 3)
    qtype, target, label = target_for("severity", labels)
    assert qtype == QTYPES["score"] and label == 3
    assert target == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_target_for_unknown_question():
    with pytest.raises(KeyError):
        target_for("nope", Labels(frozenset(), 0))


def test_choice_design_has_one_category_question_plus_severity():
    assert question_ids("choice8") == ("category", "severity")
    qs = build_questions("choice8")
    assert list(qs) == ["category", "severity"]
    assert qs["category"]["type"] == "choice"
    assert list(qs["category"]["criteria"]) == ["benign"] + list(CATEGORIES)
    assert len(render_options(to_sequence_question(qs["category"]))) == 8


def test_choice_target_single_multi_and_benign():
    qtype, target, label = target_for("category", Labels(frozenset({"grooming"}), 2), "choice8")
    assert qtype == QTYPES["choice"] and label == 1 and target[1] == 1.0 and sum(target) == 1.0
    _, target, label = target_for("category", Labels(frozenset({"grooming", "threats"}), 3), "choice8")
    assert target[1] == 0.5 and target[7] == 0.5 and sum(target) == 1.0 and label == 1
    _, target, label = target_for("category", Labels(frozenset(), 0), "choice8")
    assert target[0] == 1.0 and label == 0


def test_choice_design_severity_target_is_unchanged():
    assert target_for("severity", Labels(frozenset({"grooming"}), 3), "choice8") == \
        target_for("severity", Labels(frozenset({"grooming"}), 3))


def test_unknown_design_is_rejected():
    with pytest.raises(ValueError):
        build_questions("nope")
    with pytest.raises(ValueError):
        question_ids("nope")
    with pytest.raises(KeyError):
        target_for("grooming", Labels(frozenset(), 0), "choice8")
