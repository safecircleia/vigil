import pytest
from laya.common import QTYPES

from vigil.labels import Labels
from vigil.rows import HEAD_MAX_LEN, MAX_LEN, RowError, build_items, split_messages

MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "Analyze this conversation:\nChild: hi\nOther: hello"},
    {"role": "assistant", "content": '{"risk_level": "none", "categories": []}'},
]


def test_split_messages_strips_prefix():
    state, raw = split_messages(MESSAGES)
    assert state == "Child: hi\nOther: hello"
    assert raw == '{"risk_level": "none", "categories": []}'


def test_split_messages_without_prefix_keeps_text():
    msgs = [{"role": "user", "content": "Child: hi"}, {"role": "assistant", "content": "{}"}]
    assert split_messages(msgs)[0] == "Child: hi"


@pytest.mark.parametrize("msgs", [
    [],
    [{"role": "assistant", "content": "{}"}],
    [{"role": "user", "content": "x"}],
    [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "{}"}],
])
def test_split_messages_rejects_bad_shapes(msgs):
    with pytest.raises(RowError):
        split_messages(msgs)


@pytest.mark.integration
def test_build_items_makes_eight_sequences(tok):
    labels = Labels(frozenset({"grooming"}), 2)
    items, overflowed = build_items(tok, "Child: hi\nOther: how old are you?", labels)
    assert not overflowed and len(items) == 8
    assert [it["qtype"] for it in items] == [QTYPES["noul"]] * 7 + [QTYPES["score"]]
    for it in items:
        assert abs(sum(it["target"]) - 1.0) < 1e-9
        assert len(it["ids"]) <= MAX_LEN
        assert len(it["markers"]) == len(it["target"])
    assert items[0]["label"] == 1  # grooming is true
    assert items[1]["label"] == 0  # bullying is false
    assert items[7]["label"] == 2  # severity level


@pytest.mark.integration
def test_overflow_is_dropped_and_reported(tok):
    long_state = "Child: hello there\nOther: hi\n" * 400
    items, overflowed = build_items(tok, long_state, Labels(frozenset(), 0))
    assert items is None and overflowed


@pytest.mark.integration
def test_overflow_keep_recent_fits_the_window(tok):
    long_state = "Child: hello there\nOther: hi\n" * 400
    items, overflowed = build_items(tok, long_state, Labels(frozenset(), 0), overflow="keep_recent")
    assert overflowed and len(items) == 8
    assert all(len(it["ids"]) <= MAX_LEN for it in items)


def test_unknown_overflow_policy_rejected(tok):
    with pytest.raises(ValueError):
        build_items(tok, "x", Labels(frozenset(), 0), overflow="truncate")


def test_head_budget_constant():
    assert HEAD_MAX_LEN < MAX_LEN


@pytest.mark.integration
def test_choice_design_makes_two_sequences_without_clipping_options(tok):
    from laya.common import build_sequence

    from vigil.questions import build_questions, to_sequence_question

    items, overflowed = build_items(tok, "Child: hi\nOther: how old are you?",
                                    Labels(frozenset({"grooming"}), 2), design="choice8")
    assert not overflowed and len(items) == 2
    assert [it["qtype"] for it in items] == [QTYPES["choice"], QTYPES["score"]]
    assert len(items[0]["markers"]) == 8 and items[0]["label"] == 1
    sq = to_sequence_question(build_questions("choice8")["category"])
    stats = build_sequence(tok, "x", sq, MAX_LEN, HEAD_MAX_LEN, return_stats=True)[2]
    assert stats["tokens_per_option"] is None  # the 8 option texts fit the head budget unclipped


@pytest.mark.integration
def test_build_items_can_leave_the_severity_question_out(tok):
    labels = Labels(frozenset({"grooming"}), 2)
    items, _ = build_items(tok, "Child: hi\nOther: hey", labels, include_severity=False)
    assert len(items) == 7 and all(it["qtype"] == QTYPES["noul"] for it in items)
    items, _ = build_items(tok, "Child: hi\nOther: hey", labels, design="choice8", include_severity=False)
    assert len(items) == 1 and items[0]["qtype"] == QTYPES["choice"]
