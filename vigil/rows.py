from __future__ import annotations

from laya.common import build_sequence, render_options

from vigil.labels import Labels
from vigil.questions import build_questions, question_ids, target_for, to_sequence_question

BASE_MODEL_ID = "convaiinnovations/laya-multilingual"
MAX_LEN = 1024
HEAD_MAX_LEN = 256
_PREFIX = "Analyze this conversation:\n"
_UNBOUNDED = 10**6


class RowError(ValueError):
    pass


def split_messages(messages: list) -> tuple:
    users = [m for m in messages if m.get("role") == "user"]
    assistants = [m for m in messages if m.get("role") == "assistant"]
    if len(users) != 1:
        raise RowError(f"expected exactly one user message, found {len(users)}")
    if not assistants:
        raise RowError("no assistant message (target) found")
    state = users[0]["content"]
    if state.startswith(_PREFIX):
        state = state[len(_PREFIX):]
    return state, assistants[-1]["content"]


def state_overflows(tok, state: str, design: str = "noul7") -> bool:
    """True if the conversation does not fit the window for at least one question."""
    questions = build_questions(design)
    return any(
        len(build_sequence(tok, state, to_sequence_question(questions[qid]), _UNBOUNDED, HEAD_MAX_LEN)[0])
        > MAX_LEN
        for qid in question_ids(design)
    )


def build_items(tok, state: str, labels: Labels, *, overflow: str = "drop", design: str = "noul7",
                include_severity: bool = True):
    if overflow not in ("drop", "keep_recent"):
        raise ValueError(f"overflow must be 'drop' or 'keep_recent', got {overflow!r}")
    questions = build_questions(design)
    ids_for_design = tuple(q for q in question_ids(design) if include_severity or q != "severity")
    seq_questions = {qid: to_sequence_question(questions[qid]) for qid in ids_for_design}

    overflowed = state_overflows(tok, state, design)
    if overflowed and overflow == "drop":
        return None, True

    items = []
    for qid in ids_for_design:
        sq = seq_questions[qid]
        ids, markers = build_sequence(
            tok, state, sq, MAX_LEN, HEAD_MAX_LEN, truncate_left=overflowed
        )
        if len(markers) != len(render_options(sq)):
            raise RowError(f"question {qid!r}: option markers lost to the head budget")
        qtype, target, label = target_for(qid, labels, design)
        items.append({"ids": ids, "markers": markers, "qtype": qtype, "target": target, "label": label})
    return items, overflowed
