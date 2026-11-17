from __future__ import annotations

from laya.common import QTYPES

from vigil.labels import CATEGORIES, SEVERITY_LEVELS, Labels

QUESTION_IDS = CATEGORIES + ("severity",)
DESIGNS = ("noul7", "choice8")  # 7 yes/no questions + severity, or one 8-way choice + severity

_CATEGORY_TEXT = {
    "grooming": "an older person builds trust with the child to exploit them (flattery, secrecy, gifts, testing boundaries)",
    "bullying": "someone insults, humiliates, excludes or harasses the child",
    "sexual_content": "the conversation contains sexual content or requests directed at the child",
    "isolation": "someone tries to cut the child off from family and friends, or the child shows signs of isolation",
    "personal_info": "someone asks for or the child shares personal information (address, school, phone, photos)",
    "platform_migration": "someone tries to move the conversation to another, less monitored platform",
    "threats": "someone threatens or blackmails the child, or the child expresses self-harm",
}

_SEVERITY_TEXT = (
    "no risk: an ordinary, harmless conversation",
    "low risk: a mild warning sign that needs no action yet",
    "medium risk: a clear warning sign that a guardian should review",
    "high risk: a serious risk that needs prompt action",
    "critical risk: immediate danger to the child",
)


_CHOICE_TEXT = {
    "benign": "no risk, an ordinary harmless chat",
    "grooming": "an older person builds trust to exploit the child",
    "bullying": "insults, humiliation or harassment",
    "sexual_content": "sexual content directed at the child",
    "isolation": "cutting the child off from family and friends",
    "personal_info": "asking for or sharing personal information",
    "platform_migration": "moving the chat to a less monitored app",
    "threats": "threats, blackmail or self-harm",
}


def _check_design(design: str) -> None:
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")


def question_ids(design: str = "noul7") -> tuple:
    _check_design(design)
    return QUESTION_IDS if design == "noul7" else ("category", "severity")


def _severity_question() -> dict:
    return {
        "type": "score",
        "instructions": "How severe is the child-safety risk in this conversation?",
        "criteria": list(_SEVERITY_TEXT),
    }


def build_questions(design: str = "noul7") -> dict:
    _check_design(design)
    if design == "choice8":
        return {
            "category": {
                "type": "choice",
                "instructions": "Which kind of child-safety risk does this conversation show?",
                "criteria": {name: _CHOICE_TEXT[name] for name in ("benign",) + CATEGORIES},
            },
            "severity": _severity_question(),
        }
    questions = {}
    for cat in CATEGORIES:
        text = _CATEGORY_TEXT[cat]
        questions[cat] = {
            "type": "noul",
            "instructions": f"Does this conversation show {cat.replace('_', ' ')}?",
            "criteria": {
                "true": f"yes: {text}",
                "false": f"no: there is no sign that {text}",
            },
        }
    questions["severity"] = _severity_question()
    return questions


def to_sequence_question(q: dict) -> dict:
    return {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}


def target_for(qid: str, labels: Labels, design: str = "noul7"):
    _check_design(design)
    if qid == "severity":
        target = [0.0] * len(SEVERITY_LEVELS)
        target[labels.severity] = 1.0
        return QTYPES["score"], target, labels.severity
    if design == "choice8":
        if qid != "category":
            raise KeyError(qid)
        names = ("benign",) + CATEGORIES
        gold = sorted(labels.categories) or ["benign"]
        target = [0.0] * len(names)
        for name in gold:
            target[names.index(name)] = 1.0 / len(gold)  # several gold categories share the mass
        return QTYPES["choice"], target, target.index(max(target))
    if qid not in CATEGORIES:
        raise KeyError(qid)
    is_true = qid in labels.categories
    return QTYPES["noul"], ([0.0, 1.0] if is_true else [1.0, 0.0]), int(is_true)
