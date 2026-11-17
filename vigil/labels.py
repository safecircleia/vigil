from __future__ import annotations

import json
from dataclasses import dataclass

CATEGORIES = (
    "grooming",
    "bullying",
    "sexual_content",
    "isolation",
    "personal_info",
    "platform_migration",
    "threats",
)
SEVERITY_LEVELS = ("none", "low", "medium", "high", "critical")


class LabelError(ValueError):
    pass


@dataclass(frozen=True)
class Labels:
    categories: frozenset
    severity: int

    @property
    def inconsistent(self) -> bool:
        # risk without a category, or a category without risk
        return (self.severity == 0) != (not self.categories)


def _categories(obj: dict) -> frozenset:
    if "categories" in obj:
        raw = obj["categories"]
    elif "category" in obj:
        raw = obj["category"]
    else:
        raise LabelError("target has neither 'categories' nor 'category'")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise LabelError(f"categories must be a list or string, got {type(raw).__name__}")
    names = [c for c in raw if c != "benign"]
    unknown = [c for c in names if c not in CATEGORIES]
    if unknown:
        raise LabelError(f"unknown categories: {unknown}")
    return frozenset(names)


def _severity(obj: dict) -> int:
    key = "risk_level" if "risk_level" in obj else "severity" if "severity" in obj else None
    if key is None:
        raise LabelError("target has neither 'risk_level' nor 'severity'")
    level = obj[key]
    if level not in SEVERITY_LEVELS:
        raise LabelError(f"unknown severity level: {level!r}")
    return SEVERITY_LEVELS.index(level)


def is_refusal(raw) -> bool:
    """True for Horizon's identity/off-topic refusal targets: a JSON object whose only key is 'error'."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return False
    return isinstance(raw, dict) and set(raw) == {"error"}


def parse_target(raw) -> Labels:
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LabelError(f"target is not JSON: {exc}") from exc
    else:
        obj = raw
    if not isinstance(obj, dict):
        raise LabelError("target must be a JSON object")
    return Labels(_categories(obj), _severity(obj))
