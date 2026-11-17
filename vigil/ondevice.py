"""Reference implementation of Vigil's on-device inference: everything around the .tflite graph.

The two questions of the `choice8` design are fixed text, so their token prefixes and marker
positions are precomputed into a spec (`vigil_spec.json`). At run time a conversation only needs
tokenizing, appending after each prefix and padding to `seq_len`; the graph returns one logit per
marker and the softmax, temperature and thresholds happen here. The Kotlin port in the Android app
follows this file, and its golden test vectors are generated from it.
"""
from __future__ import annotations

import numpy as np
from laya.common import QTYPES, build_sequence, clamp_temperature, temp_bucket

from vigil.labels import CATEGORIES, SEVERITY_LEVELS
from vigil.questions import build_questions, to_sequence_question

MAX_MARKERS = 8
QUESTIONS = ("category", "severity")


class OnDevice:
    def __init__(self, tok, spec: dict):
        self.tok, self.spec = tok, spec

    @classmethod
    def from_agent(cls, agent, seq_len: int, thresholds: dict) -> "OnDevice":
        tok = agent.tok
        questions = build_questions("choice8")
        spec = {
            "seq_len": seq_len,
            "sep_id": tok.sep_token_id,
            "mask_token": tok.mask_token,
            "categories": list(CATEGORIES),
            "severity_levels": list(SEVERITY_LEVELS),
            "thresholds": {c: float(thresholds[c]) for c in CATEGORIES},
            "questions": {},
        }
        for qid in QUESTIONS:
            q = to_sequence_question(questions[qid])
            ids, markers = build_sequence(tok, "", q, seq_len, agent.cfg["head_max_len"])
            qtype = QTYPES[q["t"]]
            t = agent.cfg.get("temperature_by_options", {}).get(temp_bucket(qtype, len(markers)),
                                                                agent.cfg["temperature"][qtype])
            spec["questions"][qid] = {
                "prefix_ids": ids[:-1],  # build_sequence ends an empty state with one SEP; the app appends its own
                "marker_pos": markers,
                "qtype": qtype,
                "temperature": clamp_temperature(t),
            }
        return cls(tok, spec)

    def state_ids(self, state: str) -> list:
        return self.tok(state.replace(self.spec["mask_token"], " "), add_special_tokens=False)["input_ids"]

    def fits(self, state: str) -> bool:
        n = len(self.state_ids(state))
        return all(len(q["prefix_ids"]) + n + 1 <= self.spec["seq_len"] for q in self.spec["questions"].values())

    def encode(self, state: str) -> tuple:
        """(input_ids, attention_mask, marker_pos, marker_mask, qtype), int32, batch of the two questions.

        A conversation longer than the window keeps its most recent tokens, as training did for
        rows that overflowed.
        """
        s = self.spec["seq_len"]
        sid = self.state_ids(state)
        ids = np.zeros((2, s), np.int32)
        mask = np.zeros((2, s), np.int32)
        pos = np.zeros((2, MAX_MARKERS), np.int32)
        mmask = np.zeros((2, MAX_MARKERS), np.int32)
        qtype = np.zeros(2, np.int32)
        for r, qid in enumerate(QUESTIONS):
            q = self.spec["questions"][qid]
            prefix = q["prefix_ids"]
            room = max(0, s - len(prefix) - 1)
            row = prefix + sid[max(0, len(sid) - room):] + [self.spec["sep_id"]]
            ids[r, : len(row)] = row
            mask[r, : len(row)] = 1
            pos[r, : len(q["marker_pos"])] = q["marker_pos"]
            mmask[r, : len(q["marker_pos"])] = 1
            qtype[r] = q["qtype"]
        return ids, mask, pos, mmask, qtype


def _softmax(logits, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


def decode_category(logits: np.ndarray, spec: dict) -> tuple:
    """Graph output [2, 8] -> (probability per risk category in CATEGORIES order, expected severity 0-4)."""
    cat, sev = spec["questions"]["category"], spec["questions"]["severity"]
    p_cat = _softmax(logits[0][: len(cat["marker_pos"])], cat["temperature"])  # index 0 is "benign"
    p_sev = _softmax(logits[1][: len(sev["marker_pos"])], sev["temperature"])
    return p_cat[1:], float((np.arange(len(p_sev)) * p_sev).sum())
