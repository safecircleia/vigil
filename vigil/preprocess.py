from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import torch

from vigil.labels import Labels, LabelError, is_refusal, parse_target
from vigil.questions import DESIGNS
from vigil.checkpoint import require_pinned_revision
from vigil.rows import BASE_MODEL_ID, RowError, build_items, split_messages


def stratum(labels: Labels) -> str:
    return "+".join(sorted(labels.categories)) if labels.categories else "benign"


def stratified_indices(strata: list, n: int, seed: int) -> list:
    if n <= 0 or n >= len(strata):
        return list(range(len(strata)))
    groups: dict = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    rng = random.Random(seed)
    for members in groups.values():
        rng.shuffle(members)
    quota = {s: 0 for s in groups}
    remaining = n
    open_groups = sorted(groups)
    while remaining > 0 and open_groups:
        share = max(1, remaining // len(open_groups))
        still_open = []
        for s in open_groups:
            take = min(share, len(groups[s]) - quota[s], remaining)
            quota[s] += take
            remaining -= take
            if quota[s] < len(groups[s]):
                still_open.append(s)
            if remaining == 0:
                break
        open_groups = still_open
    return sorted(i for s, members in groups.items() for i in members[: quota[s]])


def _read_rows(input_path: str, max_malformed: int):
    parsed, malformed, refusals = [], 0, 0
    with open(input_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                state, raw = split_messages(json.loads(line)["messages"])
                if is_refusal(raw):
                    refusals += 1
                    continue
                labels = parse_target(raw)
                generated = "meta" in json.loads(line)  # generated rows: severity is by construction, not trusted
            except (KeyError, json.JSONDecodeError, RowError, LabelError) as exc:
                malformed += 1
                if malformed > max_malformed:
                    raise ValueError(f"{input_path} line {line_no}: {exc}") from exc
                continue
            parsed.append((state, labels, generated))
    return parsed, malformed, refusals


def run(input_path, output_path, report_path, tok, *, train_conversations=0,
        seed=20260922, overflow="drop", max_malformed=0, design="noul7", source_tag="h") -> dict:
    parsed, malformed, refusals = _read_rows(input_path, max_malformed)
    picked = stratified_indices([stratum(lab) for _, lab, _ in parsed], train_conversations, seed)

    items, kept, overflowed, dropped = [], 0, 0, 0
    per_stratum: Counter = Counter()
    for i in picked:
        state, labels, generated = parsed[i]
        conv_items, was_over = build_items(tok, state, labels, overflow=overflow, design=design,
                                           include_severity=not generated)
        overflowed += was_over
        if conv_items is None:
            dropped += 1
            continue
        for it in conv_items:
            it["conv"] = f"{source_tag}:{i}"  # lets training keep a conversation's 8 items on one side of the calibration split
        items.extend(conv_items)
        kept += 1
        per_stratum[stratum(labels)] += 1

    report = {
        "rows": len(parsed) + malformed + refusals,
        "malformed": malformed,
        "refusals_skipped": refusals,
        "inconsistent": sum(lab.inconsistent for _, lab, _ in parsed),
        "overflowed": overflowed,
        "dropped_overflow": dropped,
        "kept_conversations": kept,
        "items": len(items),
        "per_stratum": dict(per_stratum),
        "train_conversations_requested": train_conversations,
        "design": design,
        "source_tag": source_tag,
        "overflow_policy": overflow,
        "seed": seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(items, output_path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description="Build typed training items from a messages-format jsonl")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--model-id", default=BASE_MODEL_ID)
    p.add_argument("--base-revision", default=None,
                   help="commit sha (or tag) of --model-id; required, 'main' is refused")
    p.add_argument("--train-conversations", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--overflow", choices=["drop", "keep_recent"], default="drop")
    p.add_argument("--max-malformed", type=int, default=0)
    p.add_argument("--design", choices=DESIGNS, default="noul7")
    p.add_argument("--source-tag", default="h", help="prefix of the conversation ids; must differ per source")
    a = p.parse_args()

    base_revision = require_pinned_revision(a.base_revision, f"base model {a.model_id!r}")
    model_dir = snapshot_download(a.model_id, revision=base_revision)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    report = run(a.input, a.output, a.report, tok, train_conversations=a.train_conversations,
                 seed=a.seed, overflow=a.overflow, max_malformed=a.max_malformed, design=a.design,
                 source_tag=a.source_tag)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
