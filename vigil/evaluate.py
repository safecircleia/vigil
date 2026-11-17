from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from vigil.checkpoint import require_pinned_revision, validate_checkpoint_dir
from vigil.labels import CATEGORIES, SEVERITY_LEVELS, parse_target
from vigil.metrics import (auc, benign_false_alarms, catch_rate, category_catch_rate, per_category_recall,
                           pick_threshold, pick_thresholds_greedy, pick_thresholds_with_fa_budget, precision, severity_macro_f1,
                           split_val_test, wilson_interval)
from vigil.questions import DESIGNS, build_questions
from vigil.rows import split_messages, state_overflows

HORIZON_REFERENCE = {
    "grooming_recall": 0.9036,
    "catch_rate": 0.1290,
    "catch_rate_note": "Horizon's regex rule pre-filter, not a model",
    "latency_s": 2.1,
    "disk_mb": 1017,
}


def _category_probs(answers: dict, design: str) -> dict:
    if design == "choice8":
        probs = answers["category"]["probabilities"]
        return {c: float(probs.get(c, 0.0)) for c in CATEGORIES}
    return {c: float(answers[c]["noul"]) for c in CATEGORIES}


def sample_indices(n_lines: int, max_rows: int, seed: int) -> list:
    return sorted(random.Random(seed).sample(range(n_lines), min(max_rows, n_lines)))


def load_rows(jsonl_path: str, agent, max_rows: int, seed: int, tok, design: str = "noul7") -> tuple:
    """Run the agent over a random sample; returns (rows, stats).

    Conversations that do not fit the window are excluded and counted: the Agent would otherwise
    silently keep only the first tokens of a string state and score the rest as if it were absent.
    """
    with open(jsonl_path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    picked = sample_indices(len(lines), max_rows, seed)
    questions = build_questions(design)
    rows, excluded = [], 0
    for i in picked:
        state, raw = split_messages(json.loads(lines[i])["messages"])
        if state_overflows(tok, state, design):
            excluded += 1
            continue
        labels = parse_target(raw)
        t0 = time.perf_counter()
        answers = agent.predict(state, questions)["answers"]
        latency_ms = (time.perf_counter() - t0) * 1000
        sev = answers["severity"]["probabilities"]
        rows.append({
            "gold_categories": labels.categories,
            "gold_severity": labels.severity,
            "probs": _category_probs(answers, design),
            "severity_probs": [float(sev.get(str(k), 0.0)) for k in range(len(SEVERITY_LEVELS))],
            "latency_ms": latency_ms,
        })
    return rows, {"sampled": len(picked), "overflow_excluded": excluded}


def summarize(rows: list, *, min_precision: float, seed: int, fixed_thresholds: dict | None = None,
              max_benign_fa: float | None = None, threshold_rule: str = "precision") -> dict:
    policy = None
    if fixed_thresholds is None:
        val_idx, test_idx = split_val_test(len(rows), seed)
        val = [rows[i] for i in val_idx]
        test = [rows[i] for i in test_idx]
        thresholds, met = {}, {}
        if max_benign_fa is not None and threshold_rule == "greedy":
            found = pick_thresholds_greedy(val, max_benign_fa)
            thresholds = found["thresholds"]
            met = {cat: found["met"] for cat in CATEGORIES}
            policy = {"rule": "greedy", "max_benign_fa": max_benign_fa, "precision_target": None,
                      "met": found["met"], "benign_false_alarms_val": list(found["benign_false_alarms"])}
        elif max_benign_fa is not None:
            found = pick_thresholds_with_fa_budget(val, max_benign_fa)
            thresholds = found["thresholds"]
            met = {cat: found["met"] for cat in CATEGORIES}
            policy = {"rule": "precision", "max_benign_fa": max_benign_fa,
                      "precision_target": found["precision_target"],
                      "met": found["met"], "benign_false_alarms_val": list(found["benign_false_alarms"])}
        else:
            for cat in CATEGORIES:
                scores = [r["probs"][cat] for r in val]
                golds = [cat in r["gold_categories"] for r in val]
                thresholds[cat], met[cat] = pick_threshold(scores, golds, min_precision)
    else:  # transfer test: use the thresholds chosen elsewhere, score every row, tune nothing
        val, test = [], list(rows)
        thresholds = {cat: float(fixed_thresholds[cat]) for cat in CATEGORIES}
        met = {cat: None for cat in CATEGORIES}

    recall = per_category_recall(test, thresholds)
    recall_half = per_category_recall(test, {})  # every threshold defaults to 0.5
    per_category = {
        cat: {"recall": recall[cat], "ci95": wilson_interval(*recall[cat]),
              "precision": precision(test, thresholds, cat),
              "recall_at_half": recall_half[cat],
              "auc": auc([r["probs"][cat] for r in test], [cat in r["gold_categories"] for r in test]),
              "threshold": thresholds[cat], "precision_target_met": met[cat]}
        for cat in CATEGORIES
    }
    caught, risky = catch_rate(test, thresholds)
    fa_hits, fa_support = benign_false_alarms(test, thresholds)
    cc_hits, cc_support = category_catch_rate(test, thresholds)
    gold_sev = [r["gold_severity"] for r in test]
    pred_sev = [int(np.argmax(r["severity_probs"])) for r in test]
    latencies = [r["latency_ms"] for r in rows]
    return {
        "n_val": len(val), "n_test": len(test), "thresholds": thresholds,
        "thresholds_fixed": fixed_thresholds is not None,
        "threshold_policy": policy,
        "min_precision": min_precision, "per_category": per_category,
        "catch_rate": {"value": caught / risky if risky else 0.0, "hits": caught, "support": risky,
                       "ci95": wilson_interval(caught, risky)},
        "catch_rate_category": {"hits": cc_hits, "support": cc_support,
                                "ci95": wilson_interval(cc_hits, cc_support)},
        "benign_false_alarms": {"hits": fa_hits, "support": fa_support,
                                "ci95": wilson_interval(fa_hits, fa_support)},
        "risky_share_test": risky / len(test) if test else 0.0,
        "severity_macro_f1": severity_macro_f1(gold_sev, pred_sev),
        "latency_ms": {"p50": float(np.percentile(latencies, 50)),
                       "p95": float(np.percentile(latencies, 95))},
    }


def render_markdown(summary: dict, *, model: str, device: str) -> str:
    ref = HORIZON_REFERENCE
    lines = [
        f"# Vigil evaluation: {model} ({device})", "",
        (f"Thresholds are FIXED (taken from an earlier in-distribution evaluation, nothing tuned here) "
         f"and applied to all {summary['n_test']} rows."
         if summary.get("thresholds_fixed") else
         (f"Thresholds are picked per-category on the validation half ({summary['n_val']} rows) to keep as much "
          f"recall as possible within a benign false-alarm budget of {summary['threshold_policy']['max_benign_fa']:.1%} "
          f"(budget met: {summary['threshold_policy']['met']}; validation false alarms "
          f"{summary['threshold_policy']['benign_false_alarms_val'][0]}/{summary['threshold_policy']['benign_false_alarms_val'][1]}). "
          f"Scored on the disjoint test half ({summary['n_test']} rows).")
         if summary.get("threshold_policy") and summary["threshold_policy"].get("rule") == "greedy" else
         (f"Thresholds are picked on the validation half ({summary['n_val']} rows) by a benign false-alarm "
          f"budget of {summary['threshold_policy']['max_benign_fa']:.1%}: the smallest precision target that fits "
          f"is {summary['threshold_policy']['precision_target']} (budget met: {summary['threshold_policy']['met']}; "
          f"validation false alarms {summary['threshold_policy']['benign_false_alarms_val'][0]}/"
          f"{summary['threshold_policy']['benign_false_alarms_val'][1]}). Scored on the disjoint test half "
          f"({summary['n_test']} rows).")
         if summary.get("threshold_policy") else
         f"Thresholds are picked on the validation half ({summary['n_val']} rows) for "
         f"precision >= {summary['min_precision']} and scored on the disjoint test half "
         f"({summary['n_test']} rows). The calibration slice used in training is class-balanced "
         "(stratified) and must never be used to choose thresholds."), "",
        "| Category | Recall | 95% CI | Support | Precision | AUC | Recall @0.5 | Threshold | "
        "Precision target met (on validation; False = threshold fell back to 0.5; fixed = not tuned here) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cat, m in summary["per_category"].items():
        hits, support = m["recall"]
        lo, hi = m["ci95"]
        rate = f"{hits / support:.2%}" if support else "n/a"
        tp, n_flagged = m["precision"]
        prec = f"{tp / n_flagged:.1%} ({tp}/{n_flagged})" if n_flagged else "n/a"
        h_hits, h_support = m["recall_at_half"]
        half = f"{h_hits / h_support:.1%}" if h_support else "n/a"
        auc_txt = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
        lines.append(f"| {cat} | {rate} | {lo:.2%}-{hi:.2%} | {support} | {prec} | {auc_txt} | {half} | "
                     f"{m['threshold']:.3f} | "
                     f"{'fixed' if m['precision_target_met'] is None else m['precision_target_met']} |")
    cr = summary["catch_rate"]
    lo, hi = cr["ci95"]
    fa = summary["benign_false_alarms"]
    fa_lo, fa_hi = fa["ci95"]
    fa_rate = f"{fa['hits'] / fa['support']:.1%}" if fa["support"] else "n/a"
    sampling = summary.get("sampling")
    if sampling:
        lines[2:2] = [
            f"Excluded {sampling['overflow_excluded']} of {sampling['sampled']} sampled rows: the "
            "conversation is longer than the model window, so it was not scored.", "",
        ]
    cc = summary["catch_rate_category"]
    cc_lo, cc_hi = cc["ci95"]
    cc_rate = f"{cc['hits'] / cc['support']:.2%}" if cc["support"] else "n/a"
    lines += [
        "", f"Benign false alarms (any category flagged): {fa_rate} ({fa['hits']}/{fa['support']}, "
        f"95% CI {fa_lo:.1%}-{fa_hi:.1%})",
        f"Share of risky conversations in the test half: {summary['risky_share_test']:.0%}. Precision and "
        "false-alarm rates depend on prevalence: real traffic is mostly benign, so expect more false "
        "alarms per flag than shown here.", "",
        f"Catch rate (any category flagged, even the wrong one): {cr['value']:.2%} "
        f"({cr['hits']}/{cr['support']}, 95% CI {lo:.2%}-{hi:.2%})",
        f"Catch rate, correct category flagged: {cc_rate} ({cc['hits']}/{cc['support']}, "
        f"95% CI {cc_lo:.2%}-{cc_hi:.2%})",
        f"Severity macro-F1: {summary['severity_macro_f1']:.3f}",
        f"Latency per conversation (8 questions): p50 {summary['latency_ms']['p50']:.1f} ms, "
        f"p95 {summary['latency_ms']['p95']:.1f} ms on {device}", "",
        "## Horizon reference (recorded, static)", "",
        f"- Grooming recall: {ref['grooming_recall']:.2%}",
        f"- Catch rate: {ref['catch_rate']:.2%} ({ref['catch_rate_note']}), so not directly comparable",
        f"- Latency: {ref['latency_s']} s, disk: {ref['disk_mb']} MB",
    ]
    return "\n".join(lines) + "\n"


def _load_tokenizer(model_dir: str, revision: str | None = None):
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    path = model_dir if os.path.isdir(model_dir) else snapshot_download(model_dir, revision=revision)
    _fix_tokenizer_config(path)
    return AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))


def _checkpoint_revision(model_dir: str, revision: str | None) -> str | None:
    """Validate a local checkpoint, or require a pinned revision for a Hub id."""
    if os.path.isdir(model_dir):
        validate_checkpoint_dir(model_dir)
        return None
    return require_pinned_revision(revision, f"Hub model {model_dir!r}")


def evaluate(model_dir, eval_jsonl, out_dir, *, max_rows=5000, min_precision=0.8,
             seed=20260922, device="cpu", design="noul7", thresholds_from=None,
             max_benign_fa=None, threshold_rule="precision", revision=None) -> dict:
    import laya

    revision = _checkpoint_revision(model_dir, revision)
    agent = laya.Agent(model_dir, device=device, revision=revision)
    rows, stats = load_rows(eval_jsonl, agent, max_rows, seed, _load_tokenizer(model_dir, revision), design)
    fixed = None
    if thresholds_from:
        with open(thresholds_from) as f:
            fixed = json.load(f)["thresholds"]
    summary = summarize(rows, min_precision=min_precision, seed=seed, fixed_thresholds=fixed,
                        max_benign_fa=max_benign_fa, threshold_rule=threshold_rule)
    summary["sampling"] = stats
    summary["design"] = design
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(render_markdown(summary, model=f"{model_dir} [{design}]", device=device))
    with open(os.path.join(out_dir, "rows.json"), "w") as f:
        json.dump([{**r, "gold_categories": sorted(r["gold_categories"])} for r in rows], f)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a Vigil checkpoint")
    p.add_argument("--model-dir", required=True,
                   help="local checkpoint dir, or a Hub id with --revision (use the base id for the zero-shot baseline)")
    p.add_argument("--revision", default=None,
                   help="commit sha (or tag) for a Hub --model-dir; 'main' is refused")
    p.add_argument("--eval", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--min-precision", type=float, default=0.8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--design", choices=DESIGNS, default="noul7")
    p.add_argument("--thresholds-from", default=None,
                   help="report.json whose thresholds are applied unchanged to every row (transfer test)")
    p.add_argument("--max-benign-fa", type=float, default=None,
                   help="pick thresholds by a benign false-alarm budget on the validation half (e.g. 0.02)")
    p.add_argument("--threshold-rule", choices=("precision", "greedy"), default="precision")
    a = p.parse_args()
    summary = evaluate(a.model_dir, a.eval, a.out_dir, max_rows=a.max_rows,
                       min_precision=a.min_precision, device=a.device, design=a.design,
                       thresholds_from=a.thresholds_from, max_benign_fa=a.max_benign_fa,
                       threshold_rule=a.threshold_rule, revision=a.revision)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
