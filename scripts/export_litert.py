"""Export a Vigil checkpoint to a LiteRT (.tflite) file for the Android app, and measure how faithful it is.

The graph takes the two question rows of one conversation (the 8-way `category` choice and the
5-level `severity` score), shape [2, seq_len], and returns the marker logits [2, 8]. The softmax
(with the checkpoint's temperature), the thresholds and the tokenizer live outside the graph; see
`vigil/ondevice.py` for the reference implementation the Kotlin port mirrors.

Needs `litert-torch` (a separate environment: see slurm/export_litert.sbatch).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from vigil.checkpoint import validate_checkpoint_dir
from vigil.ondevice import OnDevice, decode_category

BATCH, MAX_MARKERS = 2, 8


class ExportWrapper(torch.nn.Module):
    """DecisionModel.forward minus the act head, with int32 inputs and explicit attention masks.

    HF builds its own masks inside the encoder, and a padded query row inside a sliding window
    is then fully masked, which the converted graph turns into NaN. Every row keeps its own
    diagonal here instead; padded positions are never read (only marker positions are).
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.half_window = model.encoder.config.local_attention // 2

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m = self.m
        valid = attention_mask.bool()
        n = input_ids.shape[1]
        r = torch.arange(n)
        full = valid[:, None, None, :] | (r[:, None] == r[None, :])[None, None]
        local = full & ((r[:, None] - r[None, :]).abs() <= self.half_window)[None, None]
        h = m.encoder(input_ids=input_ids.long(),
                      attention_mask={"full_attention": full, "sliding_attention": local}).last_hidden_state
        h = h + m.type_emb(qtype.long())[:, None, :]
        for layer in m.head.layers:
            h = layer(h, src_key_padding_mask=~valid)
        idx = marker_pos.long().clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = m.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
        return logits.masked_fill(~marker_mask.bool(), -1e4)


def convert(agent, seq_len: int, out_path: str) -> None:
    import litert_torch

    wrapper = ExportWrapper(agent.model.eval().float()).eval()
    ids = torch.randint(5, 1000, (BATCH, seq_len), dtype=torch.int32)
    mask = torch.ones(BATCH, seq_len, dtype=torch.int32)
    mask[1, seq_len // 3:] = 0
    markers = torch.tensor([[17, 27, 39, 47, 58, 69, 80, 93], [17, 30, 47, 64, 79, 0, 0, 0]], dtype=torch.int32)
    marker_mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3], dtype=torch.int32)
    qtype = torch.tensor([0, 1], dtype=torch.int32)
    litert_torch.convert(wrapper, (ids, mask, markers, marker_mask, qtype)).export(out_path)


def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


def quantize(src: str, dst: str, variant: str) -> None:
    """Weight-only variants: weights stored small, compute stays float.

    w8: every supported op int8 (per channel). emb8: only the embedding table int8, so the
    transformer weights keep full precision. fp16: float16 weights. emb8_fc16: embedding int8,
    matmul weights float16.
    """
    from ai_edge_quantizer import quantizer, recipe
    from ai_edge_quantizer.qtyping import TFLOperationName as Op
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    qt = quantizer.Quantizer(src)
    if variant == "w8":
        qt.load_quantization_recipe(recipe.weight_only_wi8_afp32())
    else:
        embedding_bits = 16 if variant == "fp16" else 8
        kw = {"algorithm_key": AlgorithmName.FLOAT_CASTING} if embedding_bits == 16 else {}
        qt.add_weight_only_config(".*", Op.EMBEDDING_LOOKUP, embedding_bits, **kw)
        if variant in ("fp16", "emb8_fc16"):
            for op in (Op.FULLY_CONNECTED, Op.BATCH_MATMUL):
                qt.add_weight_only_config(".*", op, 16, algorithm_key=AlgorithmName.FLOAT_CASTING)
    qt.quantize().export_model(dst, overwrite=True)


def run_tflite(path: str, runtime: OnDevice, states: list, threads: int) -> tuple:
    from ai_edge_litert.interpreter import Interpreter

    rss0 = _rss_mb()
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    rss = _rss_mb() - rss0
    inputs, output = it.get_input_details(), it.get_output_details()[0]
    logits, lat = [], []
    for state in states:
        batch = runtime.encode(state)
        t0 = time.perf_counter()
        for det, arr in zip(inputs, batch):
            it.set_tensor(det["index"], arr.astype(det["dtype"]))
        it.invoke()
        logits.append(it.get_tensor(output["index"]).copy())
        lat.append((time.perf_counter() - t0) * 1000)
    return logits, lat, rss


def compare(name, logits, lat, rss, runtime, ref_cat, ref_sev, thresholds):
    cats = np.array([decode_category(l, runtime.spec)[0] for l in logits])
    sevs = np.array([decode_category(l, runtime.spec)[1] for l in logits])
    flags, ref_flags = cats >= thresholds, ref_cat >= thresholds
    log_drift = np.abs(np.log(np.clip(cats, 1e-9, 1)) - np.log(np.clip(ref_cat, 1e-9, 1)))
    lat = sorted(lat)
    return {
        "variant": name,
        "max_abs_prob_diff": round(float(np.abs(cats - ref_cat).max()), 5),
        "max_abs_log_prob_diff": round(float(log_drift.max()), 4),
        "max_abs_severity_diff": round(float(np.abs(sevs - ref_sev).max()), 5),
        "decision_flips": f"{int((flags != ref_flags).sum())}/{flags.size}",
        "positives_in_reference": int(ref_flags.sum()),
        "latency_ms_p50": round(lat[len(lat) // 2]),
        "latency_ms_p95": round(lat[int(len(lat) * 0.95)]),
        "size_mb": round(os.path.getsize(name) / 1e6),
        "resident_mb_after_load": round(rss),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="trained choice8 checkpoint dir")
    ap.add_argument("--thresholds", required=True, help="evaluate report.json holding the per-category thresholds")
    ap.add_argument("--eval", default="data/source/eval.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--rows", type=int, default=300, help="eval conversations for the fidelity check")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    validate_checkpoint_dir(args.model)

    from laya.agent import Agent

    from vigil.labels import CATEGORIES
    from vigil.questions import build_questions
    from vigil.rows import split_messages

    os.makedirs(args.out_dir, exist_ok=True)
    agent = Agent(args.model, compile=False, device="cpu")
    thresholds_by_cat = json.load(open(args.thresholds))["thresholds"]
    runtime = OnDevice.from_agent(agent, args.seq_len, thresholds_by_cat)
    spec_path = os.path.join(args.out_dir, "vigil_spec.json")
    json.dump(runtime.spec, open(spec_path, "w"), indent=1)

    fp32 = os.path.join(args.out_dir, f"vigil_s{args.seq_len}_fp32.tflite")
    convert(agent, args.seq_len, fp32)
    paths = [fp32]
    for variant in ("w8", "emb8", "fp16", "emb8_fc16"):
        path = os.path.join(args.out_dir, f"vigil_s{args.seq_len}_{variant}.tflite")
        try:
            quantize(fp32, path, variant)
            paths.append(path)
        except Exception as exc:  # a recipe the quantizer rejects must not lose the other variants
            print(f"variant {variant} skipped: {exc!r}", flush=True)

    lines = [l for l in open(args.eval, encoding="utf-8") if l.strip()]
    random.Random(7).shuffle(lines)
    states = [split_messages(json.loads(l)["messages"])[0] for l in lines[: args.rows * 2]]
    fits = [s for s in states if runtime.fits(s)]
    states = fits[: args.rows]
    questions = build_questions("choice8")
    ref_cat, ref_sev = [], []
    for s in states:
        a = agent.predict(s, questions)["answers"]
        ref_cat.append([a["category"]["probabilities"].get(c, 0.0) for c in CATEGORIES])
        ref_sev.append(a["severity"]["score"])
    ref_cat, ref_sev = np.array(ref_cat), np.array(ref_sev)
    thresholds = np.array([thresholds_by_cat[c] for c in CATEGORIES])

    report = {"seq_len": args.seq_len, "rows": len(states), "rows_skipped_overflow": len(lines[: args.rows * 2]) - len(fits[: args.rows * 2]), "variants": []}
    for path in paths:
        logits, lat, rss = run_tflite(path, runtime, states, args.threads)
        report["variants"].append(compare(path, logits, lat, rss, runtime, ref_cat, ref_sev, thresholds))
        print(json.dumps(report["variants"][-1]), flush=True)
    json.dump(report, open(os.path.join(args.out_dir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
