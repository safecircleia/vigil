"""Write the non-model files the Android app needs next to the .tflite, and golden test vectors for the Kotlin port.

  vigil_tokenizer.bin       compact BPE tokenizer (see vigil/bpe.py)
  tokenizer_golden.json     [{text, ids}] from the Hugging Face tokenizer, for the Kotlin tokenizer test
  inference_golden.json     [{text, category_ids, severity_ids, category_probs, severity_score}] from the PyTorch agent

Fails (non-zero exit) if the reference `Bpe` disagrees with the Hugging Face tokenizer on any string.
"""
import argparse
import json
import os
import random

EDGE_CASES = [
    "", " ", "hello", " hello", "  hello  world  ", "a\tb", "line1\nline2\n\nline4", "Other: hi\nChild: hey\n",
    "Hello, how are you? I'm fine!", "URL https://example.com/a_b?x=1&y=2#frag", "e-mail me: a.b@c.org",
    "emoji 😀🎉 mixed 👍🏽 text", "日本語のテキスト", "中文消息：你好吗？", "مرحبا كيف حالك", "Привет, как дела?",
    "naïve café — “quotes” and ‘single’", "zero​width and  nbsp", "tab\tand\r\nCRLF",
    "<unused0>", "pre <unused0> post", "<unused0><unused1>", "<mask> token", "<bos><eos><pad>", "<h1>title</h1>",
    "x" * 300, "ab" * 200, "▁already▁spaced", "▁", "1234567890 0.5 3.14159", "ALL CAPS SHOUTING!!!",
    "i hate u u r so stupid lol", "Don't tell your parents about this. It's our little secret.",
    "\x00\x01 control chars", "mixed ́ combining accents é", "𐍈 rare plane-1 symbol", "end with space ",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="data/source/eval.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--corpus", type=int, default=3000, help="eval conversations tokenized for the parity check")
    args = ap.parse_args()
    from vigil.checkpoint import validate_checkpoint_dir

    validate_checkpoint_dir(args.model)

    from laya.agent import Agent
    from transformers import AutoTokenizer

    from vigil.bpe import Bpe, write_compact
    from vigil.labels import CATEGORIES
    from vigil.ondevice import OnDevice
    from vigil.questions import build_questions
    from vigil.rows import split_messages

    os.makedirs(args.out_dir, exist_ok=True)
    stats = write_compact(os.path.join(args.model, "tokenizer", "tokenizer.json"),
                          os.path.join(args.out_dir, "vigil_tokenizer.bin"))
    print("tokenizer.bin", stats, flush=True)
    bpe = Bpe(os.path.join(args.out_dir, "vigil_tokenizer.bin"))
    hf = AutoTokenizer.from_pretrained(os.path.join(args.model, "tokenizer"))

    lines = [l for l in open(args.eval, encoding="utf-8") if l.strip()]
    random.Random(11).shuffle(lines)
    states = [split_messages(json.loads(l)["messages"])[0] for l in lines[: args.corpus]]
    corpus = EDGE_CASES + states
    golden, bad = [], 0
    for text in corpus:
        want = hf(text, add_special_tokens=False)["input_ids"]
        got = bpe.encode(text)
        if want != got:
            bad += 1
            if bad <= 10:
                print("MISMATCH", repr(text[:80]), want[:20], got[:20], flush=True)
        if text in EDGE_CASES or len(golden) < len(EDGE_CASES) + 150:
            golden.append({"text": text, "ids": want})
    print(f"tokenizer parity: {len(corpus) - bad}/{len(corpus)} identical", flush=True)
    with open(os.path.join(args.out_dir, "tokenizer_golden.json"), "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False)

    agent = Agent(args.model, compile=False, device="cpu")
    thresholds = json.load(open(args.thresholds))["thresholds"]
    runtime = OnDevice.from_agent(agent, args.seq_len, thresholds)
    questions = build_questions("choice8")
    inference = []
    for state in [s for s in states if runtime.fits(s)][:12]:
        ids, mask, *_ = runtime.encode(state)
        a = agent.predict(state, questions)["answers"]
        inference.append({
            "text": state,
            "category_ids": ids[0][: int(mask[0].sum())].tolist(),
            "severity_ids": ids[1][: int(mask[1].sum())].tolist(),
            "category_probs": [a["category"]["probabilities"].get(c, 0.0) for c in CATEGORIES],
            "severity_score": a["severity"]["score"],
        })
    with open(os.path.join(args.out_dir, "inference_golden.json"), "w", encoding="utf-8") as f:
        json.dump(inference, f, ensure_ascii=False)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
