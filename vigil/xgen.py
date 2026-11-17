"""Cross-generator test set: labelled chats written by a different model and different prompts.

Horizon's training data came from one generator and one set of prompts. This module builds an
independent evaluation set (labels come from construction: we ask for a category and severity) to
check whether a tuned model's near-perfect scores depend on that generator. Everything requested is
non-explicit; a model that refuses is logged, not retried with pressure.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from vigil.labels import CATEGORIES, SEVERITY_LEVELS

DEFAULT_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "eu-west-3"
# ponytail: assumed Haiku 4.5 list prices in USD per million tokens; used only for the spend cap
PRICE_IN_PER_MTOK, PRICE_OUT_PER_MTOK = 1.0, 5.0
_MIN_TURNS = 4
_SEVERITIES = (1, 2, 3, 4)
_SEVERITY_WEIGHTS = (0.20, 0.35, 0.30, 0.15)

CATEGORY_GUIDE = {
    "grooming": "an older person slowly builds trust with the child: flattery, special attention, "
                "asking to keep things secret, offering gifts or favours, testing boundaries",
    "bullying": "peers insult, mock, exclude or spread rumours about the child",
    "sexual_content": "the other person makes inappropriate, suggestive remarks or requests aimed at the "
                      "child (for example asking for pictures or steering the talk toward intimate topics); "
                      "implied and non-graphic, with no explicit description",
    "isolation": "the other person persuades the child that family and friends do not understand them, "
                 "discourages telling parents, or the child says they feel cut off from everyone",
    "personal_info": "the other person asks for the child's address, school, schedule or phone number, "
                     "or the child shares such details",
    "platform_migration": "the other person pushes the child to move the chat to another, less monitored "
                          "app or a private group, or to delete the messages",
    "threats": "the other person threatens or blackmails the child (for example threatening to tell or "
               "share something), or the child says they are being threatened or hints at self-harm",
}
SEVERITY_GUIDE = {
    1: "low: a mild warning sign",
    2: "medium: a clear warning sign a guardian should review",
    3: "high: a serious, persistent risk",
    4: "critical: immediate danger to the child",
}
_PLATFORMS = ("Instagram DMs", "a game chat", "Discord", "Snapchat", "TikTok comments", "WhatsApp", "a school app")
_OTHER_ROLES = {
    "grooming": ("an adult stranger", "an adult pretending to be a teenager", "an older teenager"),
    "bullying": ("a classmate", "a group of classmates", "an older student"),
    "sexual_content": ("an adult stranger", "an older teenager", "an adult pretending to be a teenager"),
    "isolation": ("an adult who has befriended the child", "an older teenager", "a new online friend"),
    "personal_info": ("an adult stranger", "a new online friend", "someone claiming to run a contest"),
    "platform_migration": ("an adult stranger", "a new online friend", "an older teenager"),
    "threats": ("an older student", "an adult stranger", "someone the child once sent a photo to"),
}
_BENIGN_SCENES_BASIC = (
    "a teacher or sports coach talking normally about homework or training",
    "two friends joking and teasing each other playfully",
    "a parent or relative arranging pick-up and dinner",
    "two classmates chatting about a video game",
    "a classmate asking about homework",
    "the child asking an adult for advice about staying safe online",
    "friends planning a birthday party",
)
# more adult-child and family exchanges that are perfectly fine: the pilot models flagged these as risky
_BENIGN_SCENES_EXTENDED = _BENIGN_SCENES_BASIC + (
    "a tutor or music teacher confirming the next lesson",
    "an older sibling or cousin chatting about their day",
    "a grandparent or aunt asking about school and sending love",
    "the school office or a doctor's surgery confirming an appointment",
    "a coach sharing the match schedule and reminding about kit",
    "a club leader announcing when the next meeting is",
    "two teammates talking about yesterday's match",
    "a parent checking in about homework and dinner",
)
# round 2: the scenes the audit found hardest (light banter and online-safety talk), all harmless
_BENIGN_SCENES_SAFETY = (
    "a child asking a parent how to set privacy settings on a new app",
    "a teacher explaining to the class how to spot scam messages",
    "a child telling a parent about a strange message and getting calm, supportive advice",
    "friends good-naturedly joking about who lost the game, with no insults",
    "siblings lightly teasing each other about chores, all in good fun",
    "a child asking an older cousin how to block someone who keeps spamming them",
    "two friends complimenting each other's drawings and planning to draw together",
    "a child and a trusted adult discussing a news story about internet safety",
)
# round 5: harmless talk that *uses* the vocabulary of risk (the audit's confident false alarms), all educational or supportive
_BENIGN_SCENES_EDUCATIONAL = (
    "a parent explaining to a child why they should never share their home address with strangers online",
    "a teacher answering a child's questions about what grooming is and how to stay safe",
    "a child asking a school counsellor how to report a bully and getting practical advice",
    "a child asking an older sibling what to do if a stranger asks for photos",
    "a parent and child agreeing on rules for which apps the child may use and when",
    "a child telling a trusted adult that a classmate was unkind and the adult responding supportively",
    "a class group chat discussing an assembly about online safety",
    "a child asking an adult whether it is safe to move a chat from a game to another app, and being told to check with a parent first",
    "two friends sharing tips on blocking and reporting accounts they do not know",
    "a child asking a parent about a scary news story and being reassured",
)
# round 6: friendly teasing with mild, silly insults that everyone laughs at (benign by the user's policy);
# the audit found the models flag about half of these even though 43 of 46 audited held-out rows were friendly
_BENIGN_SCENES_BANTER = (
    "two friends teasing each other about being bad at a video game, using silly mild insults like noob, both laughing",
    "friends giving each other playful nicknames and joking about them, everyone laughing",
    "teammates joking about a missed goal in a friendly way, nobody upset",
    "siblings playfully teasing each other about chores and who is faster, all in good fun",
    "friends jokingly calling each other lazy or slow while planning something fun, clearly friendly",
    "classmates joking about a teacher's funny habit, laughing together, nothing mean",
    "friends trash-talking playfully about a game score and promising a rematch",
    "cousins teasing each other about their favourite football teams, all in good fun",
    "friends joking about each other's bad drawings or singing, clearly with affection",
    "friends teasing each other about eating too much pizza or being late, and laughing at themselves too",
)
_SCENE_SETS = {"basic": _BENIGN_SCENES_BASIC, "extended": _BENIGN_SCENES_EXTENDED,
               "safety": _BENIGN_SCENES_SAFETY, "educational": _BENIGN_SCENES_EDUCATIONAL,
               "banter": _BENIGN_SCENES_BANTER}


def make_specs(per_category: int, benign: int, seed: int, categories: tuple = CATEGORIES,
               scene_set: str = "basic") -> list:
    if scene_set not in _SCENE_SETS:
        raise ValueError(f"scene_set must be one of {tuple(_SCENE_SETS)}, got {scene_set!r}")
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        raise ValueError(f"unknown categories: {unknown}")
    scenes = _SCENE_SETS[scene_set]
    rng = random.Random(seed)
    specs = []
    for category in categories:
        for i in range(per_category):
            specs.append({
                "id": f"xg-{category}-{i:03d}", "category": category,
                "severity": rng.choices(_SEVERITIES, _SEVERITY_WEIGHTS)[0],
                "language": ("English", "Spanish")[i % 2], "child_age": rng.randint(9, 15),
                "platform": rng.choice(_PLATFORMS), "turns": rng.randint(8, 14),
                "other": rng.choice(_OTHER_ROLES[category]),
            })
    for i in range(benign):
        specs.append({
            "id": f"xg-benign-{i:03d}", "category": "benign", "severity": 0,
            "language": ("English", "Spanish")[i % 2], "child_age": rng.randint(9, 15),
            "platform": rng.choice(_PLATFORMS), "turns": rng.randint(8, 14),
            "scene": rng.choice(scenes),
        })
    return specs


def build_prompt(spec: dict) -> str:
    head = (
        "You are helping build a test set for a child-safety detection system used by parents and "
        f"platforms. Write a realistic chat on {spec['platform']} between a child (age {spec['child_age']}) "
        f"and another person, about {spec['turns']} turns long, in {spec['language']}"
        + (" (informal, as spoken in Spain)" if spec["language"] == "Spanish" else "") + ".\n"
    )
    if spec["category"] == "benign":
        body = (f"This is an ordinary, harmless conversation: {spec['scene']}. "
                "Nothing in it should worry a parent.\n")
    else:
        body = (f"Category: {spec['category']}. The chat shows: {CATEGORY_GUIDE[spec['category']]}.\n"
                f"The other person is {spec['other']}. Severity: {SEVERITY_GUIDE[spec['severity']]}.\n")
    rules = (
        "Rules: keep everything non-explicit and non-graphic; never describe sexual acts. "
        "Write natural, varied messages, not a lecture. Use exactly the speaker labels "
        "'Child:' and 'Other:' at the start of each line, alternating, even when writing Spanish. "
        "Output only the conversation, with no title, notes or commentary."
    )
    return head + body + rules


_TURN = re.compile(r"^[\*\s_]*(Child|Other)[\*\s_]*:[\*\s_]*(.+?)\s*$")


def parse_conversation(text: str):
    turns = []
    for line in text.splitlines():
        m = _TURN.match(line)
        if m and m.group(2).strip("*_ "):  # a turn that is only markdown noise carries no content
            turns.append((m.group(1), m.group(2).strip("* ")))
    if len(turns) < _MIN_TURNS or len({who for who, _ in turns}) < 2:
        return None
    return "\n".join(f"{who}: {msg}" for who, msg in turns)


def to_row(state: str, spec: dict) -> dict:
    target = {"risk_level": SEVERITY_LEVELS[spec["severity"]],
              "categories": [] if spec["category"] == "benign" else [spec["category"]]}
    return {
        "messages": [
            {"role": "user", "content": f"Analyze this conversation:\n{state}"},
            {"role": "assistant", "content": json.dumps(target)},
        ],
        "meta": spec,
    }


def _cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * PRICE_IN_PER_MTOK / 1e6 + output_tokens * PRICE_OUT_PER_MTOK / 1e6


def generate(specs: list, call, *, max_workers: int = 8, max_cost: float = 10.0) -> tuple:
    rows, refusals = [], []
    in_tok = out_tok = calls = 0
    stopped = False

    def one(spec):
        try:
            text, i, o = call(build_prompt(spec))
        except Exception as exc:  # noqa: BLE001 - a failed call is recorded, not fatal
            return spec, None, 0, 0, f"error: {exc}"
        return spec, parse_conversation(text), i, o, text[:200]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for start in range(0, len(specs), max_workers):
            if _cost(in_tok, out_tok) >= max_cost:
                stopped = True
                break
            for spec, state, i, o, note in pool.map(one, specs[start:start + max_workers]):
                calls += 1
                in_tok, out_tok = in_tok + i, out_tok + o
                if state is None:
                    refusals.append({"id": spec["id"], "reason": note})
                else:
                    rows.append(to_row(state, spec))
    rows.sort(key=lambda r: r["meta"]["id"])
    usage = {"calls": calls, "input_tokens": in_tok, "output_tokens": out_tok,
             "estimated_cost_usd": round(_cost(in_tok, out_tok), 4), "stopped_for_cost": stopped}
    return rows, refusals, usage


def generate_local(specs: list, batch_call, *, batch_size: int = 32) -> tuple:
    """Like generate(), for a local model that answers a whole batch of prompts at once."""
    rows, refusals = [], []
    in_tok = out_tok = calls = 0
    for start in range(0, len(specs), batch_size):
        chunk = specs[start:start + batch_size]
        for spec, (text, i, o) in zip(chunk, batch_call([build_prompt(sp) for sp in chunk])):
            calls += 1
            in_tok, out_tok = in_tok + i, out_tok + o
            state = parse_conversation(text)
            if state is None:
                refusals.append({"id": spec["id"], "reason": text[:200]})
            else:
                rows.append(to_row(state, spec))
    rows.sort(key=lambda r: r["meta"]["id"])
    return rows, refusals, {"calls": calls, "input_tokens": in_tok, "output_tokens": out_tok}


def hf_batch_call(model_id: str, *, max_new_tokens: int = 900, temperature: float = 0.9, top_p: float = 0.95):
    # ponytail: plain transformers.generate, left-padded batches; add vLLM only if this is too slow
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")  # no device_map: it needs accelerate

    def call(prompts: list) -> list:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, top_p=top_p, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [(tok.decode(g, skip_special_tokens=True), int(enc["attention_mask"][i].sum()),
                 int((g != tok.pad_token_id).sum())) for i, g in enumerate(gen)]

    return call


def _post(url: str, headers: dict, body: dict) -> dict:
    # ponytail: plain urllib with retries on throttling/5xx; not unit-tested (network)
    data = json.dumps(body).encode()
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def bedrock_call(model_id: str, region: str, api_key: str, *, post=_post, max_tokens: int = 1200):
    url = (f"https://bedrock-runtime.{region}.amazonaws.com/model/"
           f"{urllib.parse.quote(model_id, safe='.-_')}/converse")
    headers = {"Authorization": f"Bearer {api_key}"}

    def call(prompt: str):
        body = {"messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": max_tokens, "temperature": 1.0}}
        reply = post(url, headers, body)
        text = "".join(part.get("text", "") for part in reply["output"]["message"]["content"])
        usage = reply.get("usage", {})
        return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0)

    return call


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a cross-generator test set with Bedrock")
    p.add_argument("--out", required=True)
    p.add_argument("--per-category", type=int, default=100)
    p.add_argument("--benign", type=int, default=150)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-cost", type=float, default=10.0)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--backend", choices=("bedrock", "hf"), default="bedrock")
    p.add_argument("--hf-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--categories", default=",".join(CATEGORIES), help="comma-separated subset of categories")
    p.add_argument("--scenes", choices=tuple(_SCENE_SETS), default="basic")
    a = p.parse_args()
    specs = make_specs(a.per_category, a.benign, a.seed, tuple(a.categories.split(",")), a.scenes)
    if a.backend == "hf":
        import torch

        torch.manual_seed(a.seed)
        rows, refusals, usage = generate_local(specs, hf_batch_call(a.hf_model), batch_size=a.batch_size)
        a.model_id = a.hf_model
    else:
        key = os.environ.get("BEDROCK_API_KEY")
        if not key:
            raise SystemExit("set BEDROCK_API_KEY in the environment")
        rows, refusals, usage = generate(specs, bedrock_call(a.model_id, a.region, key),
                                         max_workers=a.workers, max_cost=a.max_cost)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(a.out + ".refusals.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in refusals)
    print(json.dumps({"requested": len(specs), "kept": len(rows), "refused_or_failed": len(refusals),
                      "model": a.model_id, **usage}, indent=2))


if __name__ == "__main__":
    main()
