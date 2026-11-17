import json
from collections import Counter

import pytest

from vigil.labels import CATEGORIES, parse_target
from vigil.rows import split_messages
from vigil.xgen import build_prompt, generate, generate_local, make_specs, parse_conversation, to_row

GOOD = "Child: hey, who is this?\nOther: just a friend of a friend\nChild: ok\nOther: you seem cool, how old are you?"


def test_make_specs_counts_languages_and_severities():
    specs = make_specs(per_category=10, benign=15, seed=1)
    assert len(specs) == 7 * 10 + 15
    assert len({s["id"] for s in specs}) == len(specs)
    by_cat = Counter(s["category"] for s in specs)
    assert all(by_cat[c] == 10 for c in CATEGORIES) and by_cat["benign"] == 15
    for c in CATEGORIES:
        langs = Counter(s["language"] for s in specs if s["category"] == c)
        assert abs(langs["English"] - langs["Spanish"]) <= 1
    assert all(1 <= s["severity"] <= 4 for s in specs if s["category"] != "benign")
    assert all(s["severity"] == 0 for s in specs if s["category"] == "benign")


def test_make_specs_is_deterministic():
    assert make_specs(5, 5, seed=3) == make_specs(5, 5, seed=3)
    assert make_specs(5, 5, seed=3) != make_specs(5, 5, seed=4)


def test_build_prompt_carries_the_requirements():
    spec = make_specs(1, 1, seed=1)[0]
    prompt = build_prompt(spec)
    assert spec["language"] in prompt and "Child:" in prompt and "Other:" in prompt
    assert "non-explicit" in prompt.lower() and "only the conversation" in prompt.lower()
    benign = next(s for s in make_specs(1, 3, seed=1) if s["category"] == "benign")
    assert "ordinary" in build_prompt(benign).lower()


def test_parse_conversation_keeps_only_turns_and_strips_markdown():
    text = "Here is the chat:\n\n**Child:** hey\n**Other:** hi there\nChild: what's up\nOther: not much\n\nHope this helps"
    assert parse_conversation(text) == "Child: hey\nOther: hi there\nChild: what's up\nOther: not much"


@pytest.mark.parametrize("text", [
    "I can't help with that request.",
    "Child: only one speaker\nChild: still one\nChild: again\nChild: more",
    "Child: hi\nOther: hey",  # too short
    "",
])
def test_parse_conversation_rejects_refusals_and_short_or_one_sided_text(text):
    assert parse_conversation(text) is None


def test_to_row_round_trips_through_the_training_parsers():
    spec = {"id": "x1", "category": "grooming", "severity": 3, "language": "English"}
    row = to_row(GOOD, spec)
    state, raw = split_messages(row["messages"])
    assert state == GOOD
    labels = parse_target(raw)
    assert labels.categories == frozenset({"grooming"}) and labels.severity == 3
    assert row["meta"]["id"] == "x1"
    benign = to_row(GOOD, {"id": "b", "category": "benign", "severity": 0, "language": "Spanish"})
    assert parse_target(split_messages(benign["messages"])[1]).categories == frozenset()


def test_generate_collects_rows_counts_refusals_and_sorts_by_id():
    specs = make_specs(2, 2, seed=1)

    def call(prompt):
        if "sexual_content" in prompt and "Spanish" in prompt:
            return "I can't help with that.", 100, 10
        return GOOD, 100, 60

    rows, refusals, usage = generate(specs, call, max_workers=3, max_cost=10.0)
    assert len(rows) + len(refusals) == len(specs) and len(refusals) >= 1
    assert [r["meta"]["id"] for r in rows] == sorted(r["meta"]["id"] for r in rows)
    assert usage["calls"] == len(specs) and not usage["stopped_for_cost"]


def test_generate_stops_when_the_cost_cap_is_reached():
    specs = make_specs(5, 5, seed=1)
    calls = []

    def call(prompt):
        calls.append(1)
        return GOOD, 5_000_000, 0  # $5 of input per call at $1/MTok

    rows, refusals, usage = generate(specs, call, max_workers=2, max_cost=10.0)
    assert usage["stopped_for_cost"] and len(calls) < len(specs)
    assert usage["estimated_cost_usd"] >= 10.0
    json.dumps(usage)


def test_generate_records_a_failed_call_instead_of_losing_the_run():
    specs = make_specs(1, 1, seed=1)

    def call(prompt):
        if "grooming" in prompt:
            raise RuntimeError("throttled")
        return GOOD, 10, 10

    rows, refusals, usage = generate(specs, call, max_workers=2, max_cost=10.0)
    assert any("throttled" in r["reason"] for r in refusals)
    assert len(rows) + len(refusals) == len(specs)


def test_bedrock_call_builds_the_converse_request_and_parses_the_reply():
    from vigil.xgen import bedrock_call

    seen = {}

    def post(url, headers, body):
        seen.update(url=url, headers=headers, body=body)
        return {"output": {"message": {"content": [{"text": GOOD}]}},
                "usage": {"inputTokens": 12, "outputTokens": 34}}

    call = bedrock_call("eu.anthropic.claude-haiku-4-5-20251001-v1:0", "eu-west-3", "secret-key", post=post)
    assert call("write a chat") == (GOOD, 12, 34)
    assert seen["url"] == ("https://bedrock-runtime.eu-west-3.amazonaws.com/model/"
                           "eu.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse")
    assert seen["headers"]["Authorization"] == "Bearer secret-key"
    assert seen["body"]["messages"][0]["content"][0]["text"] == "write a chat"


def test_generate_local_batches_prompts_and_records_refusals():
    specs = make_specs(2, 2, seed=1)
    sizes = []

    def batch_call(prompts):
        sizes.append(len(prompts))
        return [("I cannot write that." if "sexual_content" in p else GOOD, 50, 30) for p in prompts]

    rows, refusals, usage = generate_local(specs, batch_call, batch_size=5)
    assert sizes == [5, 5, 5, 1]  # 16 specs in batches of at most 5
    assert len(rows) + len(refusals) == len(specs) == 16
    assert {r["id"].split("-")[1] for r in refusals} == {"sexual_content"}
    assert [r["meta"]["id"] for r in rows] == sorted(r["meta"]["id"] for r in rows)
    assert usage["calls"] == 16 and usage["input_tokens"] == 50 * 16


def test_make_specs_can_leave_categories_out_and_covers_more_benign_scenes():
    specs = make_specs(4, 60, seed=2, categories=("grooming", "threats"), scene_set="extended")
    assert {s["category"] for s in specs} == {"grooming", "threats", "benign"}
    assert len([s for s in specs if s["category"] != "benign"]) == 8
    assert len({s["scene"] for s in specs if s["category"] == "benign"}) >= 10
    basic = make_specs(1, 80, seed=2)
    assert len({s["scene"] for s in basic if s["category"] == "benign"}) == 7  # default set is unchanged
    with pytest.raises(ValueError):
        make_specs(1, 1, seed=1, categories=("grooming", "nonsense"))
    with pytest.raises(ValueError):
        make_specs(1, 1, seed=1, scene_set="nope")


def test_safety_scene_set_is_benign_only_material_and_leaves_existing_sets_alone():
    specs = make_specs(0, 60, seed=5, scene_set="safety")
    assert len(specs) == 60 and {s["category"] for s in specs} == {"benign"}
    scenes = {s["scene"] for s in specs}
    assert len(scenes) >= 8
    extended = {s["scene"] for s in make_specs(0, 200, seed=5, scene_set="extended")}
    assert scenes.isdisjoint(extended)  # the earlier training data used exactly the extended scenes
    assert make_specs(2, 2, seed=1) == make_specs(2, 2, seed=1, scene_set="basic")


def test_parse_conversation_drops_turns_with_no_content_and_rejects_hollow_chats():
    hollow = "Child: **\nOther: \nChild: *\nOther: **\nChild: \nOther: *"
    assert parse_conversation(hollow) is None
    mixed = "Child: hey\nOther: **\nChild: what's up\nOther: not much\nChild: cool\nOther: yeah"
    assert parse_conversation(mixed) == "Child: hey\nChild: what's up\nOther: not much\nChild: cool\nOther: yeah"


def test_educational_scene_set_is_benign_talk_about_risky_topics_and_disjoint_from_earlier_sets():
    specs = make_specs(0, 120, seed=6, scene_set="educational")
    assert {s["category"] for s in specs} == {"benign"}
    scenes = {s["scene"] for s in specs}
    assert len(scenes) >= 9
    for earlier in ("basic", "extended", "safety"):
        assert scenes.isdisjoint({s["scene"] for s in make_specs(0, 300, seed=6, scene_set=earlier)})


def test_banter_scene_set_is_friendly_teasing_and_disjoint_from_earlier_sets():
    specs = make_specs(0, 150, seed=8, scene_set="banter")
    assert {s["category"] for s in specs} == {"benign"}
    scenes = {s["scene"] for s in specs}
    assert len(scenes) >= 9 and all("laugh" in s or "joke" in s or "playful" in s or "good fun" in s or "friendly" in s
                                    or "affection" in s for s in scenes)
    for earlier in ("basic", "extended", "safety", "educational"):
        assert scenes.isdisjoint({s["scene"] for s in make_specs(0, 300, seed=8, scene_set=earlier)})
