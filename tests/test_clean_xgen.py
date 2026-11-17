import json

from vigil.clean_xgen import clean_file, has_empty_turn


def _row(state):
    return json.dumps({"messages": [
        {"role": "user", "content": f"Analyze this conversation:\n{state}"},
        {"role": "assistant", "content": json.dumps({"risk_level": "none", "categories": []})},
    ]})


def test_has_empty_turn_sees_blank_and_markdown_only_turns():
    assert has_empty_turn("Child: hi\nOther: ")
    assert has_empty_turn("Child: hi\nOther: **")
    assert has_empty_turn("Child: *\nOther: hello")
    assert not has_empty_turn("Child: hi\nOther: hello there\nChild: bye\nOther: bye")


def test_clean_file_drops_bad_rows_and_keeps_a_backup(tmp_path):
    path = tmp_path / "gen.jsonl"
    good1, bad, good2 = "Child: a\nOther: b", "Child: a\nOther: **", "Child: c\nOther: d"
    path.write_text("\n".join(_row(s) for s in (good1, bad, good2)) + "\n")
    assert clean_file(str(path)) == {"kept": 2, "dropped": 1}
    kept = [json.loads(line)["messages"][0]["content"] for line in path.read_text().splitlines()]
    assert kept == [f"Analyze this conversation:\n{good1}", f"Analyze this conversation:\n{good2}"]
    assert len((tmp_path / "gen.jsonl.orig").read_text().splitlines()) == 3


def test_clean_file_is_idempotent_and_never_overwrites_the_backup(tmp_path):
    path = tmp_path / "gen.jsonl"
    path.write_text(_row("Child: a\nOther: b") + "\n" + _row("Child: a\nOther: *") + "\n")
    clean_file(str(path))
    assert clean_file(str(path)) == {"kept": 1, "dropped": 0}
    assert len((tmp_path / "gen.jsonl.orig").read_text().splitlines()) == 2
