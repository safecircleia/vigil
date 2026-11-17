import json

from vigil.evaluate import sample_indices
from vigil.mixval import build


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_sample_indices_is_sorted_deterministic_and_capped():
    assert sample_indices(100, 10, 3) == sample_indices(100, 10, 3)
    assert sample_indices(100, 10, 3) == sorted(sample_indices(100, 10, 3))
    assert len(sample_indices(5, 10, 3)) == 5


def test_build_excludes_the_evaluator_sample_and_appends_generated_rows(tmp_path):
    horizon = [{"messages": [], "n": i} for i in range(200)]
    gen = [{"messages": [], "meta": {"id": f"g{i}"}} for i in range(7)]
    _write(tmp_path / "h.jsonl", horizon)
    _write(tmp_path / "g.jsonl", gen)
    report = build(str(tmp_path / "h.jsonl"), [str(tmp_path / "g.jsonl")], str(tmp_path / "out.jsonl"),
                   n_horizon=30, seed=1, exclude_max_rows=50, exclude_seed=9)
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    excluded = set(sample_indices(200, 50, 9))
    horizon_rows = [r["n"] for r in rows if "n" in r]
    assert len(horizon_rows) == 30 and excluded.isdisjoint(horizon_rows)
    assert sum("meta" in r for r in rows) == 7
    assert report == {"horizon": 30, "generated": 7, "total": 37}
