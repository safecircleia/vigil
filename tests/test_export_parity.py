import json
import os

import pytest

CHECKPOINT = os.environ.get("VIGIL_CHECKPOINT")
EVAL = os.environ.get("VIGIL_EVAL", "data/source/eval.jsonl")
DESIGN = os.environ.get("VIGIL_DESIGN", "noul7")

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT and os.path.exists(EVAL)),
    reason="set VIGIL_CHECKPOINT to a trained checkpoint dir (and have data/source/eval.jsonl)",
)


def _states(n=25):
    from vigil.rows import split_messages

    out = []
    with open(EVAL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(split_messages(json.loads(line)["messages"])[0])
            if len(out) == n:
                break
    return out


@pytest.mark.integration
def test_onnx_matches_pytorch_within_tolerance():
    import laya
    from laya.onnx_agent import ONNXAgent

    from vigil.evaluate import _category_probs
    from vigil.labels import CATEGORIES
    from vigil.questions import build_questions

    onnx_path = os.path.join(CHECKPOINT, "vigil.onnx")
    assert os.path.exists(onnx_path), "run scripts/export_onnx.py first"
    torch_agent = laya.Agent(CHECKPOINT, device="cpu")
    onnx_agent = ONNXAgent(CHECKPOINT, onnx_path=onnx_path)
    questions = build_questions(DESIGN)
    worst, flips = 0.0, 0
    for state in _states():
        a = torch_agent.predict(state, questions)["answers"]
        b = onnx_agent.predict(state, questions)["answers"]
        for c in CATEGORIES:
            pa, pb = _category_probs(a, DESIGN)[c], _category_probs(b, DESIGN)[c]
            worst = max(worst, abs(pa - pb))
            flips += (pa >= 0.5) != (pb >= 0.5)
    print(f"max |dp| = {worst:.4f}, decision flips = {flips}")
    assert worst < 0.02, worst
    assert flips == 0
