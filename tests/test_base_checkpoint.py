import json
import os

import pytest


@pytest.mark.integration
def test_checkpoint_layout(base_dir):
    for name in ("rl_agent_config.json", "model.safetensors", "encoder", "tokenizer"):
        assert os.path.exists(os.path.join(base_dir, name)), f"missing {name}"
    with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    assert "max_len" in cfg and "head_max_len" in cfg


def test_laya_exposes_training_helpers():
    from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options

    assert QTYPES == {"choice": 0, "score": 1, "noul": 2}
    assert all(callable(f) for f in (build_model, build_sequence, proper_reward, render_options))
