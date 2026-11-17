import pytest
import torch

from vigil.train import collate, compute_loss, parse_args, split_calibration


def _item(k, qtype, label, n_tokens):
    target = [0.0] * k
    target[label] = 1.0
    return {"ids": list(range(1, n_tokens + 1)), "markers": list(range(k)),
            "qtype": qtype, "target": target, "label": label}


def test_collate_pads_and_masks():
    batch = collate([_item(2, 2, 1, 5), _item(5, 1, 3, 9)], pad_id=0)
    assert batch["input_ids"].shape == (2, 9)
    assert batch["attention_mask"].sum(-1).tolist() == [5, 9]
    assert batch["marker_mask"].sum(-1).tolist() == [2, 5]
    assert batch["target"].shape == (2, 5)
    assert batch["qtype"].tolist() == [2, 1] and batch["label"].tolist() == [1, 3]


def test_split_calibration_is_disjoint_and_deterministic():
    items = [_item(2, 2, i % 2, 4) for i in range(100)]
    train_a, calib_a = split_calibration(items, seed=1, calib_max=400)
    train_b, calib_b = split_calibration(items, seed=1, calib_max=400)
    assert len(calib_a) == 10 and len(train_a) == 90  # n // 10 caps calib_max
    assert {id(x) for x in train_a}.isdisjoint({id(x) for x in calib_a})
    assert [id(x) for x in calib_a] == [id(x) for x in calib_b]


def test_split_calibration_respects_calib_max():
    items = [_item(2, 2, 0, 4) for _ in range(1000)]
    _, calib = split_calibration(items, seed=0, calib_max=25)
    assert len(calib) == 25


def test_split_calibration_keeps_whole_conversations_together():
    items = []
    for conv in range(50):
        for q in range(8):
            item = _item(2, 2, q % 2, 4)
            item["conv"] = conv
            items.append(item)
    train_items, calib = split_calibration(items, seed=3, calib_max=400)
    assert len(calib) == 40 and len(train_items) == 360  # n // 10 caps calib_max, in whole conversations
    assert {it["conv"] for it in calib}.isdisjoint({it["conv"] for it in train_items})


def _batch(correct: bool):
    logits = torch.tensor([[4.0, -4.0], [4.0, -4.0]] if correct else [[-4.0, 4.0], [-4.0, 4.0]],
                          requires_grad=True)
    batch = {
        "marker_mask": torch.ones(2, 2, dtype=torch.bool),
        "target": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "qtype": torch.tensor([2, 2]),
    }
    return logits, batch


def test_cross_entropy_only_loss_prefers_correct_logits():
    good, batch = _batch(True)
    bad, _ = _batch(False)
    loss_good, _ = compute_loss(good, batch, sigma=0.3, group_size=4, rl_weight=0.0)
    loss_bad, _ = compute_loss(bad, batch, sigma=0.3, group_size=4, rl_weight=0.0)
    assert loss_good.item() < 0.01 < loss_bad.item()


def test_loss_backpropagates_with_rl_term():
    logits, batch = _batch(False)
    torch.manual_seed(0)
    loss, reward = compute_loss(logits, batch, sigma=0.3, group_size=4, rl_weight=1.0)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.isfinite(reward)


def test_padded_options_do_not_leak_into_loss():
    logits = torch.tensor([[2.0, -2.0, 99.0]], requires_grad=True)
    batch = {
        "marker_mask": torch.tensor([[True, True, False]]),
        "target": torch.tensor([[1.0, 0.0, 0.0]]),
        "qtype": torch.tensor([2]),
    }
    loss, _ = compute_loss(logits, batch, sigma=0.3, group_size=2, rl_weight=0.0)
    assert loss.item() < 0.05


@pytest.mark.integration
@pytest.mark.slow
def test_tiny_training_run_lowers_loss_and_writes_a_checkpoint(base_dir, tok, tmp_path):
    from vigil.labels import Labels
    from vigil.rows import build_items
    from vigil.train import train

    items = []
    for state, labels in [
        ("Child: hi\nOther: how old are you? keep it secret", Labels(frozenset({"grooming"}), 3)),
        ("Child: hi\nOther: nice weather today", Labels(frozenset(), 0)),
    ]:
        items += build_items(tok, state, labels)[0]
    result = train(base_dir, items * 3, str(tmp_path / "out"), epochs=20, micro_batch=4,
                   grad_accum=1, calib_max=4, max_steps=40)
    losses = result["losses"]
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    for name in ("model.safetensors", "rl_agent_config.json"):
        assert (tmp_path / "out" / name).exists()
    assert len(result["temperatures"]) == 3


def test_train_cli_takes_a_seed_and_defaults_to_the_recorded_one():
    assert parse_args(["--items", "x.pt", "--output-dir", "out", "--seed", "5"]).seed == 5
    assert parse_args(["--items", "x.pt", "--output-dir", "out"]).seed == 20260922
