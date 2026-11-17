import random

from laya.common import QTYPES

from vigil.calibrate import TEMP_MAX, TEMP_MIN, apply_temperatures, fit_one_temp, fit_temperatures


def _binary_sel(n, margin, accuracy, seed=0):
    rng = random.Random(seed)
    sel = []
    for _ in range(n):
        truth = rng.randint(0, 1)
        guess = truth if rng.random() < accuracy else 1 - truth
        logits = [0.0, 0.0]
        logits[guess] = margin
        target = [0.0, 0.0]
        target[truth] = 1.0
        sel.append((logits, target))
    return sel


def test_overconfident_model_gets_temperature_above_one():
    assert fit_one_temp(_binary_sel(400, margin=6.0, accuracy=0.75)) > 1.5


def test_underconfident_model_gets_temperature_below_one():
    assert fit_one_temp(_binary_sel(400, margin=0.2, accuracy=1.0)) < 1.0


def test_too_few_samples_returns_identity():
    assert fit_one_temp(_binary_sel(9, margin=6.0, accuracy=0.5)) == 1.0


def test_fit_temperatures_indexes_by_qtype_and_defaults_missing():
    preds = [(QTYPES["noul"], l, t) for l, t in _binary_sel(200, 6.0, 0.75)]
    temps = fit_temperatures(preds)
    assert len(temps) == 3
    assert temps[QTYPES["noul"]] > 1.5
    assert temps[QTYPES["choice"]] == 1.2 and temps[QTYPES["score"]] == 1.2


def test_apply_temperatures_replaces_and_drops_bucket_overrides():
    cfg = {"temperature": [1, 1, 1], "temperature_by_options": {"noul:2": 0.5}, "max_len": 1024}
    out = apply_temperatures(cfg, [1.1, 1.2, 1.3])
    assert out["temperature"] == [1.1, 1.2, 1.3]
    assert "temperature_by_options" not in out and out["fine_tuned"] is True
    assert "temperature_by_options" in cfg and cfg["temperature"] == [1, 1, 1]


def test_fitted_temperature_stays_inside_the_range_the_laya_runtime_honours():
    # laya clamps temperatures to [0.5, 5] at inference; a fitted value outside it would be silently replaced
    assert (TEMP_MIN, TEMP_MAX) == (0.5, 5.0)
    assert fit_one_temp(_binary_sel(400, margin=30.0, accuracy=0.5)) <= TEMP_MAX
    assert fit_one_temp(_binary_sel(400, margin=0.01, accuracy=1.0)) >= TEMP_MIN
