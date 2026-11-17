from vigil.deploy_calib import calibrate_on_deployment, run_experiment
from vigil.labels import CATEGORIES


def _row(gold, sev, grooming):
    return {"gold_categories": frozenset(gold), "gold_severity": sev,
            "probs": {c: (grooming if c == "grooming" else 0.0) for c in CATEGORIES},
            "severity_probs": [0, 0, 0, 1, 0], "latency_ms": 1.0}


def _val_rows():
    """Validation traffic: positives spread over 0.3-0.95, benign chats score low (0.05)."""
    pos = [_row({"grooming"}, 3, 0.3 + 0.65 * i / 59) for i in range(60)]
    return pos + [_row(set(), 0, 0.05) for _ in range(120)]


def _deployment_rows():
    """The deployment's benign chats score higher (0.5): the validation thresholds will flag them."""
    pos = [_row({"grooming"}, 3, 0.3 + 0.65 * i / 59) for i in range(60)]
    return pos + [_row(set(), 0, 0.5) for _ in range(120)]


def test_a_deployment_sample_moves_the_thresholds_and_cuts_false_alarms_on_the_rest():
    out = calibrate_on_deployment(_val_rows(), _deployment_rows(), n_cal=60, weight=1, seed=3, max_fa=0.02)
    assert out["baseline"]["fa"] > 0.5                      # validation-only thresholds flag the shifted benign chats
    assert out["calibrated"]["fa"] < out["baseline"]["fa"] / 5
    assert out["calibrated"]["thresholds"]["grooming"] > out["baseline"]["thresholds"]["grooming"]
    assert out["n_test"] == 180 - 60 and out["n_cal"] == 60


def test_calibration_and_test_rows_are_disjoint_and_the_split_is_deterministic():
    a = calibrate_on_deployment(_val_rows(), _deployment_rows(), n_cal=40, weight=1, seed=7, max_fa=0.02)
    b = calibrate_on_deployment(_val_rows(), _deployment_rows(), n_cal=40, weight=1, seed=7, max_fa=0.02)
    assert a == b
    assert set(a["cal_idx"]).isdisjoint(a["test_idx"]) and len(a["cal_idx"]) + len(a["test_idx"]) == 180


def test_run_experiment_reports_paired_differences_against_the_baseline():
    table = run_experiment(_val_rows(), _deployment_rows(), n_cals=(30, 60), weights=(1,), repeats=4, max_fa=0.02)
    assert [r["n_cal"] for r in table["rows"]] == [30, 60]
    for r in table["rows"]:
        assert r["fa_delta_mean"] < 0 and r["repeats"] == 4
    assert table["baseline"]["fa_mean"] > 0.5
