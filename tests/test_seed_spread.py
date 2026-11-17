from vigil.seed_spread import spread


def _report(fa_hits, fa_n, ch_hits, ch_n):
    return {"benign_false_alarms": {"hits": fa_hits, "support": fa_n},
            "catch_rate_category": {"hits": ch_hits, "support": ch_n}}


def test_spread_reports_mean_sd_and_range_across_runs():
    out = spread([_report(10, 100, 80, 100), _report(20, 100, 90, 100), _report(30, 100, 70, 100)])
    assert out["n"] == 3
    assert abs(out["fa_mean"] - 0.20) < 1e-9 and abs(out["fa_sd"] - 0.10) < 1e-9
    assert (out["fa_min"], out["fa_max"]) == (0.10, 0.30)
    assert abs(out["catch_mean"] - 0.80) < 1e-9 and abs(out["catch_sd"] - 0.10) < 1e-9


def test_spread_of_a_single_run_has_zero_sd():
    out = spread([_report(5, 50, 40, 50)])
    assert out["n"] == 1 and out["fa_sd"] == 0.0 and out["catch_sd"] == 0.0
