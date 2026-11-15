import numpy as np
import pandas as pd

from gridfc.harness import QCOLS
from gridfc.metrics import score


def test_partial_quantiles_score_what_they_can():
    """TimesFM has no q05/q95: 80% coverage is scored, 90% coverage and pinball are not."""
    y = np.array([100.0, 110.0, 90.0, 105.0])
    d = pd.DataFrame({"y": y, "mean": 100.0, "q10": 95.0, "q25": 98.0, "q50": 100.0,
                      "q75": 102.0, "q90": 106.0, "q05": np.nan, "q95": np.nan})
    s = score(d)
    assert s["cover80_%"] == 50.0  # 100 and 105 are inside 95-106
    assert "cover90_%" not in s and "pinball_%" not in s
    full = d.assign(q05=90.0, q95=110.0)
    assert set(["cover90_%", "pinball_%"]) <= set(score(full).index)
    assert score(full)["cover90_%"] == 100.0
    assert list(full[QCOLS].columns) == QCOLS


def test_dm_and_bootstrap_detect_a_real_gap():
    """A constant 0.2-point daily gap with noise is significant; pure noise is not."""
    from gridfc.metrics import block_bootstrap_ci, dm_test
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.normal(0, 1, 600))
    gap = noise + 0.2
    assert dm_test(gap)["p_two_sided"] < 0.01
    lo, hi = block_bootstrap_ci(gap)
    assert lo > 0 and hi > lo
    assert dm_test(noise)["p_two_sided"] > 0.05


def test_peak_scores():
    from gridfc.metrics import peak_scores
    t = pd.date_range("2025-07-01 04:00", periods=48, freq="h", tz="UTC")  # 00:00 Eastern
    y = np.r_[np.arange(24.0), np.arange(24.0)] + 100  # peaks at 23:00 Eastern each day
    f = y.copy()
    f[22], f[23] = 130, 100  # day 1: forecast peak at 22:00, 130 vs actual 123
    d = pd.DataFrame({"target_time": t, "y": y, "mean": f})
    s = peak_scores(d, "America/New_York")
    assert s["n_days"] == 2 and s["peak_hour_exact_%"] == 50 and s["peak_hour_MAE_h"] == 0.5
    assert np.isclose(s["peak_MAPE_%"], (7 / 123 * 100) / 2)
