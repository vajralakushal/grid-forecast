import numpy as np
import pandas as pd

from gridfc import harness
from gridfc.harness import QCOLS
from gridfc.models import REGISTRY
from gridfc.models.gbm import day_ahead_features


def make_frame(days=90):
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    temp = 10 + 8 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 2, len(idx))
    y = 90_000 + 800 * np.abs(temp - 15) + rng.normal(0, 500, len(idx))
    return pd.DataFrame({"demand": y, "wx_actual_temperature_2m": temp,
                         "wx_fcst_d2_temperature_2m": temp + 0.5}, index=idx)


def test_features_use_only_pre_cutoff_demand():
    frame = make_frame()
    issue = harness.issue_times("2025-03-01", "2025-03-01")[0]
    targets = harness.target_hours(issue)
    cutoff = issue - pd.Timedelta(hours=harness.DATA_LAG_H)
    X = harness.allowed_features(frame)
    full = day_ahead_features(frame["demand"], X.reindex(targets))
    seen = day_ahead_features(frame["demand"][:cutoff], X.reindex(targets))
    pd.testing.assert_frame_equal(full, seen)  # future demand changes nothing
    assert seen[["lag48", "lag168", "last", "d2_mean"]].notna().all().all()


def test_lgbm_runs_through_harness():
    for name in ["lgbm", "lgbm_wxact"]:
        model = REGISTRY[name]()
        model.params["n_estimators"] = 30
        out = harness.run_backtest(model, make_frame(), "2025-03-20", "2025-03-24", verbose=False)
        assert len(out) == 5 * 24 and out[["mean"] + QCOLS].notna().all().all()
        assert (np.diff(out[QCOLS].values, axis=1) >= 0).all()

