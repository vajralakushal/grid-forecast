import numpy as np
import pandas as pd

from gridfc import harness


def make_frame(days=60):
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="h", tz="UTC")
    y = 100_000 + 10_000 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24)
    return pd.DataFrame({
        "demand": y, "operator_fcst": y * 1.01,
        "wx_actual_temperature_2m": 5.0, "wx_fcst_d2_temperature_2m": 6.0,
    }, index=idx)


class Spy:
    """Fails if the harness ever shows it the future or forbidden columns."""
    name, refit_every_days = "spy", 7

    def fit(self, y, X):
        self._check(y, X)

    def predict(self, y, X_hist, X_future):
        self._check(y, X_hist)
        assert not any(c.startswith("wx_actual") for c in X_future.columns)
        assert y.index[-1] < X_future.index[0] - pd.Timedelta(hours=10)
        return pd.DataFrame({"mean": 1.0}, index=X_future.index)

    def _check(self, y, X):
        assert "operator_fcst" not in X.columns
        assert not any(c.startswith("wx_actual") for c in X.columns)
        assert "wx_fcst_d2_temperature_2m" in X.columns


def test_no_leakage_and_shape():
    out = harness.run_backtest(Spy(), make_frame(), "2025-01-10", "2025-01-20", verbose=False)
    assert len(out) == 11 * 24
    assert out["lead_h"].between(14, 38).all()


def test_targets_are_local_day():
    issue = harness.issue_times("2025-07-02", "2025-07-02")[0]
    t = harness.target_hours(issue).tz_convert(harness.TZ_PJM)
    assert t[0].hour == 0 and t[-1].hour == 23 and t[0].day == 2
    assert issue.tz_convert(harness.TZ_PJM).hour == 10
