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


class ActualSpy(Spy):
    """In hist_weather='actual' mode: old history shows observed weather, recent history
    and targets show the d2 forecast. Never any wx_actual_* column names."""
    hist_weather = "actual"

    def predict(self, y, X_hist, X_future):
        wx = X_hist["wx_fcst_d2_temperature_2m"]
        old = X_future.index[0] - pd.Timedelta(days=harness.ACTUAL_WX_LAG_D + 2)
        assert (wx[:old] == 5.0).all()
        assert (wx[wx.index > old - pd.Timedelta(days=-2)] == 6.0).all()
        assert (X_future["wx_fcst_d2_temperature_2m"] == 6.0).all()
        assert not X_hist.index.duplicated().any()
        return super().predict(y, X_hist, X_future)


def test_actual_weather_history_mode():
    out = harness.run_backtest(ActualSpy(), make_frame(), "2025-01-20", "2025-01-25", verbose=False)
    assert len(out) == 6 * 24


def test_holiday_flag():
    idx = pd.date_range("2025-07-03", periods=72, freq="h", tz=harness.TZ_PJM).tz_convert("UTC")
    cal = harness.calendar_features(idx)
    local_day = idx.tz_convert(harness.TZ_PJM).day
    assert (cal["cal_holiday"][local_day == 4] == 1).all()
    assert (cal["cal_holiday"][local_day != 4] == 0).all()


class GapSpy(Spy):
    """future_from_cutoff: X_future starts the hour after the demand cutoff, holds only
    allowed columns, and the harness keeps only day D's 24 hours of the prediction."""
    future_from_cutoff = True

    def predict(self, y, X_hist, X_future):
        self._check(y, X_hist)
        assert X_future.index[0] == y.index[-1] + pd.Timedelta(hours=1)
        assert "operator_fcst" not in X_future.columns
        assert not any(c.startswith("wx_actual") for c in X_future.columns)
        return pd.DataFrame({"mean": 1.0}, index=X_future.index)


def test_future_from_cutoff():
    out = harness.run_backtest(GapSpy(), make_frame(), "2025-01-10", "2025-01-12", verbose=False)
    assert len(out) == 3 * 24
    assert out["lead_h"].between(14, 38).all()


def test_holiday_types():
    idx = pd.date_range("2025-11-26", "2026-01-02 23:00", freq="h", tz=harness.TZ_PJM).tz_convert("UTC")
    cal = harness.calendar_features(idx)
    day = idx.tz_convert(harness.TZ_PJM).strftime("%m-%d")
    by_day = cal.groupby(day)[["cal_hol_major", "cal_hol_minor", "cal_hol_near"]].max()
    assert by_day.loc["11-27"].tolist() == [1, 0, 0]  # Thanksgiving 2025
    assert by_day.loc["11-28"].tolist() == [0, 0, 1]  # Friday after
    assert by_day.loc["12-24"].tolist() == [0, 0, 1]
    assert by_day.loc["12-25"].tolist() == [1, 0, 0]
    assert by_day.loc["01-01"].tolist() == [1, 0, 0]
    assert by_day.loc["12-10"].tolist() == [0, 0, 0]
    idx = pd.date_range("2026-02-16", periods=24, freq="h", tz=harness.TZ_PJM).tz_convert("UTC")
    assert (harness.calendar_features(idx)["cal_hol_minor"] == 1).all()  # Presidents' Day
