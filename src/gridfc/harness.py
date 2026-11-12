"""Shared day-ahead backtest harness. Every model goes through this, so results compare.

Protocol (mirrors how PJM's day-ahead forecast is used):
  issue time   10:00 America/New_York on day D-1
  targets      the 24 hours of local day D (00:00-23:00 Eastern), stored in UTC
  demand seen  only hours ending at least DATA_LAG_H before the issue time
  weather seen only forecast columns made ~2 days ahead (wx_fcst_d2_*), plus calendar
               features. Actual weather and the operator's forecast are never given
               to models; the harness enforces this, not the model.

A model implements:
  name: str
  refit_every_days: int    # 0 = no fitting needed
  fit(y, X)                # y: demand history (hourly Series), X: allowed features
  predict(y, X_hist, X_future) -> DataFrame indexed by target hours with
                           column 'mean' and optionally quantile columns QCOLS
"""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR

TZ_PJM = "America/New_York"
ISSUE_HOUR_LOCAL = 10
DATA_LAG_H = 2
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QCOLS = [f"q{int(round(q * 100)):02d}" for q in QUANTILES]
ALLOWED_WEATHER_PREFIX = "wx_fcst_d2_"
CLEAN = DATA_DIR / "clean"
OUT = DATA_DIR / "backtests"


def load_frame() -> pd.DataFrame:
    pjm = pd.read_parquet(CLEAN / "pjm.parquet")
    wx = pd.read_parquet(CLEAN / "weather.parquet")
    return pjm.join(wx, how="left")


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    local = idx.tz_convert(TZ_PJM)
    return pd.DataFrame({
        "cal_hour": local.hour,
        "cal_dow": local.dayofweek,
        "cal_month": local.month,
        "cal_doy": local.dayofyear,
        "cal_weekend": (local.dayofweek >= 5).astype(int),
    }, index=idx)


def allowed_features(frame: pd.DataFrame) -> pd.DataFrame:
    wx = frame[[c for c in frame.columns if c.startswith(ALLOWED_WEATHER_PREFIX)]]
    assert not any(c.startswith("operator") for c in wx.columns)
    return wx.join(calendar_features(frame.index))


def issue_times(start: str, end: str) -> list[pd.Timestamp]:
    days = pd.date_range(start, end, freq="D", tz=TZ_PJM)
    return [(d - pd.Timedelta(days=1)).replace(hour=ISSUE_HOUR_LOCAL).tz_convert("UTC")
            for d in days]


def target_hours(issue: pd.Timestamp) -> pd.DatetimeIndex:
    day = (issue.tz_convert(TZ_PJM) + pd.Timedelta(days=1)).normalize()
    return pd.date_range(day, periods=24, freq="h").tz_convert("UTC")


def run_backtest(model, frame: pd.DataFrame, start: str, end: str,
                 verbose: bool = True) -> pd.DataFrame:
    y_all = frame["demand"]
    X_all = allowed_features(frame)
    rows, last_fit = [], None
    issues = issue_times(start, end)
    for i, issue in enumerate(issues):
        targets = target_hours(issue)
        if targets[-1] > y_all.index[-1]:
            break
        cutoff = issue - pd.Timedelta(hours=DATA_LAG_H)
        y_hist = y_all[:cutoff]
        X_hist = X_all[:cutoff]
        if model.refit_every_days and (
                last_fit is None or (issue - last_fit).days >= model.refit_every_days):
            model.fit(y_hist, X_hist)
            last_fit = issue
        pred = model.predict(y_hist, X_hist, X_all.reindex(targets))
        pred = pred.reindex(targets)
        pred["issue_time"] = issue
        pred["target_time"] = targets
        pred["lead_h"] = ((targets - issue) / pd.Timedelta(hours=1)).astype(int)
        pred["y"] = y_all.reindex(targets).values
        rows.append(pred.reset_index(drop=True))
        if verbose and i % 60 == 0:
            print(f"  {model.name}: {issue.date()} ({i + 1}/{len(issues)})", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "model", model.name)
    return out


def operator_benchmark(frame: pd.DataFrame, start: str, end: str,
                       column: str = "operator_fcst_aligned",
                       name: str = "pjm_operator") -> pd.DataFrame:
    """PJM's own day-ahead forecast on the same issue/target grid (point only).

    Default is the time-aligned series (see clean.py); 'operator_fcst' is as published.
    """
    rows = []
    for issue in issue_times(start, end):
        targets = target_hours(issue)
        if targets[-1] > frame.index[-1]:
            break
        rows.append(pd.DataFrame({
            "mean": frame[column].reindex(targets).values,
            "issue_time": issue, "target_time": targets,
            "lead_h": ((targets - issue) / pd.Timedelta(hours=1)).astype(int),
            "y": frame["demand"].reindex(targets).values,
        }))
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "model", name)
    return out
