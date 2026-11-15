import numpy as np
import pandas as pd

from ..harness import ALLOWED_WEATHER_PREFIX, DATA_LAG_H, ISSUE_HOUR_LOCAL, QCOLS, QUANTILES, TZ_PJM

TEMP = ALLOWED_WEATHER_PREFIX + "temperature_2m"


def day_ahead_features(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Features for target hours X.index, using only what was known at each day's issue.

    Every row is treated as part of a day-ahead forecast: its demand features come from
    before the cutoff (issue - DATA_LAG_H) of its own target day, so training rows and
    live rows are built the same way.
    """
    t = X.index
    local = t.tz_convert(TZ_PJM)
    day = local.tz_localize(None).normalize()
    cutoff = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR_LOCAL - DATA_LAG_H)
              ).tz_localize(TZ_PJM).tz_convert("UTC")
    # lookups only need the last 8 days; slicing first avoids re-indexing all history
    recent = y[t.min() - pd.Timedelta(days=8):]
    f = X.copy()
    f["lag48"] = recent.reindex(t - pd.Timedelta(hours=48)).values
    f["lag168"] = recent.reindex(t - pd.Timedelta(hours=168)).values
    # last hour seen before the cutoff, and the mean of the last full day before it
    f["last"] = recent.reindex(cutoff - pd.Timedelta(hours=1)).values
    daily = recent.groupby(recent.index.tz_convert(TZ_PJM).tz_localize(None).normalize()).mean()
    f["d2_mean"] = daily.reindex(day - pd.Timedelta(days=2)).values
    # daily weather shape for the target day (forecast made ~2 days ahead)
    g = X[TEMP].groupby(day)
    f["temp_day_mean"] = g.transform("mean").values
    f["temp_day_max"] = g.transform("max").values
    f["temp_day_min"] = g.transform("min").values
    f["hdd"] = np.maximum(18.0 - f[TEMP], 0)
    f["cdd"] = np.maximum(f[TEMP] - 18.0, 0)
    return f


class LightGBMQuantile:
    """Gradient-boosted trees, one model per quantile plus one for the mean.

    One model covers all 24 target hours (hour of day is a feature). Quantiles are
    sorted after prediction so they never cross.
    """

    def __init__(self, name="lgbm", hist_weather="forecast", refit_every_days=30,
                 train_years=None, **params):
        self.name, self.hist_weather = name, hist_weather
        self.refit_every_days, self.train_years = refit_every_days, train_years
        self.params = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                           min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, verbose=-1,
                           # one thread is faster at this data size (7 s vs 32 s per fit)
                           n_jobs=1) | params
        self.models = {}

    def fit(self, y, X):
        import lightgbm as lgb
        if self.train_years:
            X = X[X.index > X.index[-1] - pd.Timedelta(days=365 * self.train_years)]
        f = day_ahead_features(y, X)
        target = y.reindex(f.index)
        ok = target.notna() & f[TEMP].notna() & f["lag168"].notna()
        f, target = f[ok], target[ok]
        self.columns = list(f.columns)
        self.models = {"mean": lgb.LGBMRegressor(objective="l2", **self.params).fit(f, target)}
        if self.params.get("linear_tree"):
            # linear leaves are fit by least squares whatever the objective, so quantile
            # models come out broken (q05 below zero); keep the mean forecast only
            return
        for q, c in zip(QUANTILES, QCOLS):
            self.models[c] = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                               **self.params).fit(f, target)

    def predict(self, y, X_hist, X_future):
        f = day_ahead_features(y, X_future)[self.columns]
        out = pd.DataFrame({k: m.predict(f) for k, m in self.models.items()}, index=f.index)
        if len(self.models) > 1:
            out[QCOLS] = np.sort(out[QCOLS].values, axis=1)
        return out
