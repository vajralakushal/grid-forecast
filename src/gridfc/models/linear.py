import numpy as np
import pandas as pd

from ..harness import ALLOWED_WEATHER_PREFIX, QCOLS, QUANTILES, TZ_PJM
from .gbm import TEMP, day_ahead_features

APP = ALLOWED_WEATHER_PREFIX + "apparent_temperature"
WIND = ALLOWED_WEATHER_PREFIX + "wind_speed_10m"
SUN = ALLOWED_WEATHER_PREFIX + "shortwave_radiation"
BASE_C = 18.0


def linear_features(y: pd.Series, X: pd.DataFrame, X_hist: pd.DataFrame | None = None
                    ) -> pd.DataFrame:
    """Degree-hour, recent-load, and calendar features for target hours X.index.

    X_hist (forecast weather before X) is only used for the D-2 daily temperature,
    which explains the level of d2_mean.
    """
    f = day_ahead_features(y, X)
    local = X.index.tz_convert(TZ_PJM)
    day = local.tz_localize(None).normalize()
    temp = X[TEMP] if X_hist is None else pd.concat([X_hist[TEMP], X[TEMP]])
    temp = temp[~temp.index.duplicated(keep="last")]
    daily = temp.groupby(temp.index.tz_convert(TZ_PJM).tz_localize(None).normalize()).mean()
    t_d2 = daily.reindex(day - pd.Timedelta(days=2)).values
    out = pd.DataFrame(index=X.index)
    out["d2_mean"] = f["d2_mean"]
    out["lag168"] = f["lag168"]
    out["last"] = f["last"]
    out["hdd"] = np.maximum(BASE_C - X[TEMP], 0)
    out["cdd"] = np.maximum(X[APP] - BASE_C, 0)
    out["hdd_wind"] = out["hdd"] * X[WIND]
    out["hdd_day"] = np.maximum(BASE_C - f["temp_day_mean"], 0)
    out["cdd_day"] = np.maximum(f["temp_day_mean"] - BASE_C, 0)
    out["hdd_d2"] = np.maximum(BASE_C - t_d2, 0)
    out["cdd_d2"] = np.maximum(t_d2 - BASE_C, 0)
    out["sun"] = X[SUN]
    for d in range(6):
        out[f"dow{d}"] = (X["cal_dow"] == d).astype(float)
    out["holiday"] = X["cal_holiday"].astype(float)
    doy = 2 * np.pi * X["cal_doy"] / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(doy), np.cos(doy)
    out["hour"] = X["cal_hour"].values
    return out


class LinearTemperature:
    """Ridge regression, one per hour of day, on degree-hours, recent load, and calendar.

    Load rises roughly in a straight line as temperature moves away from about 18 °C,
    so the fitted slopes keep going past the coldest or hottest training day, which
    trees cannot do. Quantiles come from the past year of actual / fitted ratios for
    each hour.
    """

    def __init__(self, name="linear", alpha=1.0, refit_every_days=1):
        self.name, self.alpha, self.refit_every_days = name, alpha, refit_every_days
        self.models = {}

    def fit(self, y, X):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = X[X[TEMP].notna()]
        f = linear_features(y, X)
        target = y.reindex(f.index)
        ok = target.notna() & f.notna().all(axis=1)
        f, target = f[ok], target[ok]
        self.columns = [c for c in f.columns if c != "hour"]
        self.fill = f[self.columns].median()  # for inputs missing on a forecast day
        self.models, ratios = {}, {}
        for h, g in f.groupby("hour"):
            m = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
            m.fit(g[self.columns], target[g.index])
            self.models[h] = m
            recent = g.index > g.index[-1] - pd.Timedelta(days=365)
            ratios[h] = target[g.index][recent] / m.predict(g[self.columns][recent])
        self.ratio_q = pd.DataFrame({h: r.quantile(QUANTILES).values for h, r in ratios.items()},
                                    index=QCOLS).T

    def predict(self, y, X_hist, X_future):
        f = linear_features(y, X_future, X_hist[X_hist.index > X_future.index[0]
                                                - pd.Timedelta(days=4)])
        out = pd.DataFrame(index=f.index)
        out["mean"] = np.nan
        for h, g in f.groupby("hour"):
            out.loc[g.index, "mean"] = self.models[h].predict(g[self.columns].fillna(self.fill))
        r = self.ratio_q.reindex(f["hour"]).values
        for j, c in enumerate(QCOLS):
            out[c] = out["mean"].values * r[:, j]
        return out
