import logging

import numpy as np
import pandas as pd

from ..harness import QCOLS, QUANTILES, TZ_PJM
from .gbm import TEMP
from .linear import APP, BASE_C


def prophet_frame(X: pd.DataFrame) -> pd.DataFrame:
    local = X.index.tz_convert(TZ_PJM).tz_localize(None)
    return pd.DataFrame({
        "ds": local,
        "hdd": np.maximum(BASE_C - X[TEMP].values, 0),
        "cdd": np.maximum(X[APP].values - BASE_C, 0),
        "weekend": (X["cal_weekend"].values == 1),
    })


class ProphetModel:
    """Prophet: trend + yearly, weekly, and daily cycles + holidays + degree-hours.

    The daily cycle is fitted separately for weekdays and weekends. Prophet does not
    look at recent load; it forecasts from the calendar and weather alone.
    """

    def __init__(self, name="prophet", refit_every_days=30, samples=300):
        self.name, self.refit_every_days, self.samples = name, refit_every_days, samples

    def fit(self, y, X):
        from prophet import Prophet
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        X = X[X[TEMP].notna()]
        df = prophet_frame(X)
        df["y"] = y.reindex(X.index).values
        df = df.dropna(subset=["y"])
        self.fill = df[["hdd", "cdd"]].median()  # for weather missing on a forecast day
        df["weekday"] = ~df["weekend"]
        m = Prophet(daily_seasonality=False, uncertainty_samples=self.samples)
        m.add_seasonality("daily_weekday", period=1, fourier_order=10, condition_name="weekday")
        m.add_seasonality("daily_weekend", period=1, fourier_order=10, condition_name="weekend")
        m.add_country_holidays("US")
        m.add_regressor("hdd")
        m.add_regressor("cdd")
        self.model = m.fit(df)

    def predict(self, y, X_hist, X_future):
        df = prophet_frame(X_future)
        df["weekday"] = ~df["weekend"]
        df[["hdd", "cdd"]] = df[["hdd", "cdd"]].fillna(self.fill)
        mean = self.model.predict(df)["yhat"].values
        samples = self.model.predictive_samples(df)["yhat"]
        out = pd.DataFrame({"mean": mean}, index=X_future.index)
        out[QCOLS] = np.quantile(samples, QUANTILES, axis=1).T
        return out
