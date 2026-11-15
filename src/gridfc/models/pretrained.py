"""Pretrained time-series models used as-is (zero-shot): no training on our data.

Each day they read the recent load (and optionally weather) and forecast from the hour
after the demand cutoff through the end of day D. Weights download from Hugging Face
on first use.
"""
import numpy as np
import pandas as pd

from ..harness import ALLOWED_WEATHER_PREFIX, QCOLS, QUANTILES

COVARIATES = [ALLOWED_WEATHER_PREFIX + "temperature_2m",
              ALLOWED_WEATHER_PREFIX + "apparent_temperature", "cal_holiday"]


def context(y, X_hist, X_future, hours, covariates):
    """Recent load and covariates with gaps filled, plus covariates for X_future."""
    hist = y.iloc[-hours:].interpolate(limit_direction="both")
    past = X_hist.reindex(hist.index)[covariates].interpolate(limit_direction="both")
    future = X_future[covariates].interpolate(limit_direction="both")
    return hist, past, future


class Chronos2:
    """Amazon's Chronos-2 (120M parameters, Oct 2025). Outputs our 7 quantiles directly."""

    refit_every_days = 0
    future_from_cutoff = True

    def __init__(self, name="chronos2", covariates=COVARIATES, context_hours=8192, threads=2):
        self.name, self.covariates = name, list(covariates)
        self.context_hours, self.threads = context_hours, threads
        self.pipe = None

    def fit(self, y, X):
        pass

    def predict(self, y, X_hist, X_future):
        if self.pipe is None:
            import torch
            from chronos import BaseChronosPipeline
            torch.set_num_threads(self.threads)
            self.pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        hist, past, future = context(y, X_hist, X_future, self.context_hours, self.covariates)
        inp = {"target": hist.values.astype("float32")}
        if self.covariates:
            inp["past_covariates"] = {c: past[c].values.astype("float32") for c in self.covariates}
            inp["future_covariates"] = {c: future[c].values.astype("float32")
                                        for c in self.covariates}
        q, mean = self.pipe.predict_quantiles([inp], prediction_length=len(X_future),
                                              quantile_levels=QUANTILES)
        out = pd.DataFrame(q[0][0].numpy(), index=X_future.index, columns=QCOLS)
        out.insert(0, "mean", mean[0][0].numpy())
        out[QCOLS] = np.sort(out[QCOLS].values, axis=1)
        return out


class TimesFM3:
    """Google's TimesFM 3.0 (Aug 2026). Outputs deciles 0.1-0.9, so q05/q95 stay empty
    and q25/q75 are the midpoints of the 0.2/0.3 and 0.7/0.8 deciles. Weights are
    licensed for non-commercial use only."""

    refit_every_days = 0
    future_from_cutoff = True

    def __init__(self, name="timesfm3", covariates=COVARIATES, context_hours=8192, threads=2):
        self.name, self.covariates = name, list(covariates)
        self.context_hours, self.threads = context_hours, threads
        self.model = None

    def fit(self, y, X):
        pass

    def predict(self, y, X_hist, X_future):
        if self.model is None:
            import torch
            from timesfm3 import TimesFM3Forecaster
            torch.set_num_threads(self.threads)
            self.model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch",
                                                            device="cpu")
        hist, past, future = context(y, X_hist, X_future, self.context_hours, self.covariates)
        kw = {}
        if self.covariates:
            kw["past_future_covariates"] = np.vstack(
                [np.concatenate([past[c].values, future[c].values]) for c in self.covariates]
            ).astype("float32")
        o = self.model.predict(hist.values.astype("float32"), horizon=len(X_future),
                               return_quantiles=True, **kw)
        d = np.asarray(o.quantiles)  # columns: deciles 0.1 ... 0.9
        out = pd.DataFrame({"mean": np.asarray(o.forecast), "q05": np.nan, "q10": d[:, 0],
                            "q25": (d[:, 1] + d[:, 2]) / 2, "q50": d[:, 4],
                            "q75": (d[:, 6] + d[:, 7]) / 2, "q90": d[:, 8], "q95": np.nan},
                           index=X_future.index)
        return out
