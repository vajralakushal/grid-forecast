import numpy as np
import pandas as pd

from ..harness import QCOLS, QUANTILES, TZ_PJM


class SeasonalNaive:
    """Same hour one week earlier, with uncertainty from its own past errors.

    Quantiles: for each local hour of day, take the last `window_days` of ratios
    actual / naive and use their empirical quantiles. Simple, honest, and a real
    probabilistic baseline (not just a point forecast).
    """

    def __init__(self, lag_h=168, window_days=365, name="seasonal_naive"):
        self.lag_h, self.window_days, self.name = lag_h, window_days, name
        self.refit_every_days = 30
        self.ratio_q = None

    def fit(self, y, X):
        y = y.iloc[-(self.window_days + 7) * 24:]
        ratio = (y / y.shift(self.lag_h)).replace([np.inf, -np.inf], np.nan).dropna()
        hours = ratio.index.tz_convert(TZ_PJM).hour
        self.ratio_q = ratio.groupby(hours).quantile(QUANTILES).unstack()
        self.ratio_q.columns = QCOLS

    def predict(self, y, X_hist, X_future):
        targets = X_future.index
        point = y.reindex(targets - pd.Timedelta(hours=self.lag_h)).values
        out = pd.DataFrame({"mean": point}, index=targets)
        hours = targets.tz_convert(TZ_PJM).hour
        r = self.ratio_q.reindex(hours).values
        for j, c in enumerate(QCOLS):
            out[c] = point * r[:, j]
        return out
