import numpy as np
import pandas as pd

from ..harness import QCOLS

# statsforecast returns symmetric intervals; this maps them to our quantile columns
LEVEL_OF = {"q05": ("lo", 90), "q10": ("lo", 80), "q25": ("lo", 50),
            "q75": ("hi", 50), "q90": ("hi", 80), "q95": ("hi", 90)}


class MSTLModel:
    """MSTL: split recent load into a daily cycle, a weekly cycle, and a trend.

    The cycles repeat their last values; an ETS model forecasts the trend. Uses no
    weather. Refits on the last `context_days` every day, inside predict.
    """

    refit_every_days = 0
    future_from_cutoff = True

    def __init__(self, name="mstl", context_days=56):
        self.name, self.context_days = name, context_days

    def fit(self, y, X):
        pass

    def predict(self, y, X_hist, X_future):
        from statsforecast.models import MSTL, AutoETS
        hist = y.iloc[-self.context_days * 24:].interpolate(limit_direction="both")
        h = len(X_future)
        m = MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"))
        fc = m.forecast(y=hist.values.astype(float), h=h, level=[50, 80, 90])
        out = pd.DataFrame({"mean": fc["mean"]}, index=X_future.index)
        for c in QCOLS:
            out[c] = fc["mean"] if c == "q50" else fc["{}-{}".format(*LEVEL_OF[c])]
        out[QCOLS] = np.sort(out[QCOLS].values, axis=1)
        return out
