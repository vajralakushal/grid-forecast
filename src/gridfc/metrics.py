import numpy as np
import pandas as pd

from .harness import QCOLS, QUANTILES


def score(df: pd.DataFrame) -> pd.Series:
    d = df[df["y"].notna() & df["mean"].notna()]
    err = d["y"] - d["mean"]
    out = {
        "n_hours": len(d),
        "MAPE_%": (err.abs() / d["y"]).mean() * 100,
        "MAE_MWh": err.abs().mean(),
        "RMSE_MWh": np.sqrt((err ** 2).mean()),
        "bias_%": (err / d["y"]).mean() * 100,  # + means forecast too low
    }
    def has(*cols):
        return all(c in d and d[c].notna().any() for c in cols)

    # each score needs only its own quantiles (TimesFM has no q05/q95)
    if has(*QCOLS):
        pin = []
        for q, c in zip(QUANTILES, QCOLS):
            u = d["y"] - d[c]
            pin.append(np.maximum(q * u, (q - 1) * u).mean())
        out["pinball_%"] = np.mean(pin) / d["y"].mean() * 100
    if has("q10", "q90"):
        out["cover80_%"] = ((d["y"] >= d["q10"]) & (d["y"] <= d["q90"])).mean() * 100
    if has("q05", "q95"):
        out["cover90_%"] = ((d["y"] >= d["q05"]) & (d["y"] <= d["q95"])).mean() * 100
    return pd.Series(out)


def score_table(df: pd.DataFrame, keys) -> pd.DataFrame:
    """Score per group as a flat table (models without quantiles get blank columns)."""
    rows = {k: score(g) for k, g in df.groupby(keys)}
    return pd.DataFrame(rows).T.rename_axis(keys)
