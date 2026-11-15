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


def daily_ape(df: pd.DataFrame, tz: str) -> pd.Series:
    """Mean absolute percentage error per local day (one value per issue)."""
    d = df[df["y"].notna() & df["mean"].notna()]
    day = pd.DatetimeIndex(d["target_time"]).tz_convert(tz).tz_localize(None).normalize()
    return ((d["y"] - d["mean"]).abs() / d["y"] * 100).groupby(day).mean()


def dm_test(diff: pd.Series, lag: int = 7) -> dict:
    """Diebold-Mariano test on a daily loss difference (model A minus model B).

    Hours within a day share one issue time and one weather forecast, so their errors
    move together; testing on daily means avoids counting 24 correlated hours as 24
    independent ones. The variance uses Newey-West weights out to `lag` days, because
    a bad weather week makes neighboring days miss together too.
    """
    from scipy import stats
    d = diff.dropna().values
    n, mean = len(d), d.mean()
    u = d - mean
    var = u @ u / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    t = mean / np.sqrt(var / n)
    return {"n_days": n, "mean_diff": mean, "t": t, "p_two_sided": 2 * stats.t.sf(abs(t), n - 1)}


def block_bootstrap_ci(diff: pd.Series, block: int = 7, reps: int = 5000, seed: int = 0,
                       level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the mean daily loss difference by moving-block bootstrap.

    Resamples runs of `block` consecutive days, which keeps week-long weather spells
    together instead of pretending each day is independent.
    """
    d = diff.dropna().values
    n = len(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(reps, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    means = d[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def peak_scores(df: pd.DataFrame, tz: str) -> pd.Series:
    """Daily peak: size error (% of actual peak) and timing error (hours)."""
    d = df[df["y"].notna() & df["mean"].notna()].copy()
    lt = pd.DatetimeIndex(d["target_time"]).tz_convert(tz)
    d["day"], d["hour"] = lt.tz_localize(None).normalize(), lt.hour
    act = d.loc[d.groupby("day")["y"].idxmax(), ["day", "y", "hour"]].set_index("day")
    fc = d.loc[d.groupby("day")["mean"].idxmax(), ["day", "mean", "hour"]].set_index("day")
    err = (act["y"] - fc["mean"]) / act["y"] * 100
    hours_off = (act["hour"] - fc["hour"]).abs()
    return pd.Series({
        "n_days": len(act),
        "peak_MAPE_%": err.abs().mean(),
        "peak_bias_%": err.mean(),  # + means the forecast peak is too low
        "peak_hour_MAE_h": hours_off.mean(),
        "peak_hour_exact_%": (hours_off == 0).mean() * 100,
        "peak_hour_within1_%": (hours_off <= 1).mean() * 100,
    })


def stationary_bootstrap_index(n: int, reps: int, block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: runs of random length (mean `block`), wrapping."""
    idx = np.empty((reps, n), dtype=int)
    idx[:, 0] = rng.integers(0, n, reps)
    new_run = rng.random((reps, n)) < 1 / block
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_run[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def spa_test(gain: pd.DataFrame, block: float = 7, reps: int = 5000, seed: int = 0) -> dict:
    """Hansen's (2005) test for superior predictive ability.

    gain: one row per day, one column per candidate model, value = benchmark loss minus
    model loss (positive = the model beat the benchmark that day). Null hypothesis: no
    candidate is better than the benchmark. Unlike a test on the single best model, it
    accounts for having searched over all the candidates, so picking the winner after
    the fact does not make the result look stronger than it is.

    Returns the consistent p-value plus Hansen's lower and upper bounds.
    """
    d = gain.dropna().values
    n, k = d.shape
    dbar = d.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = d[stationary_bootstrap_index(n, reps, block, rng)].mean(axis=1)  # reps x k
    omega = np.sqrt(n * boot.var(axis=0))
    t_obs = max((np.sqrt(n) * dbar / omega).max(), 0)
    # The null puts every model's true gain at zero (upper: the conservative "reality
    # check"). Models clearly worse than the benchmark are shifted down by their own
    # loss so they can't inflate the null (consistent); lower shifts every losing model.
    clearly_worse = np.sqrt(n) * dbar / omega < -np.sqrt(2 * np.log(np.log(n)))
    shifts = {"consistent": np.where(clearly_worse, dbar, 0), "lower": np.minimum(dbar, 0),
              "upper": np.zeros(k)}
    out = {"n_days": n, "n_models": k, "t_stat": t_obs,
           "best_model": gain.columns[int(np.argmax(dbar / omega))]}
    for name, mu in shifts.items():
        z = np.sqrt(n) * (boot - dbar + mu) / omega
        t_boot = np.maximum(z.max(axis=1), 0)
        out[f"p_{name}"] = float((t_boot >= t_obs).mean())
    return out
