"""Draw the README figures from the saved backtests (data/backtests/*.parquet).

  uv run python scripts/readme_figures.py

Writes docs/img/by_month.png and docs/img/cold_snap.png. Scores only hours that every
model forecast and that had a weather forecast, the same rule as the reports.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from gridfc import harness  # noqa: E402
from gridfc.metrics import score_table  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"pjm_operator": "#2a78d6", "timesfm3": "#eda100", "lgbm": "#eb6834",
         "chronos2": "#1baf7a"}
LABEL = {"pjm_operator": "PJM's own forecast", "timesfm3": "TimesFM 3 (zero-shot, with weather)",
         "lgbm": "LightGBM", "chronos2": "Chronos-2 (zero-shot, with weather)"}


def load(models):
    res = pd.concat([pd.read_parquet(harness.OUT / f"{m}.parquet") for m in models])
    res = res.dropna(subset=["y", "mean"])
    frame = harness.load_frame()
    has_wx = frame["wx_fcst_d2_temperature_2m"].notna()
    res = res[has_wx.reindex(pd.DatetimeIndex(res["target_time"])).values]
    keys = res.groupby(["issue_time", "target_time"])["model"].nunique()
    common = keys[keys == len(models)].index
    res = res.set_index(["issue_time", "target_time"]).loc[common].reset_index()
    lt = pd.DatetimeIndex(res["target_time"]).tz_convert(harness.TZ_PJM)
    res["month"], res["day"] = lt.month, lt.tz_localize(None).normalize()
    return res


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.dpi": 160,
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    models = ["pjm_operator", "timesfm3", "chronos2", "lgbm"]
    res = load(models)

    mm = score_table(res, ["month", "model"])["MAPE_%"].unstack()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    show = ["pjm_operator", "timesfm3", "lgbm"]
    w = 0.27
    for i, m in enumerate(show):
        ax.bar(mm.index + (i - 1) * w, mm[m], width=w - 0.03, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                 "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(0, mm[show].max().max() * 1.3)
    ax.legend(ncols=3, loc="upper center", fontsize=8.5)
    ax.set_title(f"Day-ahead error by month, {res['day'].min():%b %Y} to {res['day'].max():%b %Y}",
                 fontsize=10, color=INK, loc="left")
    fig.savefig(OUT / "by_month.png")
    plt.close(fig)

    win = res[(res["day"] >= "2025-01-18") & (res["day"] <= "2025-01-24")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    base = win[win["model"] == "pjm_operator"].sort_values("target_time")
    t = pd.DatetimeIndex(base["target_time"]).tz_convert(harness.TZ_PJM).tz_localize(None)
    ax.plot(t, base["y"] / 1000, color=INK, lw=2, label="actual")
    for m in models:
        d = win[win["model"] == m].sort_values("target_time")
        ax.plot(t, d["mean"].values / 1000, color=COLOR[m], lw=1.3, label=LABEL[m])
    ax.set_ylabel("PJM load (GW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.legend(ncols=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / "cold_snap.png")
    plt.close(fig)
    print(f"wrote {OUT}/by_month.png, cold_snap.png")


if __name__ == "__main__":
    main()
