"""Screen EIA-930 demand for bad values and build model-ready tables.

Rules (each flagged value is set to missing, then short gaps are filled):
  nonpositive  value <= 0
  overflow     value above a physical cap (PJM: 200 GW; subregion: 3x its 99.9th percentile)
  spike        more than 40% away from the centered 24-hour rolling median
Gaps of 3 hours or less are filled by linear interpolation (flag gets '+filled').
Longer gaps stay missing.

Weather: the 10 city series are combined into one PJM series, weighted by each
city's subregion median demand. Both kinds are kept, with names that make the
difference obvious:
  wx_actual_*   real weather (exploration / oracle only)
  wx_fcst_d1_*  forecast made about 1 day before the hour
  wx_fcst_d2_*  forecast made about 2 days before the hour (the safe default for
                day-ahead backtests; see CLAUDE.md)

Outputs in data/clean/: pjm.parquet, subba.parquet, weather.parquet, flags.csv
Run:  uv run python -m gridfc.clean
"""
import duckdb
import pandas as pd

from .config import DATA_DIR, RAW_DIR, WEATHER_POINTS

CLEAN = DATA_DIR / "clean"
PJM_CAP_MWH = 200_000
SPIKE_TOL = 0.40
MAX_FILL = 3
OPERATOR_SHIFT_H = 1  # see build_pjm


def connect():
    con = duckdb.connect()
    con.sql("set TimeZone = 'UTC'")
    return con


def screen(s: pd.Series, cap: float) -> tuple[pd.Series, pd.Series]:
    """Return (cleaned series, flag series) for one hourly series on a full hourly index."""
    flag = pd.Series("ok", index=s.index, dtype="object")
    flag[s.isna()] = "missing"
    bad = s <= 0
    flag[bad] = "nonpositive"
    over = s > cap
    flag[over] = "overflow"
    x = s.mask(bad | over)
    med = x.rolling(25, center=True, min_periods=12).median()
    spike = (x / med - 1).abs() > SPIKE_TOL
    flag[spike] = "spike"
    x = x.mask(spike)
    filled = x.interpolate(limit=MAX_FILL, limit_area="inside")
    was_filled = x.isna() & filled.notna()
    flag[was_filled] = flag[was_filled] + "+filled"  # keep the reason, e.g. "overflow+filled"
    return filled, flag


def full_hours(idx):
    return pd.date_range(idx.min(), idx.max(), freq="h")


def build_pjm(con):
    df = con.sql(f"""select period, type, value
                     from read_parquet('{RAW_DIR}/eia930/pjm_region/*.parquet')""").df()
    w = df.pivot_table(index="period", columns="type", values="value", aggfunc="last")
    w.index = pd.DatetimeIndex(w.index).tz_convert("UTC")
    w = w.reindex(full_hours(w.index))
    out = pd.DataFrame(index=w.index)
    out["demand"], out["demand_flag"] = screen(w["D"], PJM_CAP_MWH)
    out["operator_fcst"], out["operator_fcst_flag"] = screen(w["DF"], PJM_CAP_MWH)
    # In EIA-930, PJM's day-ahead forecast is labeled one hour early relative to
    # demand: shifting it +1h cuts its 2025 MAPE from 3.5% to 2.5% and removes a
    # fake midnight error spike. We benchmark against the aligned (harder) version.
    out["operator_fcst_aligned"] = out["operator_fcst"].shift(OPERATOR_SHIFT_H)
    out.index.name = "period"
    return out


def build_subba(con):
    df = con.sql(f"""select period, subba, value
                     from read_parquet('{RAW_DIR}/eia930/pjm_subba/*.parquet')""").df()
    df["period"] = pd.DatetimeIndex(df["period"]).tz_convert("UTC")
    parts = []
    for sb, g in df.groupby("subba"):
        s = g.set_index("period")["value"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(full_hours(s.index))
        cap = 3 * s[s > 0].quantile(0.999) if (s > 0).any() else 1.0
        v, f = screen(s, cap)
        parts.append(pd.DataFrame({"period": s.index, "subba": sb, "demand": v.values,
                                   "flag": f.values}))
    return pd.concat(parts, ignore_index=True)


def build_weather(con, subba):
    weights = (subba[subba["subba"].isin(WEATHER_POINTS)]
               .groupby("subba")["demand"].median())
    weights = weights / weights.sum()
    frames = {}
    for kind in ["actual", "forecast"]:
        df = con.sql(f"select * from read_parquet('{RAW_DIR}/weather/{kind}/*.parquet')").df()
        df["period"] = pd.DatetimeIndex(df["period"]).tz_convert("UTC")
        df["w"] = df["subba"].map(weights)
        cols = [c for c in df.columns if c not in ("subba", "city", "period", "w")]
        num = df[cols].mul(df["w"], axis=0).groupby(df["period"]).sum(min_count=1)
        den = df[cols].notna().mul(df["w"], axis=0).groupby(df["period"]).sum()
        agg = (num / den).where(den > 0.99)  # need (nearly) all cities for that hour
        if kind == "actual":
            agg.columns = [f"wx_actual_{c}" for c in agg.columns]
        else:
            agg.columns = [f"wx_fcst_d{c[-1]}_{c.rsplit('_previous_day', 1)[0]}"
                           for c in agg.columns]
        frames[kind] = agg
    out = frames["actual"].join(frames["forecast"], how="outer")
    out.index.name = "period"
    return out, weights


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    con = connect()
    pjm = build_pjm(con)
    subba = build_subba(con)
    wx, weights = build_weather(con, subba)
    pjm.to_parquet(CLEAN / "pjm.parquet")
    subba.to_parquet(CLEAN / "subba.parquet", index=False)
    wx.to_parquet(CLEAN / "weather.parquet")

    flags = pd.concat([
        pjm["demand_flag"].value_counts().rename("pjm_demand"),
        pjm["operator_fcst_flag"].value_counts().rename("pjm_operator_fcst"),
        subba.groupby("flag").size().rename("subregions"),
    ], axis=1).fillna(0).astype(int)
    flags.to_csv(CLEAN / "flags.csv")
    print("== Flags (hours) ==")
    print(flags.to_string())
    print("\n== Weather weights (share of subregion median demand) ==")
    print(weights.round(3).to_string())
    print(f"\nWrote {CLEAN}/pjm.parquet ({len(pjm):,} h), subba.parquet, weather.parquet "
          f"({wx.shape[1]} columns)")


if __name__ == "__main__":
    main()
