"""First look at the data: coverage, gaps, and the bar to beat.

Prints:
1. Coverage of PJM demand (D) and PJM's day-ahead forecast (DF): hours, gaps, bad values.
2. Day-ahead error by year for PJM's own forecast vs. a weekly-naive baseline
   (same hour one week earlier). The weekly lag is used because a day-ahead forecast
   is made before most of today's hours are known, so "same hour yesterday" would cheat.
3. Subregions available (CE = ComEd/Chicago) and their coverage.

Run:  uv run python -m gridfc.check_data
"""
import duckdb
import pandas as pd

from .config import RAW_DIR

EIA = RAW_DIR / "eia930"


def connect():
    con = duckdb.connect()
    con.sql("set TimeZone = 'UTC'")  # otherwise DuckDB returns times in the computer's local zone
    return con


def load_region():
    con = connect()
    df = con.sql(f"""
        select period, type, value
        from read_parquet('{EIA}/pjm_region/*.parquet')
    """).df()
    wide = df.pivot_table(index="period", columns="type", values="value", aggfunc="last")
    wide.index = pd.DatetimeIndex(wide.index).tz_convert("UTC")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(full)


def mape(actual, pred):
    ok = actual.notna() & pred.notna() & (actual > 0)
    return (abs(actual[ok] - pred[ok]) / actual[ok]).mean() * 100, int(ok.sum())


def main():
    w = load_region()
    print("== PJM coverage ==")
    print(f"range: {w.index.min()} to {w.index.max()}  ({len(w):,} hours)")
    for col in ["D", "DF"]:
        s = w[col]
        print(f"{col}: missing {s.isna().sum():,} hours, <=0 values {(s <= 0).sum():,}, "
              f"max {s.max():,.0f} MWh")

    w["naive_168"] = w["D"].shift(168)
    print("\n== Day-ahead MAPE by year (%) ==")
    print(f"{'year':<6}{'PJM forecast':>14}{'weekly naive':>14}{'hours':>8}")
    for year, g in w.groupby(w.index.year):
        m_op, n = mape(g["D"], g["DF"])
        m_nv, _ = mape(g["D"], g["naive_168"])
        print(f"{year:<6}{m_op:>14.2f}{m_nv:>14.2f}{n:>8,}")

    sub_path = EIA / "pjm_subba"
    if sub_path.exists():
        con = connect()
        print("\n== PJM subregions ==")
        print(con.sql(f"""
            select subba, any_value("subba-name") as name, count(*) as hours,
                   min(period) as first, max(period) as last,
                   round(avg(value)) as avg_mwh
            from read_parquet('{sub_path}/*.parquet')
            group by subba order by avg_mwh desc
        """).df().to_string(index=False))


if __name__ == "__main__":
    main()
