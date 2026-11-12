"""Compare every saved backtest on the same hours.

  uv run python -m gridfc.report            # overall + by year
  uv run python -m gridfc.report --by lead  # by hours ahead
"""
import argparse

import pandas as pd

from .harness import OUT, TZ_PJM
from .metrics import score_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["year", "month", "lead", "hour"], default="year")
    args = ap.parse_args()

    res = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("*.parquet"))])
    # compare only hours every model forecast (fair comparison)
    ok = res.dropna(subset=["y", "mean"])
    keys = ok.groupby(["issue_time", "target_time"])["model"].nunique()
    common = keys[keys == ok["model"].nunique()].index
    res = ok.set_index(["issue_time", "target_time"]).loc[common].reset_index()
    pd.set_option("display.width", 160)

    print("== Overall (same hours for every model) ==")
    print(score_table(res, "model").round(2).to_string())

    local = pd.DatetimeIndex(res["target_time"]).tz_convert(TZ_PJM)
    res["year"], res["month"], res["hour"] = local.year, local.month, local.hour
    res["lead"] = res["lead_h"]
    print(f"\n== MAPE % by {args.by} ==")
    t = score_table(res, [args.by, "model"])["MAPE_%"]
    print(t.unstack().round(2).to_string())


if __name__ == "__main__":
    main()
