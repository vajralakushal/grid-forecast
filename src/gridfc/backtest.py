"""Run one or more models through the shared harness and save results.

  uv run python -m gridfc.backtest --model weekly_naive
  uv run python -m gridfc.backtest --model all --start 2025-01-01

Each model writes data/backtests/<name>.parquet, so models can run in parallel
(separate terminals or machines) and be compared later with gridfc.report.
"""
import argparse

from .harness import OUT, load_frame, operator_benchmark, run_backtest
from .models import REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help=f"one of {list(REGISTRY)} or 'all'")
    ap.add_argument("--start", default="2025-01-01", help="first target day (Eastern)")
    ap.add_argument("--end", default="2100-01-01", help="last target day (Eastern)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    end = min(args.end, str(frame.index[-1].date()))

    operator_benchmark(frame, args.start, end).to_parquet(OUT / "pjm_operator.parquet", index=False)
    operator_benchmark(frame, args.start, end, column="operator_fcst",
                       name="pjm_operator_as_published").to_parquet(
        OUT / "pjm_operator_as_published.parquet", index=False)

    names = list(REGISTRY) if args.model == "all" else [args.model]
    for name in names:
        print(f"backtesting {name} ...", flush=True)
        res = run_backtest(REGISTRY[name](), frame, args.start, end)
        res.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"  wrote {len(res):,} rows")


if __name__ == "__main__":
    main()
