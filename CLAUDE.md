# CLAUDE.md

Day-ahead electricity demand forecasting for PJM, benchmarked against PJM's own forecast. Read `LOG.md` first.

## Rules
- Never read, print, or commit `.env` or the EIA key. Code gets it through `gridfc.secrets.get_eia_key()`, and error messages go through `redact()`.
- Claude's sandboxes can't reach api.eia.gov or open-meteo.com. Kushal runs the fetch commands in his own terminal; Claude writes and tests the code (offline tests use fakes).
- Honest-weather rule: models only see `wx_fcst_d2_*` weather (made ~2 days ahead) plus calendar features. `wx_fcst_d1_*` can be issued after the 10 am issue time for late hours, so it's off by default. Actual weather (`wx_actual_*`) is for exploration or a clearly labeled oracle run. `harness.allowed_features()` enforces this; don't bypass it.
- Every model goes through `harness.run_backtest` (issue 10:00 ET on D-1, targets = 24 local hours of D, demand visible up to issue - 2h). Add models to `models/REGISTRY`.
- PJM's day-ahead forecast in EIA-930 is labeled 1 hour early; the benchmark uses `operator_fcst_aligned` (see clean.py).
- All timestamps are UTC in storage. Convert to Eastern (PJM) or Central (ComEd) only for display and calendar features.
- The operator forecast (EIA type `DF`) is a benchmark, never a model input.
- `data/` is gitignored. Don't commit data files.
- Write plainly in docs and the writeup. Kushal reads the LOG.

## Commands
| Command | What it does |
|---|---|
| `uv run python -m gridfc.fetch_eia` | Download PJM + subregion demand and PJM's day-ahead forecast |
| `uv run python -m gridfc.fetch_weather` | Download actual weather + 1/2-day-ahead weather forecasts |
| `uv run python -m gridfc.check_data` | Coverage, gaps, and PJM forecast error vs. weekly naive (raw data) |
| `uv run python -m gridfc.clean` | Screen bad values, build `data/clean/` (pjm, subba, weather, flags.csv) |
| `uv run python -m gridfc.backtest --model all` | Run models + operator benchmark through the harness |
| `uv run python -m gridfc.report [--by year/month/lead/hour]` | Compare all saved backtests on the same hours |
| `uv run python -m gridfc.collect_ensemble` | Snapshot ensemble weather (GitHub Actions runs this daily) |
| `uv run pytest` | Offline tests |

## When you finish meaningful work
Add a short dated entry to `LOG.md`.
