# grid-forecast

Probabilistic day-ahead electricity demand forecasting for **PJM** (the grid that serves Chicago through ComEd), benchmarked against PJM's own published day-ahead forecast.

## The question

1. **Day-ahead:** can a small, open pipeline match or beat the grid operator's own forecast, with calibrated uncertainty?
2. **Across horizons and model families:** where does each kind of model (ETS through TimesFM, zero-shot vs. trained) win, and why?

## Data

| Source | What | From | Key? |
|---|---|---|---|
| EIA-930 (EIA API v2) | PJM hourly demand + PJM's day-ahead forecast | Jul 2015 | Yes (free) |
| EIA-930 sub-BA | Hourly demand for PJM's subregions (CE = ComEd) | Jul 2018 | Yes |
| Open-Meteo Archive (ERA5) | Actual weather, 10 PJM cities | Jul 2015 | No |
| Open-Meteo Previous Runs | 1- and 2-day-ahead weather *forecasts* | Jan 2024 | No |

**Honest-weather rule:** backtests may only use weather the forecaster would have had at forecast time (the `forecast` files). Actual weather is used only for exploration and for an "oracle" upper bound, and is labeled that way.

## Setup (macOS)

```bash
brew install uv                # once, if you don't have it
cp .env.example .env           # then paste your EIA key into .env
uv sync
uv run python -m gridfc.fetch_eia       # ~5-15 min the first time
uv run python -m gridfc.fetch_weather   # ~5 min
uv run python -m gridfc.check_data      # raw coverage
uv run python -m gridfc.clean           # screen bad values -> data/clean/
uv run python -m gridfc.backtest        # models + PJM benchmark, same protocol
uv run python -m gridfc.report          # comparison table
uv run pytest
```

Data lands in `data/` (gitignored). Daily ensemble weather snapshots are saved by GitHub Actions to monthly Releases (`ensemble-YYYY-MM`).

## Results so far (targets Jan 2025 to Nov 2026, day-ahead, 10 am ET issue)

| Model | MAPE | Notes |
|---|---|---|
| PJM's own forecast (time-aligned) | 2.54% | The bar to beat |
| PJM's forecast as published in EIA-930 | 3.48% | Labeled 1 hour off |
| Weekly naive (with quantiles) | 8.58% | 80% interval covers 80.4% |

## Layout

```
src/gridfc/    fetchers, checks (models come next)
site/          static results site (later)
notebooks/     exploration
tests/
```
