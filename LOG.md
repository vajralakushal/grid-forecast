# grid-forecast log

Newest at the bottom. Plain language.

## Nov 11, 2026: kickoff

- Question: can an open pipeline match or beat PJM's own day-ahead demand forecast with calibrated uncertainty, and which model families win at which horizons (ETS through TimesFM, zero-shot vs. trained)?
- Region: PJM, because it covers Chicago (ComEd subregion "CE") and has the most free data.
- Writeup plan: whitepaper built on an SR 11-7-style validation report, then a short blog post, then a Loom.
- Set up the repo: EIA-930 fetcher (PJM demand + PJM day-ahead forecast from 2015; 20 subregions from 2018), Open-Meteo fetcher (actual weather from 2015; 1- and 2-day-ahead weather forecasts from Jan 2024), a data check script, offline tests.
- Key handling: the EIA key lives only in `.env` (gitignored) on Kushal's Mac; later it becomes a GitHub Actions secret. Claude never sees it.
- Claude's sandboxes can't reach EIA or Open-Meteo, so Kushal runs the downloads.
- Deferred: Monte Carlo over weather ensembles (needs a daily collector, since Open-Meteo keeps ensemble members only ~3 days), the ComEd bill page, the static site.
- Next: Kushal runs the fetchers and check_data; then set up dbt models and the first round of models (baselines, ETS/MSTL, LightGBM, Prophet, TimesFM/Chronos zero-shot).

## Nov 11, 2026: data in, harness built, first findings

- Data downloaded on Kushal's Mac: PJM demand + PJM's day-ahead forecast and 20 subregions (2019 onward; EIA's API returned nothing earlier, though EIA's bulk CSVs go back to 2015 if ever needed), actual weather 2015+, 1/2-day-ahead weather forecasts 2024+.
- Data problems found (validation report material): an overflow value of 2,147,480,000 MWh (2^31) in Oct 2021, spikes 2-4x real load in 2019-2020, ~8,800 zero hours for the RECO subregion. `clean.py` flags and screens them; `data/clean/flags.csv` has the counts.
- Big finding: PJM's day-ahead forecast is labeled 1 hour early in EIA-930. Aligned, its MAPE is 2.54% (2025-26); as published it looks like 3.48% with a fake midnight error spike. The benchmark uses the aligned version (the harder bar).
- Even aligned, PJM's forecast runs 2-4% low overnight. Hypothesis to test: round-the-clock data-center load raising overnight demand. Not claimed yet.
- Built the shared backtest harness (issue 10 am ET day before; 24 local hours; harness controls what models see and tests check it), metrics (MAPE, pinball, 80/90% coverage), report, and a weekly-naive model with quantiles: 8.58% MAPE, intervals well calibrated (80.4% / 90.2%).
- Built the daily ensemble collector + GitHub Actions workflow (saves to monthly Releases, not git).
- Repo will be private on GitHub for now.
- Next: models into the harness (ETS/MSTL, LightGBM quantile, Prophet, TimesFM and Chronos zero-shot), then dbt for the cleaning layer.
- Nov 12: ensemble collector verified live from Kushal's Mac. Per city: 72 hours x 82 runs (ECMWF 51 members incl. control + NOAA GEFS 31) = 5,904 rows. All 10 cities: 0.22 MB/day, about 80 MB/year. Open-Meteo labels the models `ecmwf_ifs025_ensemble` and `ncep_gefs025`; the parser now reads names generically.
- Nov 12: first GitHub Actions run failed: Open-Meteo hung for 120 s from the GitHub runner, and the code didn't retry timeouts. Fixed: retries with backoff on timeouts/429/5xx, a failed city no longer kills the run, the workflow tries 3 times 5 minutes apart, and Python is pinned to 3.12 (`.python-version`).
