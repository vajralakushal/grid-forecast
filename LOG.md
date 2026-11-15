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
- Nov 12: the GitHub Actions ensemble collector succeeded (4.5 min, so some retries were probably needed). It runs daily at 13:17 UTC and saves to the `ensemble-YYYY-MM` release. The data side is done; next is the modeling phase in a new thread.
- Nov 12: the successful run needed the 5th (last) attempt for 2 cities: Open-Meteo is often slow from GitHub's runners. Added a second pass that retries any failed city after 60 s, plus a test for it.

## Nov 12, 2026: modeling phase starts, first LightGBM

- Harness: models can set `hist_weather = "actual"` to train on observed weather (shown under the d2 forecast column names) for hours more than 7 days before issue, with the d2 forecast after that. Targets always get the d2 forecast. This lets models train on 2019-2023, before weather forecasts exist (Jan 2024). Tests check it. Added a US federal holiday flag to the calendar features.
- LightGBM (mean + 7 quantiles, refit every 30 days, lags/levels built as of each day's cutoff) run both ways, 2025-01 to 2026-11, same hours as everything else:
  - `lgbm` (forecast weather, trains from 2024): MAPE 2.94%. PJM is 2.54%. Beats PJM in July (2.55 vs 3.02) and August (2.97 vs 3.08); loses most in winter (Jan 3.90 vs 2.40).
  - `lgbm_wxact` (actual weather back to 2019): MAPE 3.45%, forecasts 2.7% too low on average. More history hurt: trees can't extrapolate, so old lower load levels drag it down (load growth). Fixing that may change this result.
  - Intervals are much too narrow: 55% coverage for the 80% interval, 71% for the 90% interval (weekly naive: 80/90). Quantile trees fit on their own training data are overconfident.
- Runtime: ~35 min for both run side by side. Too slow, mostly thread overhead (8 models x 24 rows/day, two processes fighting over cores), not training.
- Next: speed fix; predict a ratio to a recent load level instead of raw MWh (for growth); calibrate intervals on out-of-sample errors (conformal); then rerun both ways.
- Nov 12: full report. `reports/` is gitignored (source, figures, PDFs). `uv run python reports/build.py` recomputes every number from data/backtests and writes `reports/out/report_YYYY-MM-DD.pdf`; one living report, dated copies. Needs xelatex (MacTeX) and the matplotlib dev dependency.
- Nov 12: winter diagnosis. On days forecast below -5 C, `lgbm` 3.95% MAPE vs PJM 2.34% (33 days). Its first fits had 1 training day that cold: forecast weather starts Jan 20, 2024, at the tail of a Jan 14-21 cold spell. It under-forecast the Jan 20-22, 2025 cold snap by up to 12.6% (actual peak 144.4 GW, lgbm 122.3, PJM 139.3). `lgbm_wxact` (33 cold training days) scored 3.51% on those days. Other bad winter days are holidays (Dec 24, Jan 1, Presidents' Day): one holiday flag is too coarse.
- Nov 12: speed. Full run of both variants went from ~35 min to 15 min (lgbm 592 s, lgbm_wxact 896 s, in parallel). Fixes: LightGBM n_jobs=1 (fit 31.8 s -> 7.2 s), harness matches timestamp units (pandas 3 makes us, data is ns; each lookup was 152 ms, now 0.1 ms), lag lookups on a recent slice. Forecasts identical to the first run. Training is now most of the time.
- Nov 12: added a user-level `technical-writing` skill (HumanLayer prose guide + no-LLMism rules + number rules) for all writing.

## Nov 15, 2026: six more model families

- New models, all through the harness: `linear` (ridge per hour on degree-hours), `lgbm_linear` (linear leaves), `mstl`, `prophet`, and two pretrained models used zero-shot, Chronos-2 and TimesFM 3.0 (each with and without weather inputs). Harness option `future_from_cutoff` gives sequence models d2 weather for the 15 hours between the 08:00 cutoff and day D (all issued before 10:00, so honest). Metrics score 80% coverage even when q05/q95 are missing (TimesFM has deciles only). `backtest --no-benchmark` for parallel runs.
- Results, 16,207 hours, 2025-01-01 to 2026-11-10 (hours with a weather forecast):
  - `timesfm3` 2.47% MAPE vs PJM 2.54%: first model to beat PJM overall, without training on PJM data. 80% coverage 82.4%. After its Aug 2026 release (Sep 1-Nov 10, 1,677 h): 1.89% vs PJM 2.26%, so not explained by leakage, but that's 70 autumn days.
  - `chronos2` 3.08%, `lgbm_linear` 2.93%, `lgbm` 2.94%, `linear` 3.25%, `prophet` 5.87%, `mstl` 6.14%. Without weather the pretrained models score 4.82-4.89%.
  - Winter: PJM 2.38%, `timesfm3` 2.75%, `chronos2` 2.83%, `lgbm` 3.54%. Days below -5 C: PJM 2.34%, `chronos2` 2.50%, `timesfm3` 2.81%, `linear` 3.98%. Extrapolating in temperature (`linear`, `lgbm_linear`) did not fix cold days; the pretrained models did. Jan 20-22, 2025 peak 144.4 GW: `timesfm3` 142.6, PJM 139.3, `lgbm` 122.3.
  - By hour: `timesfm3` wins overnight (1.8% vs PJM up to 3.9%), PJM wins every hour 11:00-23:00.
- `lgbm_linear` quantiles were broken (q05 down to -157,766 MWh; linear leaves are least squares regardless of objective), so it is mean-only now.
- Run time: 8 backtests in parallel in ~18 min; `timesfm3` 818 s, `chronos2` 520 s (CPU, 2 threads each).
- Part 2 report: `uv run python reports/build_models.py` -> `reports/out/models_YYYY-MM-DD.pdf`.
- Next: live daily forecasts this winter (timesfm3, chronos2, lgbm, a fixed average) saved before actuals arrive, as the clean test; better covariates for the pretrained models; drop Prophet and MSTL.

## Nov 15, 2026 (later): is the gap real, peaks, CI, public repo

- Significance (daily MAPE difference vs PJM, Diebold-Mariano with 7-day Newey-West, 7-day block bootstrap): `timesfm3` minus PJM = -0.07 points/day, 95% CI -0.26 to +0.10, p = 0.44. A tie overall. Summer (Jun-Aug) -0.62, p = 0.001; winter (Dec-Feb) +0.37, p = 0.051. `chronos2`, `lgbm`, `linear` are significantly worse than PJM.
- Daily peaks: PJM 1.87% peak-size error, `timesfm3` 2.74%, `lgbm` 3.28%. Peak hour exactly right: `timesfm3` 63.3% of days, PJM 63.0%; within 1 hour: PJM 88.9%, `timesfm3` 84.4%. PJM as published gets the peak hour right on 12.5% of days, more evidence of the 1-hour label shift.
- Harness: holiday types (`cal_hol_major`, `cal_hol_minor`, `cal_hol_near` for Dec 24, Dec 26-31, Friday after Thanksgiving). Experiment running: pretrained models with all 6 weather variables + holiday types (`timesfm3_full`, `chronos2_full`).
- Weather fetch now also asks for humidity, precipitation, snowfall, sea-level pressure (needs `fetch_weather --only forecast --refresh-all`). Open-Meteo docs say GFS 2 m temperature forecasts go back to March 2021: a way to get honest forecast weather for more cold snaps (Winter Storm Elliott, Dec 2022).
- Model libraries moved to a `models` dependency group; the ensemble collector installs only the core (`uv sync --no-default-groups`). Linux gets CPU-only torch (no CUDA downloads in Actions).
- CI (`.github/workflows/ci.yml`) runs the offline tests on every push. README rewritten with results; `scripts/readme_figures.py` draws its figures.
- Repo made public. Data stays EIA + Open-Meteo so anyone can reproduce results.
