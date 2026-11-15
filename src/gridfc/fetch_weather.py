"""Download hourly weather for one city per large PJM subregion from Open-Meteo (no key).

Two kinds of weather, kept separate on purpose:

actual   - what the weather really was (ERA5 reanalysis), mid-2015 onward.
           Fine for understanding demand and for an "oracle weather" upper bound.
           NOT allowed as a model input in an honest backtest: you never know
           tomorrow's actual weather when you forecast tomorrow.

forecast - what the weather forecast said 1 and 2 days ahead (Previous Runs API),
           January 2024 onward. This is what a real day-ahead forecaster would have
           had, so honest backtests use these columns.

Writes data/raw/weather/<kind>/<subregion>_<year>.parquet (UTC timestamps).

Run:  uv run python -m gridfc.fetch_weather
"""
import argparse
import datetime as dt
import time

import pandas as pd
import requests

from .config import RAW_DIR, WEATHER_POINTS, WEATHER_VARS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Previous Runs API. Second host is a fallback in case Open-Meteo moves it.
PREVIOUS_RUNS_URLS = [
    "https://previous-runs-api.open-meteo.com/v1/forecast",
    "https://historical-forecast-api.open-meteo.com/v1/forecast",
]
FORECAST_VARS = ["temperature_2m", "dew_point_2m", "apparent_temperature",
                 "cloud_cover", "wind_speed_10m", "shortwave_radiation",
                 # added Nov 15, 2026 (needs --refresh-all once)
                 "relative_humidity_2m", "precipitation", "snowfall", "pressure_msl"]
LEAD_DAYS = [1, 2]

KINDS = {
    "actual": {"start": dt.date(2015, 7, 1), "lag_days": 6},
    "forecast": {"start": dt.date(2024, 1, 1), "lag_days": 1},
}


def get(url, params, tries=5):
    msg = ""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            msg = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 400:  # bad request: retrying won't help
                break
        except requests.RequestException as e:
            msg = str(e)
        time.sleep(2 ** i)
    raise RuntimeError(msg)


def fetch(kind, lat, lon, start, end):
    if kind == "actual":
        params = {"latitude": lat, "longitude": lon, "start_date": start.isoformat(),
                  "end_date": end.isoformat(), "hourly": ",".join(WEATHER_VARS),
                  "timezone": "GMT"}
        return get(ARCHIVE_URL, params)
    hourly = [f"{v}_previous_day{d}" for v in FORECAST_VARS for d in LEAD_DAYS]
    params = {"latitude": lat, "longitude": lon, "start_date": start.isoformat(),
              "end_date": end.isoformat(), "hourly": ",".join(hourly), "timezone": "GMT"}
    last_err = None
    for url in PREVIOUS_RUNS_URLS:
        try:
            return get(url, params)
        except RuntimeError as e:
            last_err = e
    raise RuntimeError(f"Previous Runs API failed on all hosts: {last_err}")


def to_frame(payload, subba, city):
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.insert(0, "subba", subba)
    df.insert(1, "city", city)
    return df.rename(columns={"time": "period"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(KINDS))
    ap.add_argument("--refresh-all", action="store_true")
    args = ap.parse_args()
    today = dt.datetime.now(dt.timezone.utc).date()

    for kind, cfg in KINDS.items():
        if args.only and kind != args.only:
            continue
        out_dir = RAW_DIR / "weather" / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        last_day = today - dt.timedelta(days=cfg["lag_days"])
        for subba, (city, lat, lon) in WEATHER_POINTS.items():
            for year in range(cfg["start"].year, last_day.year + 1):
                path = out_dir / f"{subba}_{year}.parquet"
                if path.exists() and year < last_day.year and not args.refresh_all:
                    continue
                start = max(cfg["start"], dt.date(year, 1, 1))
                end = min(last_day, dt.date(year, 12, 31))
                print(f"{kind} {city} {year} ...", flush=True)
                try:
                    df = to_frame(fetch(kind, lat, lon, start, end), subba, city)
                except RuntimeError as e:
                    print(f"  FAILED: {e}")
                    continue
                df.to_parquet(path, index=False)
                print(f"  {len(df):,} rows", flush=True)
                time.sleep(1)  # be polite to the free API


if __name__ == "__main__":
    main()
