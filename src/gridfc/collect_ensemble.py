"""Snapshot today's ensemble weather forecasts for the 10 PJM cities.

Why: Open-Meteo keeps individual ensemble members for only ~3 days. Monte Carlo
backtests later need the members as they were issued, so we save them daily.
Runs in GitHub Actions each morning (~9 am Eastern, before the 10 am issue time).

Output (long format, float32):
  issued_at, subba, valid_time, model, member, <one column per variable>
  member 0 = control run.

  uv run python -m gridfc.collect_ensemble --out data/ensemble
"""
import argparse
import datetime as dt
import re
import time
from pathlib import Path

import pandas as pd
import requests

from .config import WEATHER_POINTS

URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODELS = ["ecmwf_ifs025", "gfs025"]  # ECMWF (51 members) and NOAA GFS (31 members)
VARS = ["temperature_2m", "dew_point_2m", "cloud_cover", "wind_speed_10m"]
FORECAST_DAYS = 3

MEMBER = re.compile(r"_?member(\d+)")


def parse_column(col):
    """Split an Open-Meteo column name into (variable, member, model suffix).

    Works for any suffix order, e.g. temperature_2m, temperature_2m_member03,
    temperature_2m_member03_ecmwf_ifs025, temperature_2m_ecmwf_ifs025_ensemble.
    Returns None for columns that aren't one of VARS.
    """
    var = max((v for v in VARS if col == v or col.startswith(v + "_")), key=len, default=None)
    if var is None:
        return None
    rest = col[len(var):]
    m = MEMBER.search(rest)
    member = int(m.group(1)) if m else 0
    model = MEMBER.sub("", rest).strip("_") or None
    return var, member, model


def fetch(lat, lon, models, tries=5):
    """GET with retries on timeouts, connection errors, 429 and 5xx (with backoff)."""
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(VARS),
              "forecast_days": FORECAST_DAYS, "timezone": "GMT"}
    if models:
        params["models"] = ",".join(models)
    last = ""
    for i in range(tries):
        try:
            r = requests.get(URL, params=params, timeout=(10, 60))  # (connect, read) seconds
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400:
                raise ValueError(r.text[:300])
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except (requests.Timeout, requests.ConnectionError) as e:
            last = type(e).__name__
        wait = min(5 * 2 ** i, 60)
        print(f"  attempt {i + 1}/{tries} failed ({last}); waiting {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"gave up after {tries} attempts: {last}")


def to_long(payload, subba, issued_at, default_model):
    h = payload["hourly"]
    times = pd.to_datetime(h.pop("time"), utc=True)
    parts = []
    for col, values in h.items():
        parsed = parse_column(col)
        if parsed is None:
            continue
        var, member, model = parsed
        parts.append(pd.DataFrame({
            "valid_time": times, "var": var, "member": member,
            "model": model or default_model,
            "value": pd.Series(values, dtype="float32"),
        }))
    if not parts:
        raise ValueError(f"No usable columns for {subba}. Columns returned: {list(h)[:15]}")
    df = pd.concat(parts, ignore_index=True)
    wide = df.pivot_table(index=["valid_time", "model", "member"], columns="var",
                          values="value", aggfunc="first").reset_index()
    wide.insert(0, "subba", subba)
    wide.insert(0, "issued_at", issued_at)
    return wide


def _collect(todo, issued_at, frames, failed):
    for subba, (city, lat, lon) in todo:
        try:
            try:
                payload, model_label = fetch(lat, lon, MODELS), None
            except ValueError as e:  # model names changed? fall back to the default ensemble
                print(f"{city}: models rejected ({e}); using Open-Meteo default")
                payload, model_label = fetch(lat, lon, None), "default"
        except RuntimeError as e:
            print(f"{city}: FAILED ({e}); continuing with the other cities", flush=True)
            failed.append(city)
            continue
        df = to_long(payload, subba, issued_at, model_label or MODELS[0])
        frames.append(df)
        print(f"{city}: {len(df):,} rows, models {sorted(df['model'].unique())}, "
              f"members {df['member'].nunique()}", flush=True)
        time.sleep(1)

    return frames, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ensemble")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    issued_at = pd.Timestamp(dt.datetime.now(dt.timezone.utc)).floor("h")

    frames, failed = [], []
    todo = list(WEATHER_POINTS.items())
    for pass_no in (1, 2):
        if pass_no == 2:
            if not failed:
                break
            print(f"Second pass for {failed} in 60s ...", flush=True)
            time.sleep(60)
            todo = [(sb, pt) for sb, pt in WEATHER_POINTS.items() if pt[0] in failed]
            failed = []
        frames, failed = _collect(todo, issued_at, frames, failed)

    if not frames:
        raise SystemExit("All cities failed; nothing saved.")
    all_df = pd.concat(frames, ignore_index=True)
    path = out / f"ensemble_{issued_at:%Y%m%dT%H}Z.parquet"
    all_df.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {path} ({len(all_df):,} rows, {path.stat().st_size / 1e6:.2f} MB)")
    if failed:
        print(f"WARNING: missing cities this run: {failed}")


if __name__ == "__main__":
    main()
