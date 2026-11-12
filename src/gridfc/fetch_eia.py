"""Download EIA-930 hourly data for PJM.

pjm_region: PJM demand (D) and PJM's own day-ahead demand forecast (DF), from mid-2015.
pjm_subba:  hourly demand for PJM's subregions (CE = ComEd/Chicago), from mid-2018.

Writes one Parquet file per dataset per year to data/raw/eia930/<dataset>/<year>.parquet.
Past years are skipped if already downloaded; the current year is always refreshed.
EIA revises recent data, so use --refresh-all now and then to pick up revisions.

Run:  uv run python -m gridfc.fetch_eia
"""
import argparse
import datetime as dt
import time

import pandas as pd
import requests

from .config import RAW_DIR
from .secrets import get_eia_key, redact

BASE = "https://api.eia.gov/v2/electricity/rto"
PAGE = 5000  # EIA's max rows per request

DATASETS = {
    "pjm_region": {
        "route": "region-data",
        "facets": {"respondent": ["PJM"], "type": ["D", "DF"]},
        "start_year": 2015,
    },
    "pjm_subba": {
        "route": "region-sub-ba-data",
        "facets": {"parent": ["PJM"]},
        "start_year": 2018,
    },
}


def get_json(session, url, params, key, tries=5):
    msg = ""
    for i in range(tries):
        try:
            r = session.get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            msg = f"HTTP {r.status_code}: {r.text[:300]}"
        except requests.RequestException as e:
            msg = str(e)
        wait = 2 ** i
        print(f"  retry {i + 1}/{tries} in {wait}s ({redact(msg, key)})", flush=True)
        time.sleep(wait)
    raise SystemExit(redact(f"Giving up on {url}: {msg}", key))


def fetch_range(route, facets, start, end, key, session):
    rows, offset = [], 0
    while True:
        params = [
            ("api_key", key),
            ("frequency", "hourly"),  # UTC hours
            ("data[0]", "value"),
            ("start", start),
            ("end", end),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"),
            ("offset", offset),
            ("length", PAGE),
        ]
        for f, vals in facets.items():
            for v in vals:
                params.append((f"facets[{f}][]", v))
        resp = get_json(session, f"{BASE}/{route}/data/", params, key)["response"]
        data = resp.get("data", [])
        rows.extend(data)
        offset += len(data)
        total = int(resp.get("total", 0) or 0)
        if not data or offset >= total:
            return rows
        time.sleep(0.2)


def to_frame(rows, fetched_at):
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["fetched_at_utc"] = fetched_at  # keeps track of which data vintage we used
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(DATASETS))
    ap.add_argument("--refresh-all", action="store_true", help="re-download past years too")
    args = ap.parse_args()

    key = get_eia_key()
    session = requests.Session()
    now = dt.datetime.now(dt.timezone.utc)

    for name, cfg in DATASETS.items():
        if args.only and name != args.only:
            continue
        out_dir = RAW_DIR / "eia930" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for year in range(cfg["start_year"], now.year + 1):
            path = out_dir / f"{year}.parquet"
            if path.exists() and year < now.year and not args.refresh_all:
                continue
            print(f"{name} {year} ...", flush=True)
            rows = fetch_range(cfg["route"], cfg["facets"], f"{year}-01-01T00",
                               f"{year}-12-31T23", key, session)
            if not rows:
                print("  no rows")
                continue
            df = to_frame(rows, now)
            df.to_parquet(path, index=False)
            print(f"  {len(df):,} rows, {df['period'].min()} to {df['period'].max()}", flush=True)


if __name__ == "__main__":
    main()
