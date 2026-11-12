from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# One weather point per large PJM subregion, at its main city.
# Keys are EIA-930 subregion codes (parent = PJM). CE = ComEd (Chicago).
WEATHER_POINTS = {
    "CE":   ("Chicago",      41.88, -87.63),
    "PE":   ("Philadelphia", 39.95, -75.17),
    "PEP":  ("Washington",   38.90, -77.04),
    "BC":   ("Baltimore",    39.29, -76.61),
    "DUQ":  ("Pittsburgh",   40.44, -79.99),
    "AEP":  ("Columbus",     39.96, -83.00),
    "ATSI": ("Cleveland",    41.50, -81.69),
    "PS":   ("Newark",       40.74, -74.17),
    "DOM":  ("Richmond",     37.54, -77.44),
    "DEOK": ("Cincinnati",   39.10, -84.51),
}

WEATHER_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "precipitation",
]
