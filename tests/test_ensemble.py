import pandas as pd

from gridfc.collect_ensemble import to_long


def test_member_and_model_parsing():
    payload = {"hourly": {
        "time": ["2026-11-12T00:00", "2026-11-12T01:00"],
        "temperature_2m_ecmwf_ifs025": [1.0, 2.0],
        "temperature_2m_member01_ecmwf_ifs025": [1.5, 2.5],
        "temperature_2m_member01_gfs025": [0.5, 1.5],
        "cloud_cover_member01_gfs025": [10, 20],
    }}
    df = to_long(payload, "CE", pd.Timestamp("2026-11-11T13:00Z"), "ecmwf_ifs025")
    assert set(df["model"]) == {"ecmwf_ifs025", "gfs025"}
    assert set(df["member"]) == {0, 1}
    row = df[(df.model == "gfs025") & (df.member == 1)].iloc[0]
    assert row["temperature_2m"] == 0.5 and row["cloud_cover"] == 10


def test_parse_column_variants():
    from gridfc.collect_ensemble import parse_column
    assert parse_column("temperature_2m") == ("temperature_2m", 0, None)
    assert parse_column("temperature_2m_member07") == ("temperature_2m", 7, None)
    assert parse_column("dew_point_2m_member12_gfs025") == ("dew_point_2m", 12, "gfs025")
    assert parse_column("cloud_cover_ecmwf_ifs025_ensemble") == ("cloud_cover", 0, "ecmwf_ifs025_ensemble")
    assert parse_column("temperature_2m_member03_ecmwf_ifs025_ensemble") == ("temperature_2m", 3, "ecmwf_ifs025_ensemble")
    assert parse_column("time") is None
