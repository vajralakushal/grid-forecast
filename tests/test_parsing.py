import pandas as pd

from gridfc import fetch_eia
from gridfc.secrets import redact


class FakeResp:
    def __init__(self, payload):
        self.status_code = 200
        self._p = payload

    def json(self):
        return self._p


class FakeSession:
    """Serves 3 rows in pages of 2, like EIA's offset/length paging."""
    rows = [
        {"period": "2024-01-01T00", "respondent": "PJM", "type": "D", "value": "90000"},
        {"period": "2024-01-01T00", "respondent": "PJM", "type": "DF", "value": "91000"},
        {"period": "2024-01-01T01", "respondent": "PJM", "type": "D", "value": "88000"},
    ]

    def get(self, url, params, timeout):
        p = dict(params)
        off, n = int(p["offset"]), 2
        return FakeResp({"response": {"total": "3", "data": self.rows[off:off + n]}})


def test_paging_and_frame(monkeypatch):
    monkeypatch.setattr(fetch_eia, "PAGE", 2)
    monkeypatch.setattr(fetch_eia.time, "sleep", lambda s: None)
    rows = fetch_eia.fetch_range("region-data", {"respondent": ["PJM"]}, "a", "b", "k", FakeSession())
    assert len(rows) == 3
    df = fetch_eia.to_frame(rows, pd.Timestamp("2026-11-11", tz="UTC"))
    assert str(df["period"].dt.tz) == "UTC"
    assert df["value"].sum() == 269000


def test_redact():
    assert redact("url?api_key=SECRET123&x=1", "SECRET123") == "url?api_key=***&x=1"
