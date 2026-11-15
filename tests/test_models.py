import numpy as np
import pandas as pd

from gridfc import harness
from gridfc.harness import QCOLS
from gridfc.models import REGISTRY
from gridfc.models.gbm import day_ahead_features


def make_frame(days=90):
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    temp = 10 + 8 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 2, len(idx))
    y = 90_000 + 800 * np.abs(temp - 15) + rng.normal(0, 500, len(idx))
    d2 = {f"wx_fcst_d2_{c}": v for c, v in [
        ("temperature_2m", temp + 0.5), ("apparent_temperature", temp - 1),
        ("wind_speed_10m", 10 + rng.normal(0, 2, len(idx))),
        ("shortwave_radiation", np.maximum(temp * 20, 0))]}
    actual = {c.replace("wx_fcst_d2_", "wx_actual_"): v for c, v in d2.items()}
    return pd.DataFrame({"demand": y, **actual, **d2}, index=idx)


def test_features_use_only_pre_cutoff_demand():
    frame = make_frame()
    issue = harness.issue_times("2025-03-01", "2025-03-01")[0]
    targets = harness.target_hours(issue)
    cutoff = issue - pd.Timedelta(hours=harness.DATA_LAG_H)
    X = harness.allowed_features(frame)
    full = day_ahead_features(frame["demand"], X.reindex(targets))
    seen = day_ahead_features(frame["demand"][:cutoff], X.reindex(targets))
    pd.testing.assert_frame_equal(full, seen)  # future demand changes nothing
    assert seen[["lag48", "lag168", "last", "d2_mean"]].notna().all().all()


def test_lgbm_runs_through_harness():
    for name in ["lgbm", "lgbm_wxact"]:
        model = REGISTRY[name]()
        model.params["n_estimators"] = 30
        out = harness.run_backtest(model, make_frame(), "2025-03-20", "2025-03-24", verbose=False)
        assert len(out) == 5 * 24 and out[["mean"] + QCOLS].notna().all().all()
        assert (np.diff(out[QCOLS].values, axis=1) >= 0).all()


def check_forecast(out, days, quantiles=QCOLS):
    assert len(out) == days * 24 and out[["mean"] + quantiles].notna().all().all()
    assert (np.diff(out[quantiles].values, axis=1) >= -1e-6).all()


def test_trained_models_run_through_harness():
    for name in ["linear", "mstl", "prophet", "lgbm_linear"]:
        model = REGISTRY[name]()
        if name == "lgbm_linear":
            model.params["n_estimators"] = 30
        out = harness.run_backtest(model, make_frame(), "2025-03-20", "2025-03-22", verbose=False)
        check_forecast(out, 3, [] if name == "lgbm_linear" else QCOLS)


def test_linear_extrapolates_past_training_range():
    """Colder than any training day -> higher load than any training hour."""
    frame = make_frame()
    rng = np.random.default_rng(1)
    daily = np.repeat(rng.uniform(-5, 25, len(frame) // 24), 24)
    for c in ["wx_fcst_d2_temperature_2m", "wx_fcst_d2_apparent_temperature"]:
        frame[c] = daily
    frame["demand"] = 90_000 + 1_500 * np.maximum(18 - daily, 0) + rng.normal(0, 300, len(frame))
    model = REGISTRY["linear"]()
    issue = harness.issue_times("2025-03-25", "2025-03-25")[0]
    cutoff = issue - pd.Timedelta(hours=harness.DATA_LAG_H)
    X = harness.allowed_features(frame)
    model.fit(frame["demand"][:cutoff], X[:cutoff])
    future = X.reindex(harness.target_hours(issue)).copy()
    for c in ["wx_fcst_d2_temperature_2m", "wx_fcst_d2_apparent_temperature"]:
        future[c] = -30.0
    pred = model.predict(frame["demand"][:cutoff], X[:cutoff], future)
    assert pred["mean"].min() > frame["demand"][:cutoff].max()


class FakeChronos:
    """Stands in for the Chronos-2 pipeline: returns quantiles = level * 1000."""
    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch
        assert inputs[0]["future_covariates"]["cal_holiday"].shape == (prediction_length,)
        q = torch.tensor(quantile_levels).repeat(prediction_length, 1)[None] * 1000
        return [q], [torch.full((1, prediction_length), 500.0)]


class FakeTimesFM:
    def predict(self, context, horizon, return_quantiles, past_future_covariates):
        from types import SimpleNamespace
        assert past_future_covariates.shape == (3, len(context) + horizon)
        dec = np.tile(np.arange(1, 10) / 10, (horizon, 1)) * 1000
        return SimpleNamespace(forecast=np.full(horizon, 500.0), quantiles=dec)


def test_pretrained_wrappers_map_quantiles():
    chronos, tfm = REGISTRY["chronos2"](), REGISTRY["timesfm3"]()
    chronos.pipe, tfm.model = FakeChronos(), FakeTimesFM()
    out = harness.run_backtest(chronos, make_frame(), "2025-03-20", "2025-03-21", verbose=False)
    check_forecast(out, 2)
    assert np.allclose(out[QCOLS].iloc[0].values, np.array(harness.QUANTILES) * 1000, atol=1e-3)
    out = harness.run_backtest(tfm, make_frame(), "2025-03-20", "2025-03-21", verbose=False)
    inner = ["q10", "q25", "q50", "q75", "q90"]
    check_forecast(out, 2, inner)
    assert np.allclose(out[inner].iloc[0].values, [100, 250, 500, 750, 900])
    assert out[["q05", "q95"]].isna().all().all()
