from .baselines import SeasonalNaive
from .gbm import LightGBMQuantile
from .linear import LinearTemperature
from .pretrained import COVARIATES_FULL, Chronos2, TimesFM3
from .prophet_model import ProphetModel
from .stats import MSTLModel

REGISTRY = {
    "weekly_naive": lambda: SeasonalNaive(lag_h=168, name="weekly_naive"),
    # same model, two ways to get training weather (see harness hist_weather)
    "lgbm": lambda: LightGBMQuantile(name="lgbm"),
    "lgbm_wxact": lambda: LightGBMQuantile(name="lgbm_wxact", hist_weather="actual"),
    # a straight line inside each leaf, so forecasts can go past the training range
    "lgbm_linear": lambda: LightGBMQuantile(name="lgbm_linear", linear_tree=True),
    "linear": lambda: LinearTemperature(),
    "mstl": lambda: MSTLModel(),
    "prophet": lambda: ProphetModel(),
    # pretrained, zero-shot; with and without weather inputs
    "chronos2": lambda: Chronos2(),
    "chronos2_uni": lambda: Chronos2(name="chronos2_uni", covariates=()),
    "timesfm3": lambda: TimesFM3(),
    "timesfm3_uni": lambda: TimesFM3(name="timesfm3_uni", covariates=()),
    # all six forecast weather variables + holiday types (Nov 15 experiment)
    "chronos2_full": lambda: Chronos2(name="chronos2_full", covariates=COVARIATES_FULL),
    "timesfm3_full": lambda: TimesFM3(name="timesfm3_full", covariates=COVARIATES_FULL),
}
