from .baselines import SeasonalNaive
from .gbm import LightGBMQuantile

REGISTRY = {
    "weekly_naive": lambda: SeasonalNaive(lag_h=168, name="weekly_naive"),
    # same model, two ways to get training weather (see harness hist_weather)
    "lgbm": lambda: LightGBMQuantile(name="lgbm"),
    "lgbm_wxact": lambda: LightGBMQuantile(name="lgbm_wxact", hist_weather="actual"),
}
