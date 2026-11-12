from .baselines import SeasonalNaive

REGISTRY = {
    "weekly_naive": lambda: SeasonalNaive(lag_h=168, name="weekly_naive"),
}
