"""Deterministic demos with arbitrary names, missing values and unseen categories."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from .data import DataBundle


def make_demo(kind="classification", n=360, seed=42):
    rng = np.random.default_rng(seed)
    if kind in {"classification", "regression", "multiclass"}:
        if kind == "regression":
            x, y = make_regression(n_samples=n, n_features=7, n_informative=5, noise=12, random_state=seed)
        else:
            x, y = make_classification(n_samples=n, n_features=7, n_informative=5,
                                       n_redundant=1, n_classes=3 if kind == "multiclass" else 2,
                                       random_state=seed)
            y = np.array(["down", "flat", "up"])[y] if kind == "multiclass" else np.where(y, "yes", "no")
        df = pd.DataFrame(x, columns=[f"measurement_{i}" for i in range(7)])
        df["segment"] = np.where(df.measurement_0 > 0, "alpha", "beta")
        df.loc[rng.choice(n, n // 20, replace=False), "measurement_2"] = np.nan
        df["empty_sensor"] = np.nan
        df["record"] = [f"row-{i:05d}" for i in range(n)]
        cutoff = int(n * .8)
        df.loc[cutoff::7, "segment"] = "unseen"
    elif kind == "financial":
        days = max(n // 6, 30)
        market = rng.normal(0, .01, days)
        rows, labels = [], []
        for asset in range(6):
            previous = 0.
            for day in range(days):
                value = -.35 * previous + market[day] + rng.normal(0, .006)
                rows.append((f"tick-{day}-{asset}", day, f"asset-{asset}", f"sector-{asset % 2}", value))
                labels.append(int(-.8 * value + rng.normal(0, .006) > 0))
                previous = value
        df = pd.DataFrame(rows, columns=["record", "session", "instrument", "sector", "move"])
        df["liquidity"] = rng.lognormal(size=len(df))
        df["past_a"] = df.groupby("instrument").move.shift(1)
        df["past_b"] = df.groupby("instrument").move.shift(2)
        y = np.asarray(labels)
        mask = df.session < int(days * .8)
        return DataBundle(df.loc[mask].reset_index(drop=True), pd.Series(y[mask]), df.loc[~mask].reset_index(drop=True))
    else:
        raise ValueError(f"Unknown demo: {kind}")
    return DataBundle(df.iloc[:cutoff].reset_index(drop=True), pd.Series(y[:cutoff]), df.iloc[cutoff:].reset_index(drop=True))


def financial_config(config):
    """Explicit financial example; no financial column assumptions in the toolkit."""
    from copy import deepcopy
    c = deepcopy(config)
    c["task"] = "classification"
    c["data"].update(demo="financial", time_column="session", group_column="instrument")
    c["validation"].update(strategy="temporal", gap=1, label_horizon=1, shuffle=False)
    c["features"]["histories"] = {"moves": ["move", "past_a", "past_b"]}
    c["features"]["temporal"].update(columns=["move"], entity_columns=["instrument"],
                                     lags=[1, 2], windows=[3, 5], availability="online")
    c["features"]["cross_sectional"].update(columns=["move"], keys=["session", "sector"],
                                            available_at_prediction=True)
    c["features"]["group_statistics"].update(columns=["liquidity"], keys=["sector"])
    c["metric"]["name"] = "accuracy"
    return c
