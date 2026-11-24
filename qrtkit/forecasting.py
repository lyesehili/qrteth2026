"""Fixed-origin price baselines: future prices are never inputs.

The caller supplies the future quote schedule; no exchange calendar, session
length, target transformation or competition scoring formula is inferred.
"""
import numpy as np
import pandas as pd
from .features import check_time


def forecast_prices(history, future, *, entity_columns, time_column, price_column,
                    method="last", drift_window=100, damping=.99):
    """Predict an entire future price panel from a single historical cutoff.

    last: repeat each stock's last observed price.
    damped_drift: extrapolate the training-only mean log price change, damping
    successive quote-step drift by damping**h. Steps count supplied future rows
    per entity, including across sessions; they are not calendar minutes.

    Output follows future.index/order. Entity columns may include country when
    stock identifiers are not globally unique. Prices must be positive and
    finite; clean missing quotes and verify adjustments before calling.
    """
    keys = list(entity_columns)
    if not keys or len(set(keys + [time_column, price_column])) != len(keys) + 2:
        raise ValueError("Use distinct entity, time and price columns")
    if method not in {"last", "damped_drift"}:
        raise ValueError("method must be last or damped_drift")
    if not isinstance(drift_window, int) or drift_window < 1 or not 0 < damping < 1:
        raise ValueError("drift_window must be positive and damping strictly between zero and one")
    if price_column in future:
        raise ValueError("Future prices cannot be inputs; remove the price column explicitly")
    if not len(history) or not len(future) or not future.index.is_unique:
        raise ValueError("Need nonempty history/future and a unique future index")
    for name, frame in [("history", history), ("future", future)]:
        check_time(frame[time_column])
        if frame[keys].isna().any().any() or frame.duplicated(keys + [time_column]).any():
            raise ValueError(f"{name} entity/time keys must be nonmissing and unique")
    if not history[time_column].max() < future[time_column].min():
        raise ValueError("All future timestamps must follow the shared historical cutoff")
    prices = history[price_column].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Historical prices must be finite and strictly positive")
    h = history[keys + [time_column, price_column]].sort_values(time_column, kind="stable").copy()
    h["__log_price"] = np.log(h[price_column].astype(float))
    h["__change"] = h.groupby(keys, sort=False, observed=True)["__log_price"].diff()
    last = h.groupby(keys, sort=False, observed=True).tail(1)[keys + [price_column]].rename(columns={price_column: "__last"})
    drift = (h.groupby(keys, sort=False, observed=True).tail(drift_window)
             .groupby(keys, sort=False, observed=True)["__change"].mean().fillna(0.).rename("__drift").reset_index())
    stats = last.merge(drift, on=keys, validate="one_to_one")
    f = future[keys + [time_column]].copy()
    f["__order"] = np.arange(len(f))
    f = f.sort_values(time_column, kind="stable")
    f["__horizon"] = f.groupby(keys, sort=False, observed=True).cumcount() + 1
    f = f.merge(stats, on=keys, how="left", validate="many_to_one", sort=False)
    if f["__last"].isna().any():
        raise ValueError("Future contains stocks with no historical price; define an explicit cold-start rule")
    if method == "last":
        f["__prediction"] = f["__last"]
    else:
        accumulated_drift = damping * (1 - damping ** f["__horizon"]) / (1 - damping)
        with np.errstate(over="ignore", under="ignore"):
            f["__prediction"] = np.exp(np.log(f["__last"]) + f["__drift"] * accumulated_drift)
    prediction = f.sort_values("__order")["__prediction"].to_numpy()
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("Price extrapolation overflowed/underflowed; reduce drift/damping or use last")
    return pd.Series(prediction, index=future.index, name="prediction")
