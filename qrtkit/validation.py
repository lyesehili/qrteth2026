"""Explicit fold indices with timestamp-level gaps and label-availability purging."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, TimeSeriesSplit
from .features import check_time


def last_period_holdout(X, *, time_column, test_periods, period_column=None,
                        gap=0, label_end_column=None):
    """Hold out complete final date/session blocks across the whole stock panel.

    Default blocks are observed UTC dates for timezone-aware timestamps, or dates
    in the supplied clock for naive timestamps. Numeric time requires an explicit
    period_column. test_periods counts observed blocks, not elapsed calendar days.
    gap counts unique timestamps, consistent with the other temporal splitter.
    """
    if not time_column:
        raise ValueError("Day/session holdouts require time_column")
    check_time(X[time_column])
    if not isinstance(test_periods, int) or test_periods < 1 or not isinstance(gap, int) or gap < 0:
        raise ValueError("test_periods must be positive and gap nonnegative integers")
    clock = X[time_column]
    if period_column:
        blocks = X[period_column]
    elif pd.api.types.is_datetime64_any_dtype(clock):
        blocks = clock.dt.tz_convert("UTC").dt.normalize() if clock.dt.tz is not None else clock.dt.normalize()
    else:
        raise ValueError("Numeric time requires period_column for day/session holdouts")
    if blocks.isna().any():
        raise ValueError("Holdout periods cannot be missing")
    timeline = pd.DataFrame({"time": clock.to_numpy(), "block": blocks.to_numpy()}).sort_values("time", kind="stable")
    if (timeline.groupby("time")["block"].nunique() != 1).any():
        raise ValueError("A timestamp maps to multiple periods across stocks; use a shared global clock")
    timeline = timeline.drop_duplicates("time")
    runs = timeline.loc[timeline.block.ne(timeline.block.shift()), "block"]
    if runs.duplicated().any():
        raise ValueError("Period blocks must be contiguous in chronological time")
    if len(runs) <= test_periods:
        raise ValueError("Need historical periods before the requested final holdout")
    valid_blocks = set(runs.iloc[-test_periods:])
    valid = np.flatnonzero(blocks.isin(valid_blocks))
    start = clock.iloc[valid].min()
    times = timeline.loc[timeline.time < start, "time"].to_numpy()
    if gap:
        times = times[:-gap]
    train = np.flatnonzero(clock.isin(times))
    if label_end_column:
        check_time(X[label_end_column])
        if (X[label_end_column] < clock).any():
            raise ValueError("Label availability precedes prediction time")
        train = train[(X.iloc[train][label_end_column] < start).to_numpy()]
    if not len(train) or not len(valid):
        raise ValueError("Holdout gap/purging leaves an empty training or validation set")
    return train, valid


def make_splits(X, y, config):
    v, d = config["validation"], config["data"]
    strategy, n = v["strategy"], v["n_splits"]
    seed = config["seed"] if v.get("shuffle", True) else None
    if strategy in {"standard", "stratified"}:
        if strategy == "stratified" and config["task"] != "classification":
            raise ValueError("Stratification needs classification; use standard/group/temporal for regression")
        cls = StratifiedKFold if strategy == "stratified" else KFold
        splitter = cls(n_splits=n, shuffle=v.get("shuffle", True), random_state=seed)
        splits = list(splitter.split(X, y))
    elif strategy in {"group", "stratified_group"}:
        group = d.get("group_column")
        if not group or X[group].isna().any():
            raise ValueError("Grouped validation needs a nonmissing group_column")
        if strategy == "stratified_group":
            if config["task"] != "classification":
                raise ValueError("stratified_group requires classification")
            splitter = StratifiedGroupKFold(n_splits=n, shuffle=v.get("shuffle", True), random_state=seed)
        else:
            splitter = GroupKFold(n_splits=n)
        splits = list(splitter.split(X, y, groups=X[group]))
    elif strategy == "temporal_holdout":
        if n != 1:
            raise ValueError("temporal_holdout requires n_splits=1")
        for field in ["gap", "label_horizon"]:
            if not isinstance(v.get(field, 0), int) or v.get(field, 0) < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        splits = [last_period_holdout(
            X, time_column=d.get("time_column"), test_periods=v["test_periods"],
            period_column=v.get("period_column"), gap=v.get("gap", 0) + v.get("label_horizon", 0),
            label_end_column=v.get("label_end_column"))]
    elif strategy == "temporal":
        time = d.get("time_column")
        if not time:
            raise ValueError("Temporal validation requires time_column")
        check_time(X[time])
        periods = np.sort(X[time].unique())
        for field in ["gap", "label_horizon"]:
            if not isinstance(v.get(field, 0), int) or v.get(field, 0) < 0:
                raise ValueError(f"{field} must be a nonnegative integer in distinct timestamp units")
        splitter = TimeSeriesSplit(n_splits=n, gap=v.get("gap", 0) + v.get("label_horizon", 0),
                                   max_train_size=v.get("max_train_periods"))
        splits = []
        for tr, va in splitter.split(periods):
            train = np.flatnonzero(X[time].isin(periods[tr]))
            valid = np.flatnonzero(X[time].isin(periods[va]))
            end = v.get("label_end_column")
            if end:
                check_time(X[end])
                if (X[end] < X[time]).any():
                    raise ValueError("label_end_column precedes its row's prediction time")
                train = train[(X.iloc[train][end] < periods[va[0]]).to_numpy()]
            splits.append((train, valid))
    else:
        raise ValueError(f"Unknown validation strategy: {strategy}")
    seen = np.zeros(len(X), dtype=int)
    for tr, va in splits:
        if not len(tr) or not len(va) or np.intersect1d(tr, va).size:
            raise ValueError("Empty or overlapping train/validation fold")
        seen[va] += 1
        if strategy in {"group", "stratified_group"}:
            if set(X.iloc[tr][d["group_column"]]) & set(X.iloc[va][d["group_column"]]):
                raise ValueError("Group overlap")
        if strategy in {"temporal", "temporal_holdout"} and not X.iloc[tr][d["time_column"]].max() < X.iloc[va][d["time_column"]].min():
            raise ValueError("Temporal order violated")
    if (seen > 1).any():
        raise ValueError("Validation folds overlap")
    return splits


def check_feature_splits(X, test, splits, config):
    """Reject recipes whose prediction-time assumptions disagree with folds."""
    f, d = config.get("features", {}), config["data"]
    if f.get("temporal", {}).get("columns") and config["validation"]["strategy"] not in {"temporal", "temporal_holdout"}:
        raise ValueError("Long-form lag/rolling features require temporal validation")
    cs = f.get("cross_sectional", {})
    if cs.get("columns"):
        if not cs.get("available_at_prediction") or not cs.get("keys"):
            raise ValueError("Explicit cross-section availability and keys required")
        def keyset(frame):
            return set(frame[cs["keys"]].itertuples(index=False, name=None))
        for tr, va in splits:
            if keyset(X.iloc[tr]) & keyset(X.iloc[va]):
                raise ValueError("A cross section is split across train/validation; group folds by timestamp or use temporal")
        if test is not None and keyset(X) & keyset(test):
            raise ValueError("Cross sections overlap train/test; define a complete prediction batch explicitly")
    if config["validation"]["strategy"] in {"temporal", "temporal_holdout"} and test is not None:
        time = d["time_column"]
        check_time(test[time])
        if not X[time].max() < test[time].min():
            raise ValueError("Temporal test rows must follow all training timestamps")


def split_fingerprint(splits):
    h = hashlib.sha256()
    for tr, va in splits:
        h.update(np.asarray(tr, dtype="int64").tobytes())
        h.update(b"validation")
        h.update(np.asarray(va, dtype="int64").tobytes())
    return h.hexdigest()
