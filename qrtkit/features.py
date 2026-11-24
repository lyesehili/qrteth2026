"""Person 1's feature recipe; Person 2 clones and fits it inside each CV fold.

No target is ever consumed. Histories list columns NEWEST FIRST, already known
at prediction time. Long-form lags exclude the current timestamp. Same-time
cross sections require a complete, explicitly available prediction batch.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _positive_ints(values, name):
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError(f"{name} must contain positive integers")


def check_time(series):
    if series.isna().any() or not (
        pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)
    ):
        raise ValueError("Time must be nonmissing numeric or datetime, with verified chronological order; parse dates explicitly")


def _temporal_features(X, spec, time, history=None):
    """Vectorized rolling sums per entity; frozen queries use only fitted history."""
    keys, columns = spec.get("entity_columns", []), spec["columns"]
    lags, windows = spec.get("lags", [1]), spec.get("windows", [5])
    _positive_ints(lags, "lags")
    _positive_ints(windows, "windows")
    if not time or not (lags or windows):
        raise ValueError("Temporal features need time_column and lags/windows")
    check_time(X[time])
    if keys and X[keys].isna().any().any():
        raise ValueError("Temporal entity keys cannot be missing")
    if X.duplicated(keys + [time]).any():
        raise ValueError("Duplicate entity/time rows: aggregate explicitly before creating lags")
    source = X[keys + [time] + columns].copy() if history is None else history.copy()
    if history is not None and spec.get("availability", "frozen") == "online":
        source = pd.concat([source, X[source.columns]], ignore_index=True)
        if source.duplicated(keys + [time]).any():
            raise ValueError("Prediction entity/time keys overlap fitted history")
    source = source.sort_values(time, kind="stable")
    query = X.reset_index(drop=True)
    result = {}
    grouped_source = {(): source} if not keys else dict(tuple(source.groupby(keys, sort=False)))
    grouped_query = [((), query)] if not keys else query.groupby(keys, sort=False)
    for group, queries in grouped_query:
        past = grouped_source.get(group)
        qidx = queries.index.to_numpy()
        if past is None:
            past = source.iloc[:0]
        positions = np.searchsorted(past[time].to_numpy(), queries[time].to_numpy(), side="left")
        for column in columns:
            values = past[column].to_numpy(dtype=float)
            finite = np.isfinite(values)
            clean = np.where(finite, values, 0.)
            sums = np.r_[0., np.cumsum(clean)]
            squares = np.r_[0., np.cumsum(clean ** 2)]
            counts = np.r_[0, np.cumsum(finite)]
            def put(suffix, value):
                name = f"{column}__{suffix}"
                if name not in result:
                    result[name] = np.full(len(X), np.nan)
                result[name][qidx] = value
            for lag in lags:
                take = positions - lag
                val = np.full(len(queries), np.nan)
                ok = take >= 0
                val[ok] = values[take[ok]]
                put(f"lag_{lag}", val)
            means, stds = {}, {}
            for window in windows:
                lo = np.maximum(0, positions - window)
                count = counts[positions] - counts[lo]
                total = sums[positions] - sums[lo]
                mean = np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)
                second = np.divide(squares[positions] - squares[lo], count,
                                   out=np.full(len(count), np.nan), where=count > 0)
                std = np.sqrt(np.maximum(0., second - mean ** 2))
                means[window], stds[window] = mean, std
                put(f"roll_mean_{window}", mean)
                put(f"roll_std_{window}", std)
                put(f"roll_sum_{window}", np.where(count > 0, total, np.nan))
            if windows:
                long = max(windows)
                last = np.full(len(queries), np.nan)
                ok = positions > 0
                last[ok] = values[positions[ok] - 1]
                put("reversion", last - means[long])
                put("zscore", np.divide(last - means[long], stds[long],
                                        out=np.full(len(last), np.nan), where=stds[long] > 1e-12))
                put("momentum", means[min(windows)] - means[long])
    return pd.DataFrame(result, index=X.index)


class FeatureBuilder(TransformerMixin, BaseEstimator):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        d, f = self.config["data"], self.config.get("features", {})
        if d.get("target") in X:
            raise ValueError("Remove target from X before feature engineering")
        if not X.index.is_unique or X.columns.duplicated().any():
            raise ValueError("FeatureBuilder needs unique row indices and column names")
        sources = [column for columns in f.get("histories", {}).values() for column in columns]
        sources += [column for pair in f.get("interactions", []) for column in pair]
        for block in ["temporal", "cross_sectional", "group_statistics"]:
            sources += f.get(block, {}).get("columns", [])
        forbidden = set(d.get("drop_columns", []) + d.get("id_columns", []))
        forbidden.update([d.get("target"), self.config["validation"].get("label_end_column")])
        if forbidden.intersection(sources):
            raise ValueError(f"Feature sources include excluded metadata/unavailable columns: {sorted(forbidden.intersection(sources))}")
        metadata = set(d.get("id_columns", []) + d.get("drop_columns", []))
        metadata.update([d.get("time_column"), d.get("group_column"),
                         self.config["validation"].get("label_end_column"),
                         self.config["validation"].get("period_column")])
        self.raw_columns_ = d.get("feature_columns")
        if self.raw_columns_ is None:
            self.raw_columns_ = [c for c in X if c not in metadata]
        else:
            self.raw_columns_ = list(self.raw_columns_)
            if set(self.raw_columns_) & metadata:
                raise ValueError("feature_columns includes an ID, time, group, label-end or explicitly dropped column")
        self.group_tables_ = []
        g = f.get("group_statistics", {})
        if g.get("columns"):
            if not g.get("keys"):
                raise ValueError("group_statistics requires keys")
            if X[g["keys"]].isna().any().any():
                raise ValueError("Group statistic keys cannot be missing")
            for column in g["columns"]:
                for stat in ["mean", "std", "count"]:
                    name = f"{column}__group_{stat}"
                    table = X.groupby(g["keys"], dropna=False)[column].agg(stat).rename(name)
                    self.group_tables_.append(table)
        temporal = f.get("temporal", {})
        self.history_ = None
        if temporal.get("columns"):
            if not d.get("time_column"):
                raise ValueError("Temporal features require an explicit time_column")
            if temporal.get("availability", "frozen") not in {"frozen", "online"}:
                raise ValueError("availability must be frozen or online")
            columns = temporal.get("entity_columns", []) + [d["time_column"]] + temporal["columns"]
            self.history_ = X[list(dict.fromkeys(columns))].copy()
        self.required_columns_ = list(X.columns)
        self.feature_names_out_ = self._build(X, training=True).columns.to_numpy()
        if not len(self.feature_names_out_):
            raise ValueError("No features selected")
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self._build(X, training=True)

    def transform(self, X):
        result = self._build(X, training=False)
        if list(result.columns) != list(self.feature_names_out_):
            raise ValueError("Engineered feature schema changed between training and prediction")
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    def _build(self, X, training):
        f, d = self.config.get("features", {}), self.config["data"]
        if d.get("target") in X:
            raise ValueError("Target cannot enter FeatureBuilder")
        out = X[self.raw_columns_].copy()
        extras = {}
        _positive_ints(f.get("windows", []), "history windows")
        for name, columns in f.get("histories", {}).items():
            if not columns or len(set(columns)) != len(columns):
                raise ValueError("Histories must be nonempty lists of distinct columns, newest first")
            hist = X[columns].astype(float).replace([np.inf, -np.inf], np.nan)
            means = {}
            for window in f.get("windows", []):
                if window > len(columns):
                    continue
                values = hist.iloc[:, :window]
                mean, std = values.mean(axis=1), values.std(axis=1, ddof=0)
                means[window] = mean
                extras[f"{name}__mean_{window}"] = mean
                extras[f"{name}__vol_{window}"] = std
                extras[f"{name}__hit_{window}"] = values.gt(0).sum(axis=1) / values.count(axis=1).replace(0, np.nan)
                extras[f"{name}__reversion_{window}"] = hist.iloc[:, 0] - mean
                extras[f"{name}__z_{window}"] = (hist.iloc[:, 0] - mean) / std.where(std > 1e-12)
            if means:
                extras[f"{name}__momentum"] = means[min(means)] - means[max(means)]
            extras[f"{name}__range"] = hist.max(axis=1) - hist.min(axis=1)
        for a, b in f.get("interactions", []):
            extras[f"{a}__times__{b}"] = X[a] * X[b]
        if extras:
            out = self._append(out, pd.DataFrame(extras, index=X.index))
        temporal = f.get("temporal", {})
        if temporal.get("columns"):
            out = self._append(out, _temporal_features(X, temporal, d["time_column"],
                                                      history=None if training else self.history_))
        cs = f.get("cross_sectional", {})
        if cs.get("columns"):
            if not cs.get("available_at_prediction") or not cs.get("keys"):
                raise ValueError("Cross sections require keys and available_at_prediction=true")
            if X[cs["keys"]].isna().any().any():
                raise ValueError("Cross-section keys cannot be missing")
            for column in cs["columns"]:
                # Raw or deterministic row-history summaries can supply cross sections.
                values = out[column] if column in out else X[column]
                group = values.groupby([X[k] for k in cs["keys"]], dropna=False)
                mean, std = group.transform("mean"), group.transform("std")
                out = self._append(out, pd.DataFrame({
                    f"{column}__cs_mean": mean,
                    f"{column}__cs_rank": group.rank(pct=True),
                    f"{column}__cs_relative": values - mean,
                    f"{column}__cs_z": (values - mean) / std.where(std > 1e-12),
                }, index=X.index))
        g = f.get("group_statistics", {})
        for table in self.group_tables_:
            mapped = X[g["keys"]].join(table, on=g["keys"])[table.name]
            out = self._append(out, mapped.to_frame())
        out = out.drop(columns=f.get("drop_raw_columns", []), errors="raise")
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _append(frame, extra):
        if set(frame.columns) & set(extra.columns):
            raise ValueError("Engineered name collides with an existing feature")
        return pd.concat([frame, extra], axis=1)
