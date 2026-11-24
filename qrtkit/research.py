"""Fast exploratory reports. Run supervised discovery on training/discovery rows.

Looking at validation labels while choosing features is model selection, even
when the transformation itself has no target input. Recheck on untouched folds.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from .data import check_xy


def eda(X, y=None, task=None, max_rows=20000, seed=42):
    sample = X.sample(min(len(X), max_rows), random_state=seed)
    numeric = sample.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    profile = pd.DataFrame({"dtype": X.dtypes.astype(str), "missing_fraction": X.isna().mean(),
                            "unique": X.nunique(dropna=False), "constant": X.nunique(dropna=False) <= 1})
    profile["infinite_sample"] = 0
    for c in numeric:
        profile.loc[c, "infinite_sample"] = int(np.isinf(sample[c].astype(float)).sum())
    result = {"shape": X.shape, "memory_mb": float(X.memory_usage(deep=True).sum() / 1e6),
              "duplicate_rows": int(X.duplicated().sum()), "profile": profile,
              "numeric_summary_sample": numeric.describe(percentiles=[.01, .5, .99]).T if len(numeric.columns) else pd.DataFrame()}
    if y is not None:
        check_xy(X, y)
        result["target"] = pd.Series(y).value_counts(dropna=False) if task == "classification" else pd.Series(y).describe()
    return result


def correlations(X, max_rows=10000, max_columns=200, seed=42):
    x = X.select_dtypes(include="number").iloc[:, :max_columns]
    x = x.sample(min(len(x), max_rows), random_state=seed).replace([np.inf, -np.inf], np.nan)
    return {"pearson": x.corr(method="pearson"), "spearman": x.corr(method="spearman")}


def signal_table(X, y, task="regression", max_rows=5000, seed=42):
    y = check_xy(X, y)
    if len(X) < 5:
        raise ValueError("Signal analysis needs at least five observations")
    x = X.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    idx = np.random.default_rng(seed).choice(len(X), min(len(X), max_rows), replace=False)
    x, target = x.iloc[idx].copy(), y[idx]
    x = x.fillna(x.median()).fillna(0.)
    if not len(x.columns):
        return pd.DataFrame(columns=["pearson", "spearman", "mutual_information", "abs_spearman"])
    if task == "classification":
        encoder = LabelEncoder().fit(target)
        target = encoder.transform(target)
        if np.bincount(target).min() < 2:
            raise ValueError("MI needs at least two sampled observations per class")
        mi = mutual_info_classif(x, target, random_state=seed, n_neighbors=min(3, len(x)-1))
        ordinal = len(encoder.classes_) > 2
    else:
        target = np.asarray(target, dtype=float)
        mi = mutual_info_regression(x, target, random_state=seed, n_neighbors=min(3, len(x)-1))
        ordinal = False
    ys = pd.Series(target, index=x.index)
    result = pd.DataFrame({"pearson": np.nan if ordinal else x.corrwith(ys, method="pearson"),
                           "spearman": np.nan if ordinal else x.corrwith(ys, method="spearman"),
                           "mutual_information": mi}, index=x.columns)
    result["abs_spearman"] = result.spearman.abs()
    return result.sort_values("mutual_information", ascending=False)


def feature_clusters(X, max_clusters=10, max_columns=200):
    """Cluster redundant features using 1-|Spearman|; discovery-only diagnostic."""
    numeric = X.select_dtypes(include="number").iloc[:, :max_columns].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if len(numeric.columns) < 2:
        return pd.Series(1, index=numeric.columns, name="cluster")
    corr = numeric.corr(method="spearman").fillna(0.)
    distance = np.clip(1 - corr.abs().to_numpy(), 0., 1.)
    np.fill_diagonal(distance, 0.)
    tree = linkage(squareform((distance + distance.T) / 2, checks=False), method="average")
    return pd.Series(fcluster(tree, max_clusters, criterion="maxclust"), index=numeric.columns, name="cluster")


def information_coefficient(signal, target, groups, method="spearman"):
    """Per-date/group IC; small or constant cross sections produce missing IC."""
    frame = pd.DataFrame({"signal": signal, "target": target, "group": groups})
    values = {}
    for key, group in frame.groupby("group"):
        valid = group[["signal", "target"]].dropna()
        values[key] = valid.signal.corr(valid.target, method=method) if len(valid) >= 3 and valid.nunique().min() > 1 else np.nan
    ic = pd.Series(values, name="ic")
    std = ic.std()
    return {"per_group": ic, "mean": ic.mean(), "std": std,
            "ic_ir": ic.mean() / std if std > 0 else np.nan, "n_groups": int(ic.notna().sum())}


def signal_bins(signal, target, bins=10):
    frame = pd.DataFrame({"signal": signal, "target": target}).dropna()
    frame["bin"] = pd.qcut(frame.signal, bins, duplicates="drop")
    return frame.groupby("bin", observed=True).agg(count=("target", "size"), mean_target=("target", "mean"), mean_signal=("signal", "mean"))


def lead_lag(X, columns=None, lags=(1, 2, 5), method="spearman"):
    """X must be a single, chronologically sorted wide panel. Positive lag:
    corr(leader[t-lag], follower[t]). Never shift across entities implicitly.
    """
    if not X.index.is_unique or not X.index.is_monotonic_increasing:
        raise ValueError("Lead-lag input needs a unique, increasing time index")
    if not (pd.api.types.is_numeric_dtype(X.index.dtype) or isinstance(X.index, pd.DatetimeIndex)):
        raise ValueError("Use a verified numeric or datetime index, not anonymized date strings")
    if any(not isinstance(lag, int) or lag < 1 for lag in lags):
        raise ValueError("Lead-lag requires positive observation lags")
    columns = list(columns) if columns is not None else list(X.select_dtypes(include="number"))
    records = []
    for leader in columns:
        for follower in columns:
            for lag in lags:
                pair = pd.concat([X[leader].shift(lag), X[follower]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
                corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method) if len(pair) >= 3 and pair.nunique().min() > 1 else np.nan
                records.append({"leader": leader, "follower": follower, "lag": lag,
                                "correlation": corr, "n": len(pair)})
    return pd.DataFrame(records)


def drift_report(train, test, max_rows=10000, seed=42):
    """Univariate train/test shift diagnostics; KS p-values are exploratory."""
    a = train.sample(min(len(train), max_rows), random_state=seed)
    b = test.sample(min(len(test), max_rows), random_state=seed)
    records = []
    for c in a.columns.intersection(b.columns):
        row = {"feature": c, "missing_train": a[c].isna().mean(), "missing_test": b[c].isna().mean()}
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x = a[c].replace([np.inf, -np.inf], np.nan).dropna()
            y = b[c].replace([np.inf, -np.inf], np.nan).dropna()
            row["ks_statistic"] = ks_2samp(x, y).statistic if len(x) and len(y) else np.nan
        else:
            x, y = a[c].astype(str).value_counts(normalize=True), b[c].astype(str).value_counts(normalize=True)
            labels = x.index.union(y.index)
            row["total_variation"] = .5 * (x.reindex(labels, fill_value=0) - y.reindex(labels, fill_value=0)).abs().sum()
        records.append(row)
    return pd.DataFrame(records).set_index("feature")
