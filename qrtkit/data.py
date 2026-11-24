"""Load named schema, preserve explicit row identity, never guess a target."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".feather"}:
        return pd.read_feather(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Use CSV, Parquet or Feather: {path}")


@dataclass
class DataBundle:
    train: pd.DataFrame
    y: pd.Series
    test: pd.DataFrame


def load_data(config):
    d = config["data"]
    if d.get("demo"):
        from .synthetic import make_demo
        bundle = make_demo(d["demo"], seed=config["seed"])
        if (d["demo"] == "regression") != (config["task"] == "regression"):
            raise ValueError("Demo and configured task disagree")
        return bundle
    if not d.get("target") or not d.get("train_path") or not d.get("test_path"):
        raise ValueError("Set demo=null, train_path, test_path and target in config.json")
    train, test = read_table(d["train_path"]), read_table(d["test_path"])
    target = d["target"]
    ids = d.get("id_columns", [])
    if d.get("target_path"):
        labels = read_table(d["target_path"])
        if not ids:
            raise ValueError("Separate labels require id_columns for verified alignment")
        if target in train:
            raise ValueError("Target appears in train and separate target file")
        train = train.merge(labels[ids + [target]], on=ids, how="left", validate="one_to_one", sort=False)
    if target not in train or target in test:
        raise ValueError("Target must be present in train and absent from test")
    for column in d.get("datetime_columns", []):
        train[column] = pd.to_datetime(train[column], utc=True, errors="raise")
        if column in test:
            test[column] = pd.to_datetime(test[column], utc=True, errors="raise")
    for name, frame in [("train", train), ("test", test)]:
        if frame.columns.duplicated().any():
            raise ValueError(f"Duplicate {name} column names")
        if ids and (frame[ids].isna().any().any() or frame.duplicated(ids).any()):
            raise ValueError(f"{name} identifiers must be nonmissing and unique")
    if ids:
        overlap = train[ids].merge(test[ids], on=ids, how="inner")
        if len(overlap):
            raise ValueError("Train/test row identifiers overlap")
    y = train.pop(target)
    if y.isna().any():
        raise ValueError("Missing targets or unmatched label identifiers")
    missing = set(train.columns) - set(test.columns)
    # Label availability metadata is only needed for training splits.
    missing.discard(config["validation"].get("label_end_column"))
    missing.discard(config["validation"].get("period_column"))
    if missing:
        raise ValueError(f"Test is missing training columns: {sorted(missing)}")
    return DataBundle(train.reset_index(drop=True), y.reset_index(drop=True), test.reset_index(drop=True))


def check_xy(X, y):
    if not isinstance(X, pd.DataFrame) or not X.index.is_unique:
        raise ValueError("X must be a DataFrame with a unique index")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("X/y indices differ; align explicitly rather than silently reordering")
    if len(X) != len(y) or pd.isna(y).any():
        raise ValueError("X/y length mismatch or missing target")
    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature names")
    return np.asarray(y)
