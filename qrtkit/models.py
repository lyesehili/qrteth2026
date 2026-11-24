"""CPU estimators behind ordinary sklearn pipelines; optional imports are lazy."""
import importlib.util
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression, LinearRegression, ElasticNet, Lasso, HuberRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from .features import FeatureBuilder


class CleanFrame(TransformerMixin, BaseEstimator):
    def __init__(self, categorical_columns=()):
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        missing = set(self.categorical_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Categorical columns are absent from engineered features: {missing}")
        self.numeric_ = [c for c in X.select_dtypes(include=["number", "bool"]) if c not in self.categorical_columns]
        return self

    def transform(self, X):
        if list(X.columns) != self.columns_:
            raise ValueError("Feature column order/schema changed")
        out = X.copy()
        for c in out:
            if c in self.numeric_:
                out[c] = pd.to_numeric(out[c], errors="raise").astype(float).replace([np.inf, -np.inf], np.nan)
            else:
                # Normalize strings, categories, nullable dtypes and unseen values.
                out[c] = out[c].map(lambda x: str(x) if pd.notna(x) else np.nan).astype(object)
        return out


class QuantileClipper(TransformerMixin, BaseEstimator):
    def __init__(self, quantiles=None):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        if self.quantiles is not None:
            low, high = self.quantiles
            if not 0 <= low < high <= 1:
                raise ValueError("clip_quantiles must be [lower, upper] within [0,1]")
            # Empty columns stay missing until the imputer supplies zero.
            self.bounds_ = np.asarray(pd.DataFrame(X).quantile([low, high]))
        return self

    def transform(self, X):
        if self.quantiles is None:
            return X
        return np.clip(np.asarray(X), self.bounds_[0], self.bounds_[1])


class FactorFeatures(TransformerMixin, BaseEstimator):
    """Append PCA scores and distances to training-fitted clusters to numeric data."""
    def __init__(self, pca_components=0, clusters=0, seed=42):
        self.pca_components, self.clusters, self.seed = pca_components, clusters, seed

    def fit(self, X, y=None):
        self.pca_ = self.kmeans_ = None
        if self.pca_components:
            self.pca_ = PCA(n_components=self.pca_components, random_state=self.seed).fit(X)
        if self.clusters:
            self.kmeans_ = KMeans(n_clusters=self.clusters, n_init=3, random_state=self.seed).fit(X)
        return self

    def transform(self, X):
        blocks = [np.asarray(X)]
        if self.pca_ is not None:
            blocks.append(self.pca_.transform(X))
        if self.kmeans_ is not None:
            blocks.append(self.kmeans_.transform(X))
        return np.column_stack(blocks)


def numeric_columns(X):
    return list(X.select_dtypes(include="number").columns)


def categorical_columns(X):
    return list(X.select_dtypes(exclude="number").columns)


def available_models():
    return {name: importlib.util.find_spec(module) is not None for name, module in
            [("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("catboost", "catboost")]} | {
                "ridge": True, "logistic": True, "linear": True, "lasso": True,
                "elasticnet": True, "huber": True, "histgb": True}


def make_estimator(name, task, params=None, seed=42, threads=2):
    params = dict(params or {})
    classification = task == "classification"
    if name == "ridge":
        cls, defaults = (RidgeClassifier if classification else Ridge), {"alpha": 10.}
    elif name == "logistic":
        if not classification:
            raise ValueError("Use ridge/linear for regression")
        cls, defaults = LogisticRegression, {"C": .1, "max_iter": 600, "random_state": seed}
    elif name in {"linear", "lasso", "elasticnet", "huber"}:
        if classification:
            raise ValueError(f"{name} supports regression; use logistic/ridge for classification")
        cls = {"linear": LinearRegression, "lasso": Lasso, "elasticnet": ElasticNet, "huber": HuberRegressor}[name]
        defaults = {} if name == "linear" else {"max_iter": 1500}
    elif name == "histgb":
        cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        defaults = {"max_iter": 150, "max_leaf_nodes": 15, "l2_regularization": 1.,
                    "random_state": seed, "early_stopping": False}
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if classification else LGBMRegressor
        defaults = {"n_estimators": 200, "max_depth": 4, "num_leaves": 15, "learning_rate": .05,
                    "min_child_samples": 30, "reg_lambda": 1., "verbosity": -1, "random_state": seed}
        params.update(n_jobs=threads, device_type="cpu")
    elif name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if classification else XGBRegressor
        defaults = {"n_estimators": 200, "max_depth": 3, "learning_rate": .05,
                    "reg_lambda": 2., "random_state": seed}
        params.update(n_jobs=threads, tree_method="hist", device="cpu")
    elif name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if classification else CatBoostRegressor
        defaults = {"iterations": 200, "depth": 4, "learning_rate": .05, "verbose": False,
                    "allow_writing_files": False, "random_seed": seed}
        params.update(thread_count=threads, task_type="CPU")
    else:
        raise ValueError(f"Unknown model: {name}")
    return cls(**(defaults | params))


def make_pipeline(config, model_name, params=None):
    p, seed = config.get("preprocessing", {}), config["seed"]
    numeric = Pipeline([
        ("clip", QuantileClipper(p.get("clip_quantiles"))),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler() if p.get("scale", True) else "passthrough"),
        ("factors", FactorFeatures(p.get("pca_components", 0), p.get("clusters", 0), seed)),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 min_frequency=p.get("min_category_frequency", 2),
                                 max_categories=p.get("max_categories", 64))),
    ])
    return Pipeline([
        ("features", FeatureBuilder(config)),
        ("clean", CleanFrame(config["data"].get("categorical_columns", []))),
        ("preprocess", ColumnTransformer([("numeric", numeric, numeric_columns),
                                           ("categorical", categorical, categorical_columns)])),
        ("model", make_estimator(model_name, config["task"], params, seed, config.get("threads", 2))),
    ])


def predict(model, X, task, n_classes=0):
    if task == "regression":
        return np.asarray(model.predict(X)).reshape(-1)
    estimator = model.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        raw = model.predict_proba(X)
    else:
        # RidgeClassifier scores are converted to uncalibrated probability proxies.
        decision = model.decision_function(X)
        raw = np.column_stack([1 - expit(decision), expit(decision)]) if decision.ndim == 1 else softmax(decision, axis=1)
    result = np.zeros((len(X), n_classes))
    result[:, np.asarray(estimator.classes_, dtype=int)] = raw
    if not np.isfinite(result).all() or (result < 0).any() or (result.sum(axis=1) <= 0).any():
        raise ValueError("Estimator produced invalid class probabilities")
    # Some boosters return float32 rows whose sum differs slightly from one.
    return result / result.sum(axis=1, keepdims=True)
