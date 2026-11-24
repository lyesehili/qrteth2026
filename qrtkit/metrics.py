"""Metric values retain their natural sign; direction is explicit for tuning."""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    name: str
    function: Callable
    greater_is_better: bool
    task: str


def labels_from_proba(predictions, threshold=.5):
    p = np.asarray(predictions)
    return (p[:, 1] >= threshold).astype(int) if p.shape[1] == 2 else p.argmax(axis=1)


def get_metric(spec, task):
    """Custom: Metric('my_score', fn(y_encoded, predictions), True, task)."""
    if isinstance(spec, Metric):
        if spec.task != task:
            raise ValueError("Custom metric task disagrees with configuration")
        return spec
    if isinstance(spec, str):
        spec = {"name": spec}
    name, threshold = spec["name"], spec.get("threshold", .5)
    if not 0 <= threshold <= 1:
        raise ValueError("Probability threshold must be in [0, 1]")
    regression = {
        "rmse": (lambda y, p: np.sqrt(skm.mean_squared_error(y, p)), False),
        "mse": (skm.mean_squared_error, False),
        "mae": (skm.mean_absolute_error, False),
        "r2": (skm.r2_score, True),
        "rmsle": (lambda y, p: np.sqrt(skm.mean_squared_log_error(y, p)), False),
        "mape": (skm.mean_absolute_percentage_error, False),
        "pearson": (lambda y, p: pearsonr(y, p).statistic, True),
        "spearman": (lambda y, p: spearmanr(y, p).statistic, True),
        "sign_accuracy": (lambda y, p: skm.accuracy_score(np.asarray(y) > 0, np.asarray(p) > 0), True),
    }
    def auc(y, p):
        return skm.roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else skm.roc_auc_score(
            y, p, labels=np.arange(p.shape[1]), multi_class="ovr", average="macro")
    classification = {
        "accuracy": (lambda y, p: skm.accuracy_score(y, labels_from_proba(p, threshold)), True),
        "balanced_accuracy": (lambda y, p: skm.balanced_accuracy_score(y, labels_from_proba(p, threshold)), True),
        "f1": (lambda y, p: skm.f1_score(y, labels_from_proba(p, threshold), average="binary" if p.shape[1] == 2 else "macro", zero_division=0), True),
        "roc_auc": (auc, True),
        "log_loss": (lambda y, p: skm.log_loss(y, p, labels=np.arange(p.shape[1])), False),
        "average_precision": (lambda y, p: skm.average_precision_score(y, p[:, 1]) if p.shape[1] == 2 else skm.average_precision_score(np.eye(p.shape[1])[y], p, average="macro"), True),
    }
    available = classification if task == "classification" else regression
    if name not in available:
        raise ValueError(f"Unknown {task} metric {name!r}. Choices: {sorted(available)}")
    fn, direction = available[name]
    return Metric(name, fn, direction, task)


def evaluate(y, predictions, metric):
    p = np.asarray(predictions)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Metric received missing, infinite or misaligned predictions")
    try:
        score = float(metric.function(np.asarray(y), p))
    except ValueError as exc:
        raise ValueError(f"Cannot evaluate {metric.name}; check target support in each fold and metric domain: {exc}") from exc
    if not np.isfinite(score):
        raise ValueError(f"{metric.name} is undefined on this fold (constant signal/target or insufficient classes)")
    return score
