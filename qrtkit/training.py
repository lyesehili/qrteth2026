"""Person 2's single CV entry point, OOF predictions, and fixed-weight ensembles."""
from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
from .data import check_xy
from .metrics import get_metric, evaluate, Metric
from .models import make_pipeline, predict
from .validation import make_splits, check_feature_splits, split_fingerprint
from .tracking import frame_fingerprint, save_run


@dataclass
class CVResult:
    model_name: str
    task: str
    metric: Metric
    oof: np.ndarray
    test_predictions: np.ndarray | None
    fold_ids: np.ndarray
    fold_scores: list
    score: float
    classes: np.ndarray | None
    row_index: pd.Index
    split_hash: str
    target_hash: str
    train_hash: str
    test_hash: str | None
    seconds: float
    parameters: dict = field(default_factory=dict)
    models: list = field(default_factory=list)
    output_path: str | None = None

    @property
    def covered(self):
        return self.fold_ids >= 0


def train_cv(X, y, test, config, model_name=None, params=None, metric=None, track=True, progress=True,
             deadline=None, fold_callback=None):
    started = time.monotonic()
    y = check_xy(X, y)
    task = config["task"]
    if test is not None and (not test.index.is_unique or not len(test)):
        raise ValueError("Test must have a unique index and at least one row")
    name = model_name or config["models"][0]
    parameters = dict(config.get("model_params", {}).get(name, {})) | dict(params or {})
    scorer = get_metric(metric or config["metric"], task)
    classes = None
    if task == "classification":
        encoder = LabelEncoder().fit(y)
        y_encoded, classes = encoder.transform(y), encoder.classes_
        if len(classes) < 2:
            raise ValueError("Classification needs at least two classes")
    else:
        y_encoded = np.asarray(y, dtype=float)
        if not np.isfinite(y_encoded).all():
            raise ValueError("Regression targets must be finite")
    splits = make_splits(X, y_encoded, config)
    check_feature_splits(X, test, splits, config)
    shape = (len(X), len(classes)) if classes is not None else (len(X),)
    oof = np.full(shape, np.nan)
    fold_ids = np.full(len(X), -1, dtype=int)
    test_predictions, scores, models = [], [], []
    with threadpool_limits(limits=config.get("threads", 2)):
        for fold, (tr, va) in enumerate(splits):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("CV time budget exhausted between folds")
            if classes is not None and len(np.unique(y_encoded[tr])) != len(classes):
                raise ValueError(f"Fold {fold} training lacks a class; change folds or collect more history")
            pipeline = make_pipeline(config, name, parameters)
            # Nothing fitted on X[va] or test, including feature group means or factors.
            pipeline.fit(X.iloc[tr], y_encoded[tr])
            p = predict(pipeline, X.iloc[va], task, 0 if classes is None else len(classes))
            score = evaluate(y_encoded[va], p, scorer)
            scores.append(score)
            oof[va], fold_ids[va] = p, fold
            if test is not None:
                test_predictions.append(predict(pipeline, test, task, 0 if classes is None else len(classes)))
            models.append(pipeline)
            if progress:
                print(f"{name} fold {fold + 1}/{len(splits)} {scorer.name}={score:.6f}", flush=True)
            if fold_callback is not None:
                fold_callback(fold, scores)
    covered = fold_ids >= 0
    result = CVResult(name, task, scorer, oof,
                      np.mean(test_predictions, axis=0) if test_predictions else None,
                      fold_ids, scores, evaluate(y_encoded[covered], oof[covered], scorer), classes,
                      X.index.copy(), split_fingerprint(splits), frame_fingerprint(pd.Series(y, index=X.index)),
                      frame_fingerprint(X), None if test is None else frame_fingerprint(test),
                      time.monotonic() - started, parameters, models)
    if track:
        result.output_path = save_run(result, config, X, y, test)
    return result


def ensemble(results, y, weights=None):
    """Average compatible OOF/test predictions. Weights must be chosen separately.

    Tuning weights on these OOF labels makes the blend score optimistic; reserve
    an outer holdout if selecting weights or choosing among many recipes.
    """
    if not results:
        raise ValueError("Provide at least one result")
    first = results[0]
    if isinstance(y, pd.Series) and not y.index.equals(first.row_index):
        raise ValueError("Ensemble target row order disagrees")
    target_hash = frame_fingerprint(pd.Series(np.asarray(y), index=first.row_index))
    if target_hash != first.target_hash:
        raise ValueError("Ensemble targets differ from training targets")
    for r in results[1:]:
        if (r.task != first.task or r.split_hash != first.split_hash or r.target_hash != first.target_hash
                or r.train_hash != first.train_hash or r.test_hash != first.test_hash
                or not r.row_index.equals(first.row_index) or not np.array_equal(r.classes, first.classes)
                or not np.array_equal(r.fold_ids, first.fold_ids)):
            raise ValueError("Ensembles require identical rows, targets, folds, class order and test data")
    w = np.ones(len(results)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(results),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Provide one finite nonnegative weight per model with a positive sum")
    oof = np.average(np.stack([r.oof for r in results]), axis=0, weights=w)
    has_test = [r.test_predictions is not None for r in results]
    if any(has_test) and not all(has_test):
        raise ValueError("All ensemble models must provide test predictions")
    test = np.average(np.stack([r.test_predictions for r in results]), axis=0, weights=w) if all(has_test) else None
    y_encoded = LabelEncoder().fit(first.classes).transform(y) if first.classes is not None else np.asarray(y)
    scores = [evaluate(y_encoded[first.fold_ids == i], oof[first.fold_ids == i], first.metric)
              for i in range(len(first.fold_scores))]
    return CVResult("ensemble", first.task, first.metric, oof, test, first.fold_ids.copy(), scores,
                    evaluate(y_encoded[first.covered], oof[first.covered], first.metric), first.classes,
                    first.row_index.copy(), first.split_hash, first.target_hash, first.train_hash, first.test_hash,
                    sum(r.seconds for r in results), {"models": [r.model_name for r in results], "weights": (w/w.sum()).tolist()})
