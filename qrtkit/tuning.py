"""Sequential Optuna search with a soft wall-clock limit and fold-level pruning."""
from dataclasses import dataclass
import time
from .metrics import get_metric
from .training import train_cv


@dataclass
class TuningResult:
    best_params: dict
    best_score: float
    trials: object
    seconds: float


def suggest(trial, model):
    if model == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if model == "logistic":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if model in {"lasso", "elasticnet"}:
        params = {"alpha": trial.suggest_float("alpha", 1e-4, 10., log=True)}
        if model == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", .05, .95)
        return params
    if model == "huber":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1., log=True),
                "epsilon": trial.suggest_float("epsilon", 1.1, 2.)}
    if model == "histgb":
        return {"max_iter": trial.suggest_int("max_iter", 60, 240),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 25),
                "l2_regularization": trial.suggest_float("l2_regularization", .1, 10., log=True)}
    if model in {"lightgbm", "xgboost", "catboost"}:
        params = {"learning_rate": trial.suggest_float("learning_rate", .02, .15, log=True)}
        if model == "catboost":
            params.update(iterations=trial.suggest_int("iterations", 60, 300),
                          depth=trial.suggest_int("depth", 2, 5),
                          l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1., 20., log=True))
        else:
            params.update(n_estimators=trial.suggest_int("n_estimators", 60, 300),
                          max_depth=trial.suggest_int("max_depth", 2, 5),
                          reg_lambda=trial.suggest_float("reg_lambda", .1, 20., log=True))
            if model == "lightgbm":
                params["num_leaves"] = trial.suggest_int("num_leaves", 5, 15)
        return params
    raise ValueError(f"No search space for {model}; pass a custom search_space(trial)")


def tune(X, y, config, model_name=None, seconds=None, max_trials=None, metric=None, search_space=None):
    import optuna
    settings = config.get("tuning", {})
    model = model_name or settings.get("model", config["models"][0])
    seconds = seconds if seconds is not None else settings.get("seconds", 120)
    max_trials = max_trials if max_trials is not None else settings.get("max_trials", 20)
    if seconds <= 0 or max_trials < 1:
        raise ValueError("Tuning requires positive seconds and max_trials")
    scorer = get_metric(metric or config["metric"], config["task"])
    start = time.monotonic()
    study = optuna.create_study(direction="maximize" if scorer.greater_is_better else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=config["seed"]),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    def objective(trial):
        params = search_space(trial) if search_space else suggest(trial, model)
        # Keep the actual dictionary: custom spaces may return fixed parameters too.
        trial.set_user_attr("model_parameters", params)
        def callback(fold, scores):
            trial.report(sum(scores) / len(scores), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        try:
            result = train_cv(X, y, None, config, model, params=params, metric=scorer,
                              track=False, progress=False, deadline=start + seconds, fold_callback=callback)
        except TimeoutError as exc:
            raise optuna.TrialPruned(str(exc)) from exc
        return result.score
    study.optimize(objective, n_trials=max_trials, timeout=seconds, n_jobs=1)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise TimeoutError("No complete CV trial fit the budget; increase seconds or reduce folds/trees")
    return TuningResult(study.best_trial.user_attrs["model_parameters"], study.best_value,
                        study.trials_dataframe(), time.monotonic() - start)
