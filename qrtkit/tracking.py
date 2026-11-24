"""Local run directories only; no account, server, credentials or model upload."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import uuid
import numpy as np
import pandas as pd


def frame_fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def environment():
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost", "catboost", "optuna"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"python": platform.python_version(), "packages": versions, "git_commit": git.stdout.strip() or None}


def save_run(result, config, X, y, test=None):
    # Independent per-run files avoid two teammates racing on a shared CSV.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + result.model_name + "-" + uuid.uuid4().hex[:8]
    path = Path(config.get("output_dir", "outputs")) / run_id
    path.mkdir(parents=True, exist_ok=False)
    summary = {"run_id": run_id, "model": result.model_name, "metric": result.metric.name,
               "greater_is_better": result.metric.greater_is_better,
               "oof_score": result.score, "fold_scores": result.fold_scores,
               "fold_mean": float(np.mean(result.fold_scores)), "fold_std": float(np.std(result.fold_scores)),
               "coverage": float(result.covered.mean()), "seconds": result.seconds,
               "split_hash": result.split_hash, "target_hash": result.target_hash,
               "train_hash": frame_fingerprint(X), "test_hash": None if test is None else frame_fingerprint(test),
               "classes": None if result.classes is None else result.classes.tolist(),
               "parameters": result.parameters, "environment": environment()}
    (path / "summary.json").write_text(json.dumps(summary, indent=2, default=str, allow_nan=False))
    (path / "config.json").write_text(json.dumps(config, indent=2, default=str))
    names = ["prediction"] if result.task == "regression" else [f"class_{i}" for i in range(len(result.classes))]
    oof = pd.DataFrame(result.oof, columns=names)
    oof.insert(0, "row_index", X.index)
    for col in reversed(config["data"].get("id_columns", [])):
        oof.insert(0, col, X[col].to_numpy())
    oof["fold"] = result.fold_ids
    oof["target"] = np.asarray(y)
    oof.to_csv(path / "oof.csv", index=False)
    if result.test_predictions is not None:
        pred = pd.DataFrame(result.test_predictions, columns=names)
        if test is not None:
            for col in reversed(config["data"].get("id_columns", [])):
                pred.insert(0, col, test[col].to_numpy())
        pred.to_csv(path / "test_predictions.csv", index=False)
    return str(path)


def leaderboard(directory="outputs"):
    """Compare only runs with the same metric, task and split hash."""
    records = [json.loads(p.read_text()) for p in Path(directory).glob("*/summary.json")]
    return pd.DataFrame(records)
