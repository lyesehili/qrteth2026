"""One shared JSON configuration; all paths are relative to the config file."""
from copy import deepcopy
import json
from pathlib import Path


def load_config(path="config.json"):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for section, names in {
        "data": ["train_path", "test_path", "target_path"],
        "submission": ["template_path"],
    }.items():
        for name in names:
            value = config.get(section, {}).get(name)
            if value:
                config[section][name] = str((path.parent / value).resolve())
    config["output_dir"] = str((path.parent / config.get("output_dir", "outputs")).resolve())
    validate_config(config)
    return config


def validate_config(config):
    if config["task"] not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression; choose explicitly")
    if config["validation"]["strategy"] == "temporal_holdout":
        if config["validation"]["n_splits"] != 1:
            raise ValueError("temporal_holdout produces one split; set n_splits=1")
    elif config["validation"]["n_splits"] < 2:
        raise ValueError("At least two validation folds are required")
    if config.get("threads", 2) < 1:
        raise ValueError("threads must be a positive CPU thread count")
    if not 0 <= config.get("metric", {}).get("threshold", 0.5) <= 1:
        raise ValueError("classification probability threshold must be in [0, 1]")


def with_updates(config, **sections):
    """Convenience for notebook experiments; top-level sections are replaced."""
    result = deepcopy(config)
    result.update(sections)
    validate_config(result)
    return result
