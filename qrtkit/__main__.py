"""python -m qrtkit --config config.json [--tune]"""
import argparse
from copy import deepcopy
from pathlib import Path
from .config import load_config
from .data import load_data
from .models import available_models
from .training import train_cv, ensemble
from .tracking import save_run
from .submission import make_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--tune", action="store_true", help="Tune the configured tuning.model, then train it")
    parser.add_argument("--demo", choices=["classification", "regression", "financial", "multiclass"])
    args = parser.parse_args()
    c = load_config(args.config)
    if args.demo:
        c = deepcopy(c)
        c["data"]["demo"] = args.demo
        if args.demo == "financial":
            from .synthetic import financial_config
            c = financial_config(c)
        elif args.demo == "regression":
            c["task"] = "regression"
            c["validation"]["strategy"] = "standard"
            c["models"] = ["ridge", "lightgbm"]
            c["metric"]["name"] = "rmse"
            c["submission"]["kind"] = "value"
        elif args.demo == "multiclass":
            c["task"] = "classification"
            c["submission"].update(kind="probability", prediction_columns=["down", "flat", "up"])
    data = load_data(c)
    print("Available models:", available_models(), flush=True)
    if args.tune:
        from .tuning import tune
        tuned = tune(data.train, data.y, c)
        name = c["tuning"]["model"]
        c["model_params"][name] = c.get("model_params", {}).get(name, {}) | tuned.best_params
        c["models"] = [name]
    results = [train_cv(data.train, data.y, data.test, c, name) for name in c["models"]]
    for result in results:
        path = Path(result.output_path) / "submission.csv"
        make_submission(result, data.test, c, path)
        print(f"{result.model_name}: {result.metric.name}={result.score:.6f}, OOF coverage={result.covered.mean():.1%}; {path}")
    if len(results) > 1:
        blend = ensemble(results, data.y)
        blend.output_path = save_run(blend, c, data.train, data.y, data.test)
        make_submission(blend, data.test, c, Path(blend.output_path) / "submission.csv")
        print(f"Equal-weight blend: {blend.metric.name}={blend.score:.6f}; {blend.output_path}")


if __name__ == "__main__":
    main()
