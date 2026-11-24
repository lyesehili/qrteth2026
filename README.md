# QRT ETH hackathon toolkit

Two people, four hours, CPU only. Start with a dependable baseline, then spend
time on the problem and signals. The default configuration runs on small
synthetic data immediately. **No challenge-specific schema is assumed.**

New challenge details: five-minute prices for 386 stocks in six countries, with
prices forecast across the final 199 days. See
[the current challenge notes and whole-period price baselines](docs/CHALLENGE_NOTES.md).
The exact metric and data schema are still pending; the default config remains
the generic synthetic demo.

This checkout is named `qrteth2026` and is the shared working repository referred
to as `qrt-eth-2026` in the brief. All toolkit files live here; the three sibling
reference repositories are unchanged.

## Start here

Use Python 3.11 or 3.12. Install dependencies before going offline:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m qrtkit
python -m unittest discover -s tests -v
python -m jupyter lab
```

On this machine an existing environment already contains the dependencies:
`/home/lyes/miniforge3/envs/qrt/bin/python -m qrtkit` works without installation.
The library never downloads data, contacts tracking services or requires a GPU.
If a booster is absent, use `logistic`, `ridge` or `histgb`; optional imports occur
only when that model is selected. The test suite skips missing booster packages.

For an offline installation on the **same OS, CPU architecture and Python version**:

```bash
python -m pip download -r requirements.txt -d wheelhouse
# Transfer the repository and wheelhouse to the competition machine.
python -m pip install --no-index --find-links=wheelhouse -r requirements.txt
```

Do this and run the tests before the event. Dependencies are not vendored here.
The exact environment used for this build is in [docs/TESTED_ENVIRONMENT.md](docs/TESTED_ENVIRONMENT.md).

## Adapt a new dataset in under five minutes

1. **0–1 minute:** put authorized CSV, Parquet or Feather files in `data/`. Look at
   the target, identifiers, timestamps and submission example. Determine which
   inputs are known when each prediction is made.
2. **1–2 minutes:** edit [config.json](config.json). Set `data.demo` to `null`, set
   `train_path`, `test_path`, `target`, and `id_columns` (or `[]` if there are no
   identifiers). For a separate label file set `target_path`; it is joined by ID,
   never assumed to be in row order. Choose `task` explicitly. Paths resolve from
   the config file. IDs must uniquely identify rows across train and test.
3. **2–3 minutes:** choose the official metric and validation strategy. Use
   `standard` for independent regression rows, `stratified` for independent
   classification rows, `group`/`stratified_group` for shared subjects or dates,
   and `temporal` for a real chronological prediction task. Set `time_column` and
   `group_column` as needed. Anonymized date labels are groups, not a reliable
   chronology. Set a temporal gap/horizon if labels overlap.
4. **3–4 minutes:** set `models` to `["logistic", "lightgbm"]` for classification
   or `["ridge", "lightgbm"]` for regression. Set `metric.name`, and configure
   `submission.kind` (`label`, `probability`, or regression `value`),
   `prediction_columns`, and optional `template_path`. Keep all feature blocks
   disabled initially. Use `drop_columns` to exclude other unavailable inputs.
5. **4–5 minutes:** run `python -m qrtkit`. Check fold scores, coverage, class
   mapping, IDs and generated `outputs/<run>/submission.csv`. Open your notebook
   and improve one feature block at a time on the same folds.

Example settings for a continuous response (replace these example names):

```json
{
  "task": "regression",
  "data": {
    "demo": null, "train_path": "data/train.csv", "test_path": "data/test.csv",
    "target_path": null, "target": "response", "id_columns": ["sample_id"],
    "drop_columns": [], "feature_columns": null, "categorical_columns": [], "datetime_columns": [],
    "time_column": null, "group_column": null
  },
  "models": ["ridge", "lightgbm"],
  "metric": {"name": "rmse"},
  "submission": {"kind": "value", "prediction_columns": ["response"], "template_path": null}
}
```

Replace these sections in the complete config; this snippet is not a separate
complete configuration. Set `validation.strategy` to `standard` for this example.
To read chronological date strings, name those columns in `data.datetime_columns`
(including `label_end_column` if used); the loader parses them to UTC in train and
test. Numeric chronological time can be used directly. Never sort anonymized date
strings as time.

## Two responsibilities, one interface

| Person | Open | Owns | Handoff |
|---|---|---|---|
| 1 — Quant Research | [01_quant_research.ipynb](notebooks/01_quant_research.ipynb) | EDA, signal diagnostics, feature recipes, linear baselines, factor and optional pair analysis | `features`/`preprocessing` sections in config, or `outputs/feature_recipe.json` |
| 2 — ML Engineering | [02_ml_engineering.ipynb](notebooks/02_ml_engineering.ipynb) | Split choice, models, budgets, OOF, ensembles, tracking, submission | Fold scores, stable OOF artifacts, validated submission |

Both notebooks run top to bottom using synthetic data. Change the same config to
use real data. Person 2 can point `RECIPE_PATH` at Person 1's exported config.
**Share the recipe and raw data, then refit inside folds.** A matrix produced by a
full-data-fitted imputer, PCA, clusterer or group mapping is unsuitable for CV.

```python
from qrtkit import load_config, load_data, train_cv
from qrtkit.submission import make_submission

cfg = load_config("config.json")
data = load_data(cfg)
result = train_cv(data.train, data.y, data.test, cfg, "lightgbm")
submission = make_submission(result, data.test, cfg)  # inspect before writing
```

The modules are ordinary small functions and sklearn pipelines:

| Module | Purpose |
|---|---|
| `research.py` | Missingness, constants, duplicate rows, sampled distributions, Pearson/Spearman, MI, binned signals, per-group IC, drift, correlation clusters, lead-lag |
| `features.py` | Explicit row histories; entity lags, rolling mean/std/sum, momentum/reversion; contemporaneous ranks and deviations; fold-fitted group statistics; interactions |
| `models.py` | Fold-fitted median imputation, optional quantile clipping, bounded one-hot encoding, scaling, PCA factors and cluster distances, CPU estimators |
| `validation.py` | Standard, stratified, grouped, stratified-group and purged forward temporal CV |
| `training.py` | OOF/test predictions, class mapping, fold models and compatible fixed-weight ensembles |
| `metrics.py`, `tuning.py` | Explicit metric direction, custom scores, sequential Optuna search with time budget |
| `tracking.py`, `submission.py` | Independent local run directories, data/split hashes, ID-aligned submissions |
| `finance.py` | Training-fitted pair hedge/z-scores, optional cointegration, cost-aware signal diagnostic |
| `forecasting.py` | Whole-period per-stock persistence and damped-drift price forecasts; no future prices consumed |

## Feature recipes and validation

Every column is explicit. See [docs/FEATURES_AND_VALIDATION.md](docs/FEATURES_AND_VALIDATION.md)
for recipes, forecasting assumptions, purging, metrics and known limits.
Raw columns are selected by name, never by position. The target, IDs, split group,
time, label-end and explicitly dropped columns are excluded from model inputs.
`feature_columns` can restrict predictors; to model a split group's category,
create a separate predictor column deliberately. Integer categorical features can
be named in `data.categorical_columns`.

Regression baselines: `linear`, `ridge`, `lasso`, `elasticnet`, `huber`, `histgb`,
`lightgbm`, `xgboost`, `catboost`. Classification: `logistic`, `ridge`, `histgb`
and all three boosters. Ridge classification probabilities are uncalibrated score
proxies; use logistic for probability-sensitive metrics. Binary and multiclass
predictions have shape `(rows, classes)`; `result.classes` gives the order.

Classification metrics: `accuracy`, `balanced_accuracy`, `f1`, `roc_auc`,
`average_precision`, `log_loss`. Regression: `rmse`, `mse`, `mae`, `r2`, `rmsle`,
`mape`, `pearson`, `spearman`, `sign_accuracy`. No metric or sign conversion is
inferred from a numeric target. Custom metrics are documented in the guide.

## Budgets and experiments

```bash
python -m qrtkit --tune
python -m qrtkit --demo regression
python -m qrtkit --demo financial
python -m qrtkit --demo multiclass
python scripts/check_notebooks.py
```

`tuning.seconds` is a **soft** wall-clock budget. New folds/trials stop at the
deadline; an active estimator fit finishes. Reduce folds/trees for tight budgets.
Tuning and training are sequential with two CPU threads by default. No validation
fold is used for automatic early stopping. A custom Optuna search-space function
can be supplied in Python. `--tune` tunes only `tuning.model` and then trains it.

Each model run writes `config.json`, `summary.json`, `oof.csv`,
`test_predictions.csv` and a submission under its own directory. Summaries include
versions, git revision, parameters, timings, data hashes, split hash, fold scores
and OOF coverage. `tracking.leaderboard()` reads these files; compare like-for-like
metrics and split hashes. Models remain in memory (`result.models`), and are not
serialized automatically. Generated outputs, data, models and wheel caches are
git-ignored. There is no external tracking service.

Equal-weight ensembles are generated separately from individual submissions.
Choose one based on sound validation; blends are not guaranteed to improve.
Tuning parameters, thresholds or weights against the same OOF labels makes the
reported selection score optimistic. Keep an untouched holdout when possible.

## Four-hour working rhythm

| Time | Person 1 | Person 2 |
|---|---|---|
| 0–20 min | Understand target, availability, groups and time | Confirm official metric, split and a valid baseline submission |
| 20–90 min | Discovery-only diagnostics; test 2–3 feature hypotheses | Compare linear/boosted baselines on fixed folds |
| 90–170 min | Check signal stability and useful feature ablations | Short tuning runs, OOF error analysis, simple blends |
| 170–210 min | Review shift and assumptions together | Select using validation and robustness |
| 210–240 min | Audit feature availability | Verify template IDs, class mapping, file format; retain a backup submission |

The review of the three references, their limitations and the methods retained
is in [docs/REFERENCE_REVIEW.md](docs/REFERENCE_REVIEW.md). None has an attached
license in the inspected checkout. The toolkit contains independently written
implementations of general statistical methods, without copied source or data.
