# Validation results

Validated locally on 2026-11-24 in the environment recorded in
[TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md). Synthetic results demonstrate
functional behavior; they do not estimate performance at the hackathon.

## Automated checks

`python -m unittest discover -s tests -v`: **30 tests passed**, approximately
25 seconds on this machine. This includes many individual model/fold fits:

- Binary classification: Ridge, Logistic Regression, HistGradientBoosting,
  LightGBM, XGBoost and CatBoost; all three boosting libraries actually installed
  and exercised on CPU.
- Regression: OLS, Ridge, Lasso, ElasticNet, Huber, HistGradientBoosting,
  LightGBM, XGBoost and CatBoost; RMSE direction and finite predictions checked.
- Multiclass: Ridge, logistic and all three boosters, probability normalization,
  class mapping, probability/label submissions and the complete CLI workflow.
- Financial panel: row histories, strictly past lags/rolling features,
  contemporaneous sector cross sections, training-fitted group maps, clipping,
  scaling, PCA, KMeans, forward CV, partial OOF coverage, blending and submission.
- Independent, stratified, grouped, stratified-group and temporal splits;
  grouped training exercised end to end, timestamp gaps, label-horizon and
  irregular label-end purging, bounded training periods, string-date rejection
  and explicitly configured UTC date parsing.
- Leakage checks: future changes cannot affect earlier lags, no cross-entity
  lag spillover, frozen versus online predictions differ as specified,
  validation/test values cannot change fitted imputation/PCA state, group maps
  use training covariates only, random folds with temporal features and split
  cross sections fail, target/unavailable/ID/label-end feature sources rejected.
- Alignment checks: shuffled target indices, absent training classes,
  mismatched ensemble folds/rows, changed test order, invalid weights, template
  ID mismatch, output schema mismatch and submission overwrites rejected.
- Arbitrary target/column names, separate labels joined by ID, missing numeric
  values, wholly missing columns, unseen categories and unseen entities.
- EDA, Pearson/Spearman/MI, feature clustering, signal bins, group IC, drift,
  a known lead-lag relationship, pair hedge/cointegration and transaction costs.
- Classification/regression Optuna searches, metric direction, custom metrics,
  natural metric-domain errors and a deliberately exhausted tuning budget.
- Independent local tracking directories, config/version/hash records and
  reading summaries back into a leaderboard.
- Whole-period price forecasts from one historical cutoff, complete final-day
  holdouts across the stock panel, per-stock persistence and damped log drift,
  unseen-stock/future-price rejection, row order and exact forecast arithmetic.
  A reduced-bars synthetic panel exercises 386 stocks across six countries and
  a full 199-observed-day holdout without using any held-out prices as inputs.

Undefined-AUC and constant-signal warnings in the test log are intentional
edge cases; undefined metrics raise instead of silently returning a usable score.

`python scripts/check_notebooks.py`: both notebooks execute top to bottom with
real Jupyter kernels. Clean source notebooks contain no outputs. Executed copies
are under ignored `outputs/notebook_checks/` for inspection. The engineering
notebook includes a short actual Optuna search and writes actual submissions.

## Complete CLI smoke runs

All four `python -m qrtkit --demo ...` commands produced per-model and equal-weight
blend submissions with checked row IDs. Default seed 42, three folds:

| Synthetic demo | Model | OOF metric | Value | Coverage |
|---|---|---|---:|---:|
| Binary classification | Logistic | ROC AUC | .898779 | 100% |
| Binary classification | LightGBM | ROC AUC | .950982 | 100% |
| Binary classification | Equal-weight blend | ROC AUC | .943890 | 100% |
| Regression | Ridge | RMSE | 17.552229 | 100% |
| Regression | LightGBM | RMSE | 48.949321 | 100% |
| Regression | Equal-weight blend | RMSE | 29.840413 | 100% |
| Financial panel | Logistic | Accuracy | .791667 | 75% |
| Financial panel | LightGBM | Accuracy | .754630 | 75% |
| Financial panel | Equal-weight blend | Accuracy | .777778 | 75% |
| Multiclass | Logistic | Macro OVR ROC AUC | .832164 | 100% |
| Multiclass | LightGBM | Macro OVR ROC AUC | .855929 | 100% |
| Multiclass | Equal-weight blend | Macro OVR ROC AUC | .862528 | 100% |

The regression example deliberately shows why a blend is a separate candidate:
it can be worse than the simpler linear model. Financial coverage is below 100%
because the initial training interval has no preceding fold model; missing OOF
rows are excluded, never filled with artificial predictions.

## Repository integrity and limits

All three reference `git status --porcelain` checks were empty after inspection.
No training datasets, model binaries, submissions or notebook outputs are
included in the source deliverables. `.gitignore` excludes generated data,
outputs, serialized models and cached dependency wheels. Syntax compilation and
`git diff --check` pass. No commit or push was made.

The proprietary historical training datasets were not available, so their scores
were not reproduced. Predictor availability and the official evaluation protocol
must still be established from the new challenge statement. Time limits are soft
around an individual model fit; no universal unbiased score is claimed after
repeated feature/model/threshold selection on the same folds.
