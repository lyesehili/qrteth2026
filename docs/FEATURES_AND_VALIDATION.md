# Feature and validation guide

## The shared contract

`train_cv(X, y, test, config, model_name)` takes raw pandas frames with unique row
indices and aligned labels. Train and test must have matching predictor names.
Targets are removed by `load_data`; separate labels are verified by an ID join.
All supervised targets must be observed. Missing and infinite numeric predictors
are handled by a training-fitted median imputer. Completely empty columns remain
in the schema and are imputed with zero. Unknown categories are accepted.
`data.feature_columns` selects raw predictors; derived features are controlled
by their recipe blocks. Use `data.datetime_columns` to parse named date fields
in the loader, consistently as UTC timestamps in training and test.

`FeatureBuilder(config)` follows sklearn `fit`, `fit_transform`, `transform`.
It preserves row order and index. It ignores `y`; passing a named target in X is
an error. Its group maps are fitted on training rows. The CV pipeline creates a
fresh builder and preprocessing pipeline per fold.

Person 1 may inspect `FeatureBuilder(cfg).fit_transform(discovery_rows)`, but
Person 2 should receive the **recipe**, not a globally fitted feature matrix.
Supervised feature selection, MI rankings and feature ablations also require a
discovery subset or nested validation if an unbiased final score is needed.
The supplied research notebook uses only the first fold's training rows for its
supervised diagnostics. This preserves that fold's validation labels for a check;
the aggregate CV after manual feature selection is still a selection score.

## Row histories

If a row already contains known history, specify its columns **newest first**:

```json
"histories": {"sensor_history": ["latest", "previous", "older", "oldest"]},
"windows": [2, 4]
```

This adds window means, population volatility, observed-positive fractions,
deviation of newest value from mean, standardized deviation, short-minus-long
mean, and historical range. Windows exceeding available columns are skipped.
All-missing histories remain missing. This is useful for arbitrary histories,
not just returns. These computations do not learn across rows.

For financial returns, rolling sums/means approximate additive momentum; they
are not compounded returns. Do not feed price levels into return formulas
without deciding which interpretation is intended.

## Long-form temporal features

```json
"data": {"time_column": "event_time", "group_column": "entity"},
"temporal": {
  "columns": ["observed_measurement"], "entity_columns": ["entity"],
  "lags": [1, 2, 5], "windows": [3, 10], "availability": "frozen"
}
```

Merge these into the relevant sections of the full config. This block requires
temporal CV and verified numeric/datetime time. Rows are sorted internally and
returned in original order. `(entity_columns, time)` must be unique. With no
entity columns, the data is treated as a single series. Positive lags and rolling
statistics use observations **strictly before the current timestamp**. No
backfill, centered window, future observation or target is used.

- `frozen`: validation/test lags use only the training history remembered by the
  fitted builder. Later validation values never enter another prediction. Use
  this for a batch forecast whose future measurements are not yet observed.
- `online`: earlier validation/test measurements may enter later predictions in
  that same prediction batch. Use this only when those measurements really
  become available between predictions. This is walk-forward prediction with a
  fixed fitted model, not model retraining.

Training features are always computed from earlier training observations.
Lag/window sizes count **observations within an entity**, not calendar durations.
Missing timestamps, purge gaps and withheld rows therefore shorten the available
history: this toolkit does not interpolate or invent missing periods. A fold's
test predictions use its own training history plus the online test batch, not
covariates from other validation folds. Supply a purpose-built adapter if the
challenge requires a full uninterrupted external covariate history.

These guarantees concern lag/rolling features. Raw predictors in the supplied
row must themselves be available at the forecast origin. Exclude unknown future
measurements with `drop_columns` or an explicit `feature_columns` list; the
toolkit cannot infer availability from a column's name or numeric values.

## Cross sections and group mappings

```json
"cross_sectional": {
  "columns": ["observed_measurement"],
  "keys": ["event_time", "sector"],
  "available_at_prediction": true
},
"group_statistics": {"columns": ["liquidity"], "keys": ["sector"]},
"interactions": [["observed_measurement", "liquidity"]]
```

Cross-sectional mean, percentile rank, relative value and z-score use only the
contemporaneous supplied prediction batch. This is valid only if the entire
cross section's predictors are simultaneously available. Entire key groups must
stay in one fold. Random row folds or group-by-instrument folds that split a
timestamp/sector cross section are rejected. Train/test cross-section keys must
also be disjoint. Group by timestamp for anonymized financial dates.

Group statistics are a different operation: mean, standard deviation and count
of a covariate are learned **only from the training fold**, then mapped to
validation/test groups. Unknown groups yield missing values, subsequently
imputed by the fold's imputer. There is deliberately no target encoder: a correct
generic implementation would also need training-row cross-fitting appropriate
to the grouped/temporal split. Add it only if there is time and clear evidence.

Use `drop_raw_columns` inside `features` to remove raw histories after generating
summaries. Keep column names consistent between training and test. Cross-section
inputs may also be deterministic row-history summaries generated by the builder.

## Preprocessing and factors

`preprocessing.clip_quantiles=[0.01,0.99]` learns clipping bounds on the training
fold. Median imputation, optional scaling, PCA components and KMeans centers all
learn only from the training fold. Numeric factors and cluster distances are
appended to numeric predictors; they do not replace them. Set `pca_components`
and `clusters` to zero to disable. Choose counts smaller than the smallest fold's
row/column counts. PCA loadings and cluster geometry depend on scaling, so keep
`scale=true` when using them unless you deliberately want raw-unit geometry.

Categoricals use one-hot encoding with `max_categories=64` per column by default.
This keeps dimensionality bounded for the hackathon; extremely wide input can
still need explicit column selection. Feature-correlation clustering in
`research.feature_clusters` is a discovery diagnostic; model-fitted KMeans
clusters observations. Both use CPU only.

## Splits and label horizons

| Strategy | Appropriate setting | Guarantee |
|---|---|---|
| `standard` | Independent rows | Disjoint random KFold rows |
| `stratified` | Independent classification rows | Approximately balanced class proportions |
| `group` | Shared subject, allocation or anonymized date | No group appears on both sides |
| `stratified_group` | Grouped classification | Group isolation with approximate class balance |
| `temporal` | Known real time order | Expanding forward train windows; timestamps never split |
| `temporal_holdout` | Entire future day/session block | One final holdout of `test_periods` complete date/session blocks; `n_splits=1` |

For `temporal_holdout`, `period_column` can name an explicit common session
identifier. Otherwise datetime timestamps define observed date blocks (UTC for
timezone-aware timestamps). Numeric clocks require an explicit period column.
The gap/horizon still counts distinct timestamps. This split is useful for a
whole-period forecast; see [CHALLENGE_NOTES.md](CHALLENGE_NOTES.md). It does not
make future price-derived predictors available.

Temporal validation applies `gap + label_horizon` **distinct timestamps** between
the final training timestamp and validation start. Example: `gap=1` and
`label_horizon=2` leave three full timestamps out of training before validation.
This is conservative around endpoints. `max_train_periods` optionally bounds the
training history. Use `label_end_column` for irregular horizons: training labels
are retained only when their end is strictly before validation start. Label-end
and time values must use compatible units/dtypes. Label-end metadata is excluded
from the predictor matrix and need not exist in test. IDs, label-end metadata and
`data.drop_columns` are also rejected as numeric sources for engineered features;
use `features.drop_raw_columns` when a known history should supply summaries but
its raw columns should be removed afterward.

Forward-only folds need no post-validation training embargo because no later
training rows exist. This is not a general purged two-sided KFold implementation.
Temporal training entities may recur in validation; it estimates future behavior
of known entities, not unseen-entity generalization. For simultaneous time and
unseen-entity constraints, adapt `make_splits` explicitly and test both invariants.

Temporal CV does not predict the first training block. Its OOF predictions stay
NaN, its fold ID stays `-1`, and only `result.covered` rows enter scores. Never
replace those NaNs by zeros or treat them as genuine OOF observations.

## Metrics, classes and selection

Targets are never converted to signs automatically. To predict sign, transform
the target explicitly **before** training; to regress a return but evaluate its
sign choose `task=regression`, `metric.name=sign_accuracy`. Numeric class labels
are labels, not continuous magnitudes.

Classification classes are sorted by sklearn's LabelEncoder. `result.classes`
records this mapping. Binary metrics use class index 1 as the positive class;
choose/recode your labels explicitly if another class should be positive.
The default binary decision threshold is 0.5, configurable in `metric.threshold`.
Multiclass F1/AP/AUC use macro averaging and AUC uses one-vs-rest.

For a binary probability submission, `submission.positive_class` can name either
class; null means `result.classes[1]`. A multiclass probability submission needs
one prediction column per class, in class order. Label submissions return the
original class labels. Regression submissions use `kind=value`. Templates with
IDs are aligned by those IDs; without IDs the contract is positional and you
must confirm template/test row order. Existing submission files are never
overwritten by the writer.

Metrics retain their natural values (e.g. RMSE is positive) and carry a separate
optimization direction. An undefined metric, such as AUC on a single-class fold,
raises an actionable error instead of silently favoring an invalid trial. RMSLE
requires nonnegative targets/predictions; MAPE is unstable near zero; correlation
is undefined for constants. No clipping of predictions to make a metric valid is
done automatically.

Custom Python metric example:

```python
from qrtkit.metrics import Metric
from qrtkit.training import train_cv
import numpy as np
metric = Metric("weighted_error", lambda y, p: np.mean((p-y)**2 * (1+abs(y))),
                greater_is_better=False, task="regression")
result = train_cv(X, y, test, cfg, "ridge", metric=metric)
```

Custom classification metrics receive integer-encoded labels and a probability
matrix with all classes. A custom function is not serialized; retain its source
in your working code, and record its name in run notes.

OOF and fold-mean scores differ for unequal folds or nonadditive metrics such as
AUC. Neither their difference nor fold standard deviation measures a train-to-
test overfit gap. Tune on stable splits, record all tried recipes, and use an
outer holdout if claiming unbiased performance after substantial selection.

## Financial diagnostics

`lead_lag` expects a verified chronological wide panel with one column per
series. Positive lag compares `leader[t-lag]` with `follower[t]`; it never shifts
through stacked entity rows. Correlation is evidence for a hypothesis, not proof
of predictability or causality. Multiple pairs/lags require out-of-sample checks.

`fit_pair` learns a log-price hedge ratio and spread normalization from the
training period only. Its optional cointegration p-value is exploratory; use
subsequent periods for testing and account for searching many candidate pairs.
`backtest_signal` assumes `signal[t]` is known before the supplied
`forward_returns[t]`. If you pass same-period returns, shift the signal first.
The diagnostic charges turnover costs but does not model financing, borrow,
market impact or overlapping positions. Set annualization to the true frequency.
