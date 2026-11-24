# Inspection of previous QRT challenge repositories

Inspected the local checkouts before implementation on 2026-11-24. This review
distinguishes code behavior, author-reported results, and our methodological
assessment. Proprietary training data is absent, so historical performance could
not be independently reproduced. Only supplied local files were inspected; no
external dataset was fetched or executed. No reference source was modified.

## Scope and provenance

| Checkout | Origin | Inspected revision |
|---|---|---|
| `QRT_challenge` | [malerbe/QRT_challenge](https://github.com/malerbe/QRT_challenge) | `b24ee375eb9e15d22f1ef4931282fb74a5a6178f` |
| `qrt-data-challenge` | [comemuntuqtb/qrt-data-challenge](https://github.com/comemuntuqtb/qrt-data-challenge) | `29b6333dbb63b6692e0afdb2656174bed0e7d69c` |
| `trust-or-short-QRT-Data-Challenge-2026` | [PatthieuM/trust-or-short-QRT-Data-Challenge-2026](https://github.com/PatthieuM/trust-or-short-QRT-Data-Challenge-2026) | `29105b7716d5aa18c9a7237f0355450ba227129d` |

The audit covered the READMEs, dependency manifests, Python modules, readable
notebook sources, experiment scripts/logs and submission schemas. The inventory
is in [REFERENCE_INVENTORY.json](REFERENCE_INVENTORY.json). Cached bytecode, images,
git internals and previously generated predictions are not implementation sources.
The first repository contains 33 readable notebooks and three empty notebook files
(`dataset_analysis`, `test_markowitz`, `preprocessing/features_markowitz`); the
second contains one notebook; the third has no notebooks. Exact counts are
recorded in the inventory. Reference covariance/portfolio ideas were assessed
from their implemented formulas rather than the general Markowitz resource PDF.

None of the checkouts contains a LICENSE/COPYING/NOTICE file. The third README
explicitly states that no license is attached and data has separate terms. Public
availability does not supply a source-code reuse license. We retained general
methodological ideas, independently implemented them, and copied neither source,
notebook cells, data, weights nor trained models. No license is asserted over
reference material; preserve attribution if further material is incorporated.

## Dataset commonality and what remains unknown

These are three approaches to closely related allocation-performance forecasting,
not evidence that the ETH event will use this exact task. A row is a date and
allocation, with approximately twenty previous returns, twenty signed volumes,
turnover and group metadata. The eventual target is the next return's sign;
some implementations regress the return before thresholding at zero.

| Repository | Data evidence | Target / evaluation |
|---|---|---|
| `QRT_challenge` | README reports 180,245 training rows, 20-day windows; training data absent; code commonly expects `AVG_DAILY_TURNOVER` | Sign of next-day allocation return; binary accuracy; README reports rank 50/104 |
| `qrt-data-challenge` | README reports 527,073 training rows, 278 allocations, 2,522 timestamps; local test CSV actually contains 31,870 rows and 46 columns including `ROW_ID`; local training labels absent | Next-day sign; accuracy; README reports public 0.5160, rank 487/1,017 |
| `trust-or-short...` | Same 2026 allocation schema, raw data absent; data README requests authorized download; full results are aggregate logs only | Next-session sign; accuracy; tracker reports best public 0.5187 |

The local second-repository test headers are `ROW_ID`, `TS`, `ALLOCATION`,
`RET_20` through `RET_1`, `SIGNED_VOLUME_20` through `_1`,
`MEDIAN_DAILY_TURNOVER`, `GROUP`. Its sample submission is `ROW_ID,prediction`.
This was checked directly, rather than inferred from the README (which says data
is excluded). Do not confuse CSV column order, lag order, anonymized timestamp
labels or the differing turnover names across versions.

No private labels, leaderboard results or the reported train size/class balance
could be verified. The second README reports 50.72% positive targets and 16.17%
positive signed volume. Such marginals are dataset-specific observations, not
universal constants or a sufficient test of feature usefulness.

## 1. QRT_challenge: broad feature research, uneven validation

Primary implementation: [preprocess.py](https://github.com/malerbe/QRT_challenge/blob/b24ee375eb9e15d22f1ef4931282fb74a5a6178f/preprocessing/preprocess.py),
[feature_engineering.py](https://github.com/malerbe/QRT_challenge/blob/b24ee375eb9e15d22f1ef4931282fb74a5a6178f/preprocessing/feature_engineering.py),
[models.py](https://github.com/malerbe/QRT_challenge/blob/b24ee375eb9e15d22f1ef4931282fb74a5a6178f/models/models.py)
and notebooks under `feature_engineering`, `specialized_models`, `linear`,
`xgboost` and `ensembling`.

**EDA and signal analysis.** Missing-value notebooks, Pearson correlations with
p-values, mutual information for both regression/classification, univariate Ridge
ranking and hierarchical clustering of `1-|Spearman correlation|`. Several
notebooks rank features or search combinations of representatives from correlated
clusters. Ridge, ElasticNet and Huber are useful cheap hypothesis tests.

**Feature engineering.** Short/long return means and dispersions; contemporaneous
cross-allocation market summaries; momentum, standardized mean-reversion,
volatility ratios, Sharpe/Sortino proxies; return-volume correlations; signed-volume
means/variance; skewness, sign entropy/flip rates; local volatility and drawdown.
More specialized blocks estimate cross-allocation covariance, market betas,
idiosyncratic residual volatility, leading covariance eigenvector loadings,
marginal/component risk contributions, minimum-variance and inverse-volatility
reference weights, and correlation/volatility regime summaries.

**Models.** Ridge and regularized/robust regressions, shallow XGBoost and LightGBM,
random forests with Optuna, LSTM regression/classification and hybrid sequential
plus static-feature networks. Specialized short-term, risk, market and portfolio
models are combined in a later notebook. Some regression outputs are converted
to labels by sign; classification outputs use their own class decisions.

**Validation.** Many linear/tree experiments partition unique dates, which prevents
same-date rows from crossing folds. Some use shuffled date folds, others
`KFold(shuffle=False)`. Neural notebooks also contain row-level `train_test_split`;
the later normalization experiment instead splits dates. Thus there is no single
consistent validation strategy. Accuracy on return sign is the competition score,
while some discovery diagnostics use R², and neural training uses MSE or cross-entropy.

**Material weaknesses found in code.**

- The ensemble notebook obtains base predictions on the full training set after
  the wrappers refit on that same set. Those in-sample predictions become
  meta-features before another CV. This contaminates the meta-model evaluation;
  proper stacking requires inner OOF predictions (and outer evaluation when
  selecting the stack). The toolkit uses compatible OOF averaging instead.
- The notebook called `robust_cross_validation` uses non-shuffled date KFold,
  which is not a forward temporal split. Its outer loop selects a subset, then
  calls a trainer's internal CV and reports that internal score, without scoring
  the outer held-out dates. The outer loop therefore does not provide an
  independent nested-CV estimate.
- Combination searches and target-based rankings repeatedly reuse labels/folds.
  The resulting best score includes feature-selection optimism, regardless of
  whether individual feature formulas are label-free.
- Predictor selection by `columns[2:]`, hardcoded turnover/lag names, duplicate
  preprocessor definitions, mutable notebook state and full-data refits make
  the implementation brittle. One benchmark builds `range(1,20)`, excluding the
  twentieth lag despite windows nominally named 20.
- A function named signed square-root computes a reciprocal square-root. This
  magnifies small values and illustrates why names are insufficient evidence
  of numerical behavior. Several ratio features divide by noisy near-zero
  means, and 20-observation covariance matrices are rank deficient for a large
  allocation universe. Pseudoinverses/shrinkage help numerically but do not
  create reliable covariance estimates.
- Eigenvector signs are arbitrary unless anchored; independent per-date PCA
  loadings can flip sign. Expensive per-date eigendecompositions/pseudoinverses
  and forest `n_jobs=50` are poor defaults for a two-person CPU hackathon.
- Some formulas cap 60-day windows to the 20 available history columns, creating
  nominally different but identical signals. Some methods append feature names
  that are not consistently produced. Notebook experiment scores and the
  speculative performance targets in the narrative are not reproducible evidence.

**Retained:** explicit row-history summaries, Pearson/Spearman/MI diagnostics,
correlation-based redundancy inspection, regularized linear baselines, optional
PCA and simple risk/volatility ideas. **Deferred:** deep sequence models,
combinatorial feature search, covariance optimization and stacking.

## 2. qrt-data-challenge: simple, useful baseline

Primary implementation: [benchmark_submission.ipynb](https://github.com/comemuntuqtb/qrt-data-challenge/blob/29b6333dbb63b6692e0afdb2656174bed0e7d69c/benchmark_submission.ipynb).
The README mentions another submission notebook, but it is absent from the checkout.

**EDA and hypotheses.** Class balance and sign distributions are inspected.
Momentum is a difference between short and long return means. Mean-reversion is
the latest return minus a recent mean. Group-specific interactions condition
signals on four anonymized groups. Group 1 versus other-group historical-return
correlations are explored.

**Features.** Raw 20 returns, 20 signed volumes, turnover; window means for
3/5/10/15/20 observations, same-date cross-allocation means and 20-period
volatility; mean-reversion for 3/5/20, short/long differences, return-volume
products, short-lag return differences and group-masked interactions. The actual
notebook's selected feature list still includes volume interactions which the
README describes as removed. Distinguish the narrative from the executable state.

**Models and metric.** Ridge and low-depth LightGBM regression (`objective=mse`),
with outputs thresholded at zero for sign accuracy. The main model has depth 3,
learning rate .01 and 500 trees. Eight models from date-grouped folds are averaged
before thresholding. Optuna explores depth, learning rate and boosting rounds,
using four date-grouped folds for faster trials.

**Reported outcomes.** README: mean CV 52.26%, public 51.60%; tuned model public
51.24%, benchmark 50.79%. These show local tuning gains need not transfer. The
notebook reports positive predictions around 59% versus training balance near
50.7%, a useful reason to inspect score/label distributions.

**Assessment.** This is the most practical starting point: a small feature set,
shallow trees, date folds and model averaging. Cross-sectional statistics are
label-free and valid if all same-date histories are available together. The date
folds protect that batch structure. They are not suitable for a genuine forward
time task, and anonymization alone does not prove there is no train/test shift.
The baseline assumes a continuous return target; it cannot be reused unchanged
for arbitrary labels or probability metrics. A reported mean ± fold standard
deviation is **not** a statistical confidence interval. Low fold variance does
not rule out overfitting or shift. Asymmetric volume signs do not prove that
return-volume products are useless, and a small observed group correlation does
not rule out nonlinear or other-horizon lead-lag effects.

**Retained:** shallow boosting, explicit windows, mean-reversion, full same-date
context, fixed-fold comparisons, fold prediction averaging, distribution checks.
**Changed:** arbitrary target/schema/metric support, learned imputation/scaling
inside folds, explicit class probabilities and checked submission schemas.

## 3. Trust or Short: stronger research logging, incomplete leakage guarantees

Primary sources: [features.py](https://github.com/PatthieuM/trust-or-short-QRT-Data-Challenge-2026/blob/29105b7716d5aa18c9a7237f0355450ba227129d/src/features.py),
[models.py](https://github.com/PatthieuM/trust-or-short-QRT-Data-Challenge-2026/blob/29105b7716d5aa18c9a7237f0355450ba227129d/src/models.py),
[validation.py](https://github.com/PatthieuM/trust-or-short-QRT-Data-Challenge-2026/blob/29105b7716d5aa18c9a7237f0355450ba227129d/src/validation.py),
experiment scripts and `results/experiments_log.md` / `submissions_tracker.csv`.

**Features.** Independent blocks cover multi-window means, volatilities,
mean/volatility ratios, positive-return hit rates, exponentially weighted
history means, momentum, z-scores, autocorrelation, extremes, skewness, streaks,
volume summaries and return-volume correlations. Same-date global/group ranks,
relative alpha and relative volatility add market context. Turnover ranks and
interactions, row-history slope/R² and allocation/group encodings are optional.
Later scripts explore allocation-level historical statistics and percentiles,
pure ranking/normalization and group-market interactions.

**Models and validation.** Main binary LightGBM uses depth 4, 500 rounds, learning
rate .05 and regularization. Eight shuffled unique-date folds prevent date
overlap. Logistic has training-fitted scaling. Alternatives include per-group
models, CatBoost, regularized Optuna tuning, multiple seeds, random forests,
MLPs, return regression objectives and adversarially selected validation/weights.
OOF/test prediction arrays and experiment/submission logs support comparisons.

**Author-reported evidence.** The tracker, which is more current than parts of the
experiment narrative, records:

| Experiment | Local CV | Public accuracy |
|---|---:|---:|
| Raw + engineered LightGBM | .5235 | .5149 |
| Pruned engineered LightGBM (best submitted) | .5238 | .5187 |
| CatBoost-heavy blend | .5275 | .5140 |
| Optuna run 9 | .5278 | .5146 |
| Remove drifting features | .5242 | .5123 |
| Neural network | .5238 | .5040 |
| Full-data refit blend | not recorded | .5120 |
| Return regression to sign | .5263 | .5129 |

The adversarial experiment reports train-versus-test AUC .8657. This supports
checking distribution shift, but it does not prove which features should be
removed or which labels transfer. The removal experiment performed worse.

**Material weaknesses.**

- Encodings are estimated inside the outer training fold, but their smoothing
  prior `global_mean` is computed from all labels, including validation labels.
  This is a small but real target-information leak.
- Training rows receive encodings calculated using their own labels. That is
  in-sample target encoding; correct training-row encodings need an inner
  cross-fit compatible with groups/time, or strictly historical labels. Outer
  validation labels are mostly excluded, but train/evaluation representations
  differ, especially for sparse identities.
- The main run passes OOF accuracy as `train_score`; the guardian subtracts
  mean fold accuracy. These are effectively the same prediction set with
  different weighting. Their difference is not a measure of overfitting.
  Other scripts sometimes use fold standard deviation under the same name,
  further mixing two unrelated diagnostics.
- The leakage checker checks flags and feature names; it does not inspect
  fitted state, actual fold intersections, availability times or encodings.
  Its blanket rejection of TimeSeriesSplit is specific to this assumed
  anonymized dataset, not a generic leakage rule.
- Category vocabulary is fitted on train+test. This is transductive use of
  unlabeled test metadata, not the same as target leakage; it is unnecessary
  for a toolkit that can handle unseen categories using training-only encoders.
- Rules such as never use logistic, never use CatBoost, depth above four is
  forbidden, or CV above 52.4% is dangerous are empirical judgments from one
  repeatedly queried public test. They should not become generic constraints.
  Repeated leaderboard feedback itself creates selection bias.
- Broadening feature lists and tuning repeatedly on the same folds does not
  yield an independent generalization estimate. High OOF correlation is a
  diversity diagnostic, not a mathematical threshold beyond which blends
  cannot help. A near-zero OOF-versus-fold-mean gap cannot validate a model.
- Saving a model alone does not preserve all fitted encoders, feature mappings
  and scaling. The toolkit retains whole fitted pipelines in memory and
  records recipes instead of auto-saving partial model artifacts.

**Retained:** feature blocks, same-date ranks/relative context, training-fitted
covariate mappings, conservative model capacity, feature ablations, shift
diagnostics, OOF prediction comparison and honest negative-result logging.
**Changed:** actual split and fitted-state tests, no target encoders by default,
per-run independent artifacts and configurable tasks/metrics/schemas.

## Reuse priorities for a four-hour event

| Priority | Technique | Reason / implementation |
|---|---|---|
| First | Establish target semantics, availability and validation | `data`, `validation`, first notebook cells |
| First | Linear + shallow boosted baseline | Fast comparison and a backup submission |
| First | Missingness, category/ID audits, train/test shift | `research.eda` / `drift_report` |
| High | Small multi-window summaries and reversion | Explicit column histories, no challenge-specific names |
| High | Same-time ranks and relative signals | Available batch only, whole-cross-section folds enforced |
| High | Regularization, feature ablations and OOF averaging | Low operational cost; retain individual candidates |
| Conditional | Group covariate statistics, PCA, observation clusters | Training-fitted inside each fold, optional |
| Conditional | Long-form lags, rolling volatility and lead-lag | Only with verified chronology and availability |
| Optional | Pair hedge/cointegration and cost-adjusted diagnostics | Training-only fit and later evaluation |
| Defer | Deep networks, covariance optimization, large stacking searches | Expensive, dataset-specific and easy to misvalidate |

The useful common lesson is to improve signal quality under believable validation.
None of the historical scores, feature names, sign thresholds, class balances,
date assumptions or model preferences is hardcoded into the new toolkit.
