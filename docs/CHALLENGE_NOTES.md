# Current ETH challenge information

Confirmed by the team:

- One observed price every five minutes.
- 386 stocks across six countries.
- Forecast prices throughout the final 199 days.
- Larger moves contribute more to the score than smaller moves.

The Kaggle statement, files and exact scoring formula have not arrived yet.
The number of observations per session, exchange calendars, country/timezone
mapping, currency/price adjustments, calendar versus trading days, forecast
origin, provided future inputs and submission format still need confirmation.

## Forecast protocol

Treat the task provisionally as a **single-origin, whole-period price forecast**.
Train on observations before the forecast boundary and predict every requested
future quote. Do not use realized prices, returns, volatility or market returns
from inside those 199 days to construct another prediction. An online backtest
that observes each new true price would answer a different question.

Use complete final date/session blocks for validation across all stocks.
`last_period_holdout` supports that directly; `test_periods=199` counts **observed
date/session blocks**. If the statement means elapsed calendar days, select the
explicit cutoff from the statement instead. By default, timezone-aware timestamps
are grouped by UTC date. For exchange-specific calendars, first define an
appropriate common prediction boundary rather than assigning conflicting local
dates to the same global timestamp.

```python
from qrtkit.validation import last_period_holdout
from qrtkit.forecasting import forecast_prices

# Example names only: adapt after inspecting the files and parsing timestamps.
train_idx, valid_idx = last_period_holdout(
    prices, time_column="timestamp", test_periods=199,
    # period_column="session_date",  # optional explicit common session blocks
)
history = prices.iloc[train_idx]
heldout = prices.iloc[valid_idx]

pred = forecast_prices(
    history,
    heldout.drop(columns="price"),  # actual held-out prices never enter the model
    entity_columns=["stock_id"],    # include country if IDs are not globally unique
    time_column="timestamp",
    price_column="price",
    method="last",
)
# Keep heldout['price'] separate for evaluation using the official metric.
```

For the real prediction period, use the **provided future quote schedule** in
place of `heldout.drop(...)`. The toolkit does not invent trading hours or fill
overnight/weekend intervals. Predictions preserve its row order and index. Feed
only finite positive historical prices; handle missing values and verify split/
dividend adjustments from the dataset documentation first.

Two CPU baselines are ready in `qrtkit.forecasting`:

- `last`: repeat each stock's last historical observed price over the full grid.
- `damped_drift`: estimate mean historical log-price changes per stock and damp
  their extrapolation with horizon. Window and damping are configurable. This
  uses no future realized prices and is a comparison candidate, not an asserted
  improvement over persistence.

The drift horizon counts supplied future quotes per stock, not elapsed minutes.
No fixed number of bars per country/session is assumed. These are direct price
forecast baselines, separate from the original classification/short-horizon
financial demo. The main `config.json` intentionally remains a runnable generic
demo until the actual schema and scoring protocol are known.

The generic model runner also accepts a complete-day validation configuration:

```json
{
  "strategy": "temporal_holdout",
  "n_splits": 1,
  "test_periods": 199,
  "period_column": null,
  "shuffle": false,
  "gap": 0,
  "label_horizon": 0,
  "label_end_column": null
}
```

Use this in `validation` with a verified time column. It defines the split, not
the forecasting model. A tabular model must receive predictors that can actually
be produced at the shared forecast origin. Historical lag/rolling features
must use `availability="frozen"`; future same-time price cross sections cannot
be enabled. For a stronger direct-horizon model, construct historical forecast
origins with horizon and calendar features, using only each origin's available
history; split and purge by forecast/label availability, not random rows.

## Division of work once the files arrive

Person 1: establish time ordering and sessions; check missing/stale quotes and
price adjustments; examine returns, volatility, country/stock factors and their
stability **before the cutoff**. Build features that remain available for the
whole forecast horizon. Do not average raw price levels across currencies or
stocks; establish appropriate normalization from the data description.

Person 2: reproduce the official metric exactly; generate the persistence
submission first; compare whole-period holdouts; check per-stock/country/horizon
errors; then train budgeted models using the same information constraints.

"Large moves count more" does not identify a unique loss. It could refer to a
weighted error, a squared-error score, a directional score weighted by return
magnitude, or another definition. Neither ordinary accuracy nor a guessed
absolute-return weighting should replace the official formula. The existing
`Metric` interface can accept the exact scorer once supplied; any external
evaluation weights must remain separate from prediction features.

## Checks added for this information

Tests cover complete-day boundaries, global stock-panel isolation, timestamp
gaps, frozen-origin price forecasts, unseen-stock errors, future-price rejection,
row alignment and exact damped-drift arithmetic. A synthetic panel exercises all
386 stocks, six countries and a 199-day holdout with **two illustrative quotes
per day** to keep the test small; it is not a guessed exchange schedule.
