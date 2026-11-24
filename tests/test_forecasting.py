"""Whole-horizon tests; no future realized price is passed to a predictor."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from qrtkit.forecasting import forecast_prices
from qrtkit.validation import last_period_holdout, make_splits
from qrtkit import load_config, train_cv


def small_panel(days=10, stocks=3):
    # Two illustrative quote times per day: no exchange session length is assumed.
    dates = pd.date_range("2025-01-01", periods=days, freq="B", tz="UTC")
    quotes = pd.DatetimeIndex([date + pd.Timedelta(hours=9, minutes=minute)
                               for date in dates for minute in [0, 5]])
    frame = pd.MultiIndex.from_product([np.arange(stocks), quotes], names=["stock", "clock"]).to_frame(index=False)
    frame["country"] = frame.stock % 6
    frame["price"] = (100 + frame.stock) * np.exp(.001 * frame.groupby("stock").cumcount())
    return frame


class ForecastTests(unittest.TestCase):
    def test_baselines_whole_horizon_order_and_formula(self):
        panel = small_panel()
        tr, va = last_period_holdout(panel, time_column="clock", test_periods=4)
        history = panel.iloc[tr]
        future = panel.iloc[va].drop(columns="price").sample(frac=1., random_state=2)
        last = forecast_prices(history, future, entity_columns=["stock"], time_column="clock", price_column="price")
        expected = future.stock.map(history.groupby("stock").price.last())
        assert_array_equal(last.index, future.index)
        assert_allclose(last, expected)
        drift = forecast_prices(history, future, entity_columns=["stock"], time_column="clock", price_column="price",
                                method="damped_drift", drift_window=4, damping=.5)
        for stock, group in future.sort_values("clock").groupby("stock"):
            horizon = np.arange(1, len(group) + 1)
            expected = history.loc[history.stock == stock, "price"].iloc[-1] * np.exp(.001 * (1 - .5 ** horizon))
            assert_allclose(drift.loc[group.index], expected)

    def test_no_future_prices_or_overlapping_history(self):
        panel = small_panel()
        tr, va = last_period_holdout(panel, time_column="clock", test_periods=4)
        kwargs = dict(entity_columns=["stock"], time_column="clock", price_column="price")
        with self.assertRaisesRegex(ValueError, "Future prices"):
            forecast_prices(panel.iloc[tr], panel.iloc[va], **kwargs)
        with self.assertRaisesRegex(ValueError, "historical cutoff"):
            forecast_prices(panel, panel.iloc[va].drop(columns="price"), **kwargs)
        with self.assertRaisesRegex(ValueError, "no historical price"):
            future = panel.iloc[va].drop(columns="price").copy()
            future["stock"] += 999
            forecast_prices(panel.iloc[tr], future, **kwargs)

    def test_day_holdout_whole_panel_and_gap(self):
        panel = small_panel().sample(frac=1., random_state=4)
        tr, va = last_period_holdout(panel, time_column="clock", test_periods=4, gap=2)
        self.assertEqual(panel.iloc[va].clock.dt.normalize().nunique(), 4)
        self.assertEqual(panel.iloc[tr].clock.dt.normalize().nunique(), 5)
        self.assertLess(panel.iloc[tr].clock.max(), panel.iloc[va].clock.min())
        self.assertEqual(set(panel.iloc[tr].stock), set(panel.iloc[va].stock))
        for stamp, group in panel.groupby("clock"):
            self.assertTrue(set(group.index) <= set(panel.iloc[tr].index) or
                            set(group.index) <= set(panel.iloc[va].index) or
                            not set(group.index).intersection(panel.iloc[np.r_[tr, va]].index))

    def test_explicit_session_blocks_and_bad_mapping(self):
        panel = small_panel()
        panel["day"] = panel.clock.dt.strftime("%Y-%m-%d")
        panel["bar"] = panel.clock.astype("int64")
        tr, va = last_period_holdout(panel, time_column="bar", period_column="day", test_periods=4)
        self.assertEqual(panel.iloc[va].day.nunique(), 4)
        with self.assertRaisesRegex(ValueError, "period_column"):
            last_period_holdout(panel, time_column="bar", test_periods=4)
        panel.loc[0, "day"] = "wrong-country-local-date"
        with self.assertRaisesRegex(ValueError, "multiple periods"):
            last_period_holdout(panel, time_column="bar", period_column="day", test_periods=4)

    def test_holdout_training_integration(self):
        c = load_config(Path(__file__).resolve().parents[1] / "config.json")
        panel = small_panel(days=20, stocks=6)
        c.update(task="regression", metric={"name": "rmse"})
        c["data"].update(id_columns=[], time_column="clock", group_column="stock")
        c["validation"].update(strategy="temporal_holdout", n_splits=1, test_periods=4, period_column=None)
        X = panel.drop(columns="price")
        X["horizon_clock"] = (X.clock - X.clock.min()).dt.total_seconds() / 86400
        r = train_cv(X, panel.price, None, c, "ridge", track=False, progress=False)
        self.assertEqual(len(r.fold_scores), 1)
        self.assertEqual(int(r.covered.sum()), 4 * 2 * 6)
        self.assertTrue(np.isnan(r.oof[~r.covered]).all())
        self.assertTrue(np.isfinite(r.oof[r.covered]).all())

    def test_386_stocks_six_countries_full_199_day_forecast(self):
        # Deliberately reduced bars/day; exercises the full entity/day dimensions.
        panel = small_panel(days=220, stocks=386)
        tr, va = last_period_holdout(panel, time_column="clock", test_periods=199)
        future = panel.iloc[va].drop(columns="price")
        self.assertEqual(future.stock.nunique(), 386)
        self.assertEqual(future.country.nunique(), 6)
        self.assertEqual(future.clock.dt.normalize().nunique(), 199)
        for method in ["last", "damped_drift"]:
            p = forecast_prices(panel.iloc[tr], future, entity_columns=["country", "stock"],
                                time_column="clock", price_column="price", method=method)
            self.assertEqual(len(p), 199 * 2 * 386)
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue((p > 0).all())


if __name__ == "__main__":
    unittest.main()
