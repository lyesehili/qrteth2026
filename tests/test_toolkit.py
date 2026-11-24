"""Offline correctness and leakage tests. Run: python -m unittest discover -s tests -v"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from qrtkit import load_config, load_data, FeatureBuilder, train_cv
from qrtkit.synthetic import make_demo, financial_config
from qrtkit.data import read_table
from qrtkit.models import available_models, make_pipeline
from qrtkit.metrics import get_metric, evaluate, Metric
from qrtkit.validation import make_splits, check_feature_splits
from qrtkit.training import ensemble
from qrtkit.submission import make_submission
from qrtkit.tracking import leaderboard
from qrtkit.research import eda, correlations, signal_table, feature_clusters, lead_lag, information_coefficient, signal_bins, drift_report
from qrtkit.finance import fit_pair, backtest_signal

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "config.json")


class PipelineTests(unittest.TestCase):
    def test_multiclass_cli(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            c["output_dir"] = tmp
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(c))
            run = subprocess.run([sys.executable, "-m", "qrtkit", "--config", str(path), "--demo", "multiclass"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(len(list(Path(tmp).glob("*/submission.csv"))), 3)

    def test_classification_all_baselines(self):
        c = config()
        d = load_data(c)
        for name in ["ridge", "logistic", "histgb", "lightgbm", "xgboost", "catboost"]:
            with self.subTest(model=name):
                if not available_models()[name]:
                    continue
                params = {"n_estimators": 30} if name in {"lightgbm", "xgboost"} else {"iterations": 30} if name == "catboost" else {}
                r = train_cv(d.train, d.y, d.test, c, name, params, track=False, progress=False)
                self.assertTrue(r.covered.all())
                self.assertGreater(r.score, .65)
                assert_allclose(r.test_predictions.sum(1), 1.)
                self.assertEqual(r.oof.shape, (len(d.y), 2))
                self.assertEqual(make_submission(r, d.test, c).shape, (len(d.test), 2))

    def test_regression_all_baselines(self):
        c = config()
        c.update(task="regression", metric={"name": "rmse"})
        c["data"]["demo"] = "regression"
        c["validation"]["strategy"] = "standard"
        c["submission"]["kind"] = "value"
        d = load_data(c)
        for name in ["linear", "ridge", "lasso", "elasticnet", "huber", "histgb", "lightgbm", "xgboost", "catboost"]:
            with self.subTest(model=name):
                if not available_models()[name]:
                    continue
                params = {"n_estimators": 30} if name in {"lightgbm", "xgboost"} else {"iterations": 30} if name == "catboost" else {}
                r = train_cv(d.train, d.y, d.test, c, name, params, track=False, progress=False)
                self.assertTrue(np.isfinite(r.oof).all())
                self.assertLess(r.score, 160.)
                if name in {"linear", "ridge", "huber"}:
                    self.assertLess(r.score, 30.)
                self.assertFalse(r.metric.greater_is_better)
                make_submission(r, d.test, c)

    def test_multiclass_all_classifiers(self):
        c = config()
        c["data"]["demo"] = "multiclass"
        c["metric"]["name"] = "log_loss"
        c["submission"]["prediction_columns"] = ["down", "flat", "up"]
        d = load_data(c)
        for name in ["ridge", "logistic", "lightgbm", "xgboost", "catboost"]:
            if not available_models()[name]:
                continue
            with self.subTest(model=name):
                params = {"n_estimators": 20} if name in {"lightgbm", "xgboost"} else {"iterations": 20} if name == "catboost" else {}
                r = train_cv(d.train, d.y, d.test, c, name, params, track=False, progress=False)
                assert_allclose(r.test_predictions.sum(1), 1.)
                self.assertEqual(make_submission(r, d.test, c).shape[1], 4)
                c["submission"]["prediction_columns"] = ["label"]
                c["submission"]["kind"] = "label"
                labels = make_submission(r, d.test, c)
                self.assertTrue(set(labels.label) <= set(d.y))
                c["submission"].update(prediction_columns=["down", "flat", "up"], kind="probability")

    def test_financial_complete_pipeline(self):
        c = financial_config(config())
        c["features"]["windows"] = [2, 3]
        c["preprocessing"].update(pca_components=2, clusters=2, clip_quantiles=[.01, .99])
        d = load_data(c)
        with tempfile.TemporaryDirectory() as tmp:
            c["output_dir"] = tmp
            a = train_cv(d.train, d.y, d.test, c, "logistic", progress=False)
            b = train_cv(d.train, d.y, d.test, c, "ridge", progress=False)
            r = ensemble([a, b], d.y)
            self.assertGreater(r.score, .55)
            self.assertLess(r.covered.mean(), 1.)
            self.assertTrue(np.isnan(r.oof[~r.covered]).all())
            make_submission(r, d.test, c, Path(tmp) / "submission.csv")
            self.assertEqual(len(leaderboard(tmp)), 2)
            self.assertTrue((Path(a.output_path) / "config.json").exists())

    def test_tracking_and_ensemble_alignment(self):
        c, d = config(), make_demo()
        with tempfile.TemporaryDirectory() as tmp:
            c["output_dir"] = tmp
            a = train_cv(d.train, d.y, d.test, c, "logistic", progress=False)
            b = train_cv(d.train, d.y, d.test, c, "ridge", progress=False)
            r = ensemble([a, b], d.y, [.7, .3])
            assert_allclose(r.oof, .7 * a.oof + .3 * b.oof)
            summary = json.loads((Path(a.output_path) / "summary.json").read_text())
            self.assertIn("environment", summary)
            self.assertEqual(summary["coverage"], 1.)
            with self.assertRaises(ValueError):
                ensemble([a, b], d.y.iloc[::-1])
            with self.assertRaises(ValueError):
                ensemble([a, b], d.y, [-1, 1])
            b.split_hash = "different"
            with self.assertRaises(ValueError):
                ensemble([a, b], d.y)
            with self.assertRaises(ValueError):
                make_submission(a, d.test.iloc[::-1], c)


class ValidationTests(unittest.TestCase):
    def test_all_split_strategies(self):
        c, d = config(), make_demo(n=300)
        X = d.train.copy()
        X["team"] = np.arange(len(X)) // 8
        X["clock"] = np.arange(len(X)) // 4
        c["data"].update(group_column="team", time_column="clock")
        for strategy in ["standard", "stratified", "group", "stratified_group", "temporal"]:
            with self.subTest(strategy=strategy):
                c["validation"]["strategy"] = strategy
                splits = make_splits(X, d.y, c)
                for tr, va in splits:
                    self.assertFalse(set(tr) & set(va))
                    if "group" in strategy:
                        self.assertFalse(set(X.iloc[tr].team) & set(X.iloc[va].team))
                    if strategy == "temporal":
                        self.assertLess(X.iloc[tr].clock.max(), X.iloc[va].clock.min())

    def test_grouped_models_end_to_end(self):
        c, d = config(), make_demo(n=240)
        X = d.train.assign(subject=np.arange(len(d.train)) // 6)
        test = d.test.assign(subject=1000 + np.arange(len(d.test)) // 6)
        c["data"]["group_column"] = "subject"
        for strategy in ["group", "stratified_group"]:
            c["validation"]["strategy"] = strategy
            r = train_cv(X, d.y, test, c, "logistic", track=False, progress=False)
            self.assertTrue(r.covered.all())
            self.assertGreater(r.score, .6)
            make_submission(r, test, c)

    def test_temporal_purge_gap_horizon_and_end_times(self):
        c = config()
        c["data"]["time_column"] = "clock"
        c["validation"].update(strategy="temporal", gap=1, label_horizon=2, label_end_column="ends", max_train_periods=12)
        X = pd.DataFrame({"clock": np.repeat(np.arange(40), 2)})
        X["ends"] = X.clock + np.where(X.clock % 3 == 0, 10, 2)
        for tr, va in make_splits(X, np.arange(len(X)), c):
            self.assertGreaterEqual(X.iloc[va].clock.min() - X.iloc[tr].clock.max(), 4)
            self.assertTrue((X.iloc[tr].ends < X.iloc[va].clock.min()).all())
            self.assertLessEqual(X.iloc[tr].clock.nunique(), 12)
        X.clock = X.clock.astype(str)
        with self.assertRaises(ValueError):
            make_splits(X, np.arange(len(X)), c)

    def test_cross_sections_require_whole_batches(self):
        c, d = financial_config(config()), make_demo("financial")
        c["features"]["temporal"]["columns"] = []
        c["validation"]["strategy"] = "standard"
        splits = make_splits(d.train, d.y, c)
        with self.assertRaisesRegex(ValueError, "cross section"):
            check_feature_splits(d.train, d.test, splits, c)
        c["validation"]["strategy"] = "group"
        c["data"]["group_column"] = "session"
        check_feature_splits(d.train, d.test, make_splits(d.train, d.y, c), c)
        c["features"]["cross_sectional"]["available_at_prediction"] = False
        with self.assertRaises(ValueError):
            FeatureBuilder(c).fit_transform(d.train)

    def test_lags_reject_random_validation(self):
        c, d = financial_config(config()), make_demo("financial")
        c["validation"]["strategy"] = "stratified"
        with self.assertRaisesRegex(ValueError, "require temporal"):
            train_cv(d.train, d.y, d.test, c, track=False)

    def test_class_and_index_errors(self):
        c, d = config(), make_demo()
        with self.assertRaises(ValueError):
            train_cv(d.train, d.y.iloc[::-1], d.test, c, track=False)
        c["data"]["group_column"] = "label_group"
        X = d.train.assign(label_group=d.y)
        c["validation"].update(strategy="group", n_splits=2)
        with self.assertRaisesRegex(ValueError, "lacks a class"):
            train_cv(X, d.y, None, c, track=False)


class FeatureSafetyTests(unittest.TestCase):
    def temporal_setup(self, mode="online"):
        c = config()
        c["data"].update(id_columns=[], time_column="t", group_column="asset")
        c["features"]["temporal"].update(columns=["v"], entity_columns=["asset"],
                                         lags=[1, 2], windows=[2], availability=mode)
        return c

    def test_causal_sorted_and_entity_lags(self):
        c = self.temporal_setup()
        X = pd.DataFrame({"t": [2, 1, 3, 1, 2, 3], "asset": ["a", "a", "a", "b", "b", "b"], "v": [20., 10., 30., 1., 2., 3.]}, index=[10, 11, 12, 13, 14, 15])
        a = FeatureBuilder(c).fit_transform(X)
        self.assertEqual(a.loc[10, "v__lag_1"], 10.)
        self.assertEqual(a.loc[12, "v__roll_mean_2"], 15.)
        self.assertEqual(a.loc[15, "v__lag_1"], 2.)
        self.assertTrue(np.isnan(a.loc[11, "v__lag_1"]))
        changed = X.copy()
        changed.loc[12, "v"] = 1e12
        b = FeatureBuilder(c).fit_transform(changed)
        assert_allclose(a.filter(like="__"), b.filter(like="__"), equal_nan=True)
        with self.assertRaises(ValueError):
            FeatureBuilder(c).fit_transform(pd.concat([X, X.iloc[[0]]], ignore_index=True))

    def test_online_and_frozen_forecasts(self):
        train = pd.DataFrame({"t": [1, 2, 3], "asset": ["a"] * 3, "v": [1., 2., 3.]})
        valid = pd.DataFrame({"t": [4, 5, 6], "asset": ["a"] * 3, "v": [4., 5., 6.]})
        online = FeatureBuilder(self.temporal_setup("online")).fit(train).transform(valid)
        frozen = FeatureBuilder(self.temporal_setup("frozen")).fit(train).transform(valid)
        assert_array_equal(online["v__lag_1"], [3., 4., 5.])
        assert_array_equal(frozen["v__lag_1"], [3., 3., 3.])
        unseen = valid.assign(asset="unseen")
        out = FeatureBuilder(self.temporal_setup("frozen")).fit(train).transform(unseen)
        self.assertTrue(out["v__lag_1"].isna().all())

    def test_group_stats_and_factors_fitted_on_training_only(self):
        c = config()
        c["data"]["id_columns"] = []
        c["features"]["group_statistics"].update(columns=["x"], keys=["g"])
        c["preprocessing"].update(pca_components=1, clusters=2, clip_quantiles=[.1, .9])
        X = pd.DataFrame({"x": [1., 2., np.nan, 4., 6., 7.], "z": [1., 2., 4., 8., 9., 11.], "g": ["a"]*3 + ["b"]*3})
        valid = pd.DataFrame({"x": [1e9, np.nan], "z": [1e10, 2.], "g": ["a", "unseen"]})
        p = make_pipeline(c, "ridge").fit(X, np.arange(6.))
        numeric = p.named_steps["preprocess"].named_transformers_["numeric"]
        impute = numeric.named_steps["impute"].statistics_.copy()
        factors = numeric.named_steps["factors"]
        components = factors.pca_.components_.copy()
        out = p.named_steps["features"].transform(valid)
        self.assertEqual(out.loc[0, "x__group_mean"], 1.5)
        self.assertTrue(np.isnan(out.loc[1, "x__group_mean"]))
        p.predict(valid)
        assert_array_equal(impute, numeric.named_steps["impute"].statistics_)
        assert_array_equal(components, factors.pca_.components_)

    def test_histories_and_target_exclusion(self):
        c = config()
        c["data"]["id_columns"] = []
        c["features"].update(histories={"history": ["new", "old", "oldest"]}, windows=[2, 3])
        X = pd.DataFrame({"new": [3., np.nan], "old": [2., np.nan], "oldest": [1., np.nan]})
        out = FeatureBuilder(c).fit_transform(X)
        self.assertEqual(out.loc[0, "history__mean_3"], 2.)
        self.assertEqual(out.loc[0, "history__reversion_3"], 1.)
        self.assertTrue(np.isnan(out.loc[1, "history__hit_3"]))
        with self.assertRaisesRegex(ValueError, "target"):
            FeatureBuilder(c).fit_transform(X.assign(outcome=1.))
        c["data"]["drop_columns"] = ["new"]
        with self.assertRaisesRegex(ValueError, "excluded metadata/unavailable"):
            FeatureBuilder(c).fit_transform(X)
        c["data"].update(drop_columns=[], id_columns=["new"])
        with self.assertRaisesRegex(ValueError, "excluded metadata/unavailable"):
            FeatureBuilder(c).fit_transform(X)
        c["data"]["id_columns"] = []
        c["validation"]["label_end_column"] = "old"
        with self.assertRaisesRegex(ValueError, "excluded metadata/unavailable"):
            FeatureBuilder(c).fit_transform(X)


class SubmissionAndIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c, cls.d = config(), make_demo()
        cls.r = train_cv(cls.d.train, cls.d.y, cls.d.test, cls.c, "logistic", track=False, progress=False)

    def test_template_alignment_and_overwrite(self):
        c = deepcopy(self.c)
        with tempfile.TemporaryDirectory() as tmp:
            template = self.d.test[["record"]].iloc[::-1].assign(prediction=0.)
            template.to_csv(Path(tmp) / "template.csv", index=False)
            c["submission"]["template_path"] = str(Path(tmp) / "template.csv")
            out = make_submission(self.r, self.d.test, c, Path(tmp) / "sub.csv")
            assert_array_equal(out.record, template.record)
            assert_allclose(out.prediction, self.r.test_predictions[::-1, 1])
            with self.assertRaises(FileExistsError):
                make_submission(self.r, self.d.test, c, Path(tmp) / "sub.csv")
            template.iloc[0, 0] = "wrong"
            template.to_csv(Path(tmp) / "template.csv", index=False)
            with self.assertRaises(ValueError):
                make_submission(self.r, self.d.test, c)

    def test_positive_class_labels_and_schema(self):
        c = deepcopy(self.c)
        c["submission"]["positive_class"] = "no"
        assert_allclose(make_submission(self.r, self.d.test, c).prediction, self.r.test_predictions[:, 0])
        c["submission"]["kind"] = "label"
        out = make_submission(self.r, self.d.test, c)
        self.assertTrue(set(out.prediction) <= {"no", "yes"})
        c["submission"]["prediction_columns"] = ["a", "b"]
        with self.assertRaises(ValueError):
            make_submission(self.r, self.d.test, c)

    def test_generic_separate_labels_by_id(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"identity": ["b", "a"], "arbitrary": [2., 1.]}).to_csv(root / "tr.csv", index=False)
            pd.DataFrame({"identity": ["c"], "arbitrary": [3.]}).to_csv(root / "te.csv", index=False)
            pd.DataFrame({"identity": ["a", "b"], "response": [10., 20.]}).to_csv(root / "y.csv", index=False)
            c["task"] = "regression"
            c["data"].update(demo=None, train_path=str(root / "tr.csv"), test_path=str(root / "te.csv"),
                              target_path=str(root / "y.csv"), target="response", id_columns=["identity"])
            d = load_data(c)
            assert_array_equal(d.y, [20., 10.])
            self.assertNotIn("response", d.train)
            c["data"]["id_columns"] = []
            with self.assertRaises(ValueError):
                load_data(c)

    def test_named_datetime_loading_and_temporal_training(self):
        c = config()
        d = make_demo("financial")
        c = financial_config(c)
        c["data"].update(demo=None, target="response", datetime_columns=["session"])
        with tempfile.TemporaryDirectory() as tmp:
            for name, frame in [("train", d.train.assign(response=d.y)), ("test", d.test)]:
                frame = frame.copy()
                frame["session"] = (pd.Timestamp("2020-01-01") + pd.to_timedelta(frame.session, unit="D")).astype(str)
                frame.to_csv(Path(tmp) / (name + ".csv"), index=False)
                c["data"][name + "_path"] = str(Path(tmp) / (name + ".csv"))
            loaded = load_data(c)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded.train.session))
            r = train_cv(loaded.train, loaded.y, loaded.test, c, "logistic", track=False, progress=False)
            self.assertGreater(r.score, .55)


class ResearchTests(unittest.TestCase):
    def test_reports_and_signal_diagnostics(self):
        d = make_demo()
        self.assertIn("profile", eda(d.train, d.y, "classification"))
        self.assertIn("pearson", correlations(d.train))
        self.assertIn("mutual_information", signal_table(d.train, d.y, "classification"))
        self.assertGreater(len(feature_clusters(d.train)), 0)
        self.assertEqual(len(drift_report(d.train, d.test)), d.train.shape[1])
        table = signal_bins(d.train.measurement_0, (d.y == "yes").astype(int))
        self.assertGreater(len(table), 0)
        ic = information_coefficient(d.train.measurement_0, (d.y == "yes").astype(int), np.arange(len(d.y)) // 12)
        self.assertGreater(ic["n_groups"], 1)

    def test_lead_lag_and_pair_analysis(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=100)
        wide = pd.DataFrame({"leader": x, "follower": np.r_[0., x[:-1]]})
        report = lead_lag(wide, lags=[1])
        found = report[(report.leader == "leader") & (report.follower == "follower")].iloc[0]
        self.assertAlmostEqual(found.correlation, 1.)
        base = np.cumsum(rng.normal(0, .01, 100)) + 4
        prices = pd.DataFrame({"a": np.exp(base + rng.normal(0, .005, 100)), "b": np.exp(base)})
        pair = fit_pair(prices.iloc[:70], cointegration=True)
        transformed = pair.transform(prices.iloc[70:])
        self.assertTrue(np.isfinite(transformed.zscore).all())
        pnl = backtest_signal(pd.Series([1., -1., 0.]), pd.Series([.01, -.01, .02]), cost_bps=1)
        self.assertAlmostEqual(pnl["pnl"].iloc[0], .0099)
        self.assertLessEqual(pnl["max_drawdown"], 0.)

    def test_metrics_custom_and_domains(self):
        p = np.array([[.8, .2], [.1, .9], [.6, .4], [.3, .7]])
        for name in ["accuracy", "balanced_accuracy", "f1", "roc_auc", "log_loss", "average_precision"]:
            self.assertTrue(np.isfinite(evaluate([0, 1, 0, 1], p, get_metric(name, "classification"))))
        for name in ["rmse", "mse", "mae", "r2", "rmsle", "mape", "pearson", "spearman", "sign_accuracy"]:
            self.assertTrue(np.isfinite(evaluate([1., 2., 3.], [1.1, 1.8, 2.9], get_metric(name, "regression"))))
        custom = Metric("bias", lambda y, p: np.mean(p-y), False, "regression")
        self.assertAlmostEqual(evaluate([1., 2.], [2., 3.], custom), 1.)
        with self.assertRaises(ValueError):
            evaluate([0, 0], p[:2], get_metric("roc_auc", "classification"))

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna not installed")
    def test_tuning_classification_regression_and_timeout(self):
        from qrtkit.tuning import tune
        c, d = config(), make_demo(n=180)
        r = tune(d.train, d.y, c, "logistic", seconds=15, max_trials=2)
        self.assertIn("C", r.best_params)
        c.update(task="regression", metric={"name": "rmse"})
        c["validation"]["strategy"] = "standard"
        d = make_demo("regression", n=180)
        r = tune(d.train, d.y, c, "ridge", seconds=15, max_trials=2)
        self.assertIn("alpha", r.best_params)
        with self.assertRaises(TimeoutError):
            tune(d.train, d.y, c, "ridge", seconds=1e-9, max_trials=1)


if __name__ == "__main__":
    unittest.main()
