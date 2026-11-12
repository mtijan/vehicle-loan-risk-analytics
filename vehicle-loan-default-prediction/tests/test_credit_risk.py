"""Checks for data boundaries and the decision policy, using synthetic inputs."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from credit_risk import Config, TARGET, choose_threshold, evaluate, preprocess, split_data, CATEGORICAL, engineer_features
from eda import wilson_interval
from operations import bootstrap_intervals, choose_low_threshold, population_stability_index, route_scores
from experiments import cost_sensitivity


class DataBoundaryTests(unittest.TestCase):
    def test_preprocessing_accepts_feature_ablation_without_entity_columns(self):
        x = pd.DataFrame({"numeric": np.arange(100), "EMPLOYMENT_TYPE": ["A", "B"] * 50})
        y = pd.Series([0, 1] * 50)
        transformed = preprocess(x, 42).fit_transform(x, y)
        self.assertEqual(transformed.shape, (100, 2))
        self.assertTrue(np.isfinite(transformed).all())

    def test_split_is_disjoint_complete_and_reproducible(self):
        raw = pd.DataFrame({"UNIQUEID": range(1000), TARGET: [0, 0, 0, 0, 1] * 200})
        parts = split_data(raw, Config())
        self.assertEqual([len(p) for p in parts], [600, 200, 200])
        identifiers = [set(p.UNIQUEID) for p in parts]
        self.assertFalse(identifiers[0] & identifiers[1])
        self.assertFalse(identifiers[0] & identifiers[2])
        self.assertFalse(identifiers[1] & identifiers[2])
        self.assertEqual(len(set.union(*identifiers)), 1000)
        for first, second in zip(parts, split_data(raw, Config())):
            pd.testing.assert_frame_equal(first, second)

    def test_training_encoding_does_not_memorize_unique_category_targets(self):
        n = 100
        x = pd.DataFrame({c: [f"unique-{i}" for i in range(n)] for c in CATEGORICAL})
        x["numeric"] = np.arange(n, dtype=float)
        y = pd.Series([0, 1] * 50)
        encoder = preprocess(x, 42)
        encoded = encoder.fit_transform(x, y)
        # Each unique level is unseen in its held-out encoding fold.
        self.assertTrue(np.allclose(encoded[:, 1:], 0.5))
        unseen = pd.DataFrame({c: ["never-seen"] for c in CATEGORICAL})
        unseen["numeric"] = np.nan
        transformed = encoder.transform(unseen)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.allclose(transformed[:, 1:], y.mean()))
        self.assertAlmostEqual(transformed[0, 0], 49.5)

    def test_threshold_respects_capacity_and_minimizes_validation_cost(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        scores = np.array([0.1, 0.9, 0.3, 0.8, 0.4, 0.7, 0.2, 0.6])
        threshold, table = choose_threshold(y, scores, Config(max_review_share=0.25), 45, 8)
        policy = evaluate(y, scores, threshold, 45, 8)
        self.assertLessEqual(policy["review_share"], 0.25)
        self.assertEqual(policy["scenario_cost"], table.loc[table.review_share <= 0.25, "scenario_cost"].min())
        self.assertEqual(policy["tp"], 2)

    def test_tied_unit_scores_have_feasible_no_review_endpoint(self):
        threshold, _ = choose_threshold([0, 1] * 10, np.ones(20), Config(), 45, 8)
        self.assertGreater(threshold, 1.0)

    def test_cost_accounts_for_both_error_types(self):
        result = evaluate([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], 0.5, 45, 8)
        self.assertEqual([result[k] for k in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        self.assertEqual(result["scenario_cost"], 53)
        self.assertEqual(result["approve_all_cost"], 90)
        self.assertAlmostEqual(result["scenario_cost_reduction"], 37 / 90)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            Config(test_size=0.6, validation_size=0.6)
        with self.assertRaises(ValueError):
            Config(max_review_share=0)

    def test_engineering_is_row_local_and_excludes_labels_and_application_ids(self):
        row = {c: "example" for c in CATEGORICAL}
        row.update({
            "UNIQUEID": 1, TARGET: 0, "DATE_OF_BIRTH": "01-01-1990",
            "DISBURSAL_DATE": "01-01-2018", "AVERAGE_ACCT_AGE": "2yrs 3mon",
            "CREDIT_HISTORY_LENGTH": "5yrs 1mon", "ASSET_COST": 100000,
            "DISBURSED_AMOUNT": 75000, "PRI_ACTIVE_ACCTS": 1, "SEC_ACTIVE_ACCTS": 2,
            "PRI_OVERDUE_ACCTS": 1, "SEC_OVERDUE_ACCTS": 0, "PRI_CURRENT_BALANCE": 2000,
            "SEC_CURRENT_BALANCE": 500, "PRI_NO_OF_ACCTS": 3, "SEC_NO_OF_ACCTS": 2,
            "PERFORM_CNS_SCORE": 600, "PRIMARY_INSTAL_AMT": 3000, "PRI_SANCTIONED_AMOUNT": 0,
        })
        single = engineer_features(pd.DataFrame([row]))
        altered = {**row, TARGET: 1, "UNIQUEID": 999, "ASSET_COST": 200000}
        batch = engineer_features(pd.DataFrame([row, altered]))
        pd.testing.assert_frame_equal(single, batch.iloc[:1])
        self.assertNotIn(TARGET, single)
        self.assertNotIn("UNIQUEID", single)
        self.assertNotIn("DATE_OF_BIRTH", single)
        self.assertEqual(single.loc[0, "AVG_ACCT_AGE_MONTHS"], 27)
        self.assertEqual(single.loc[0, "CREDIT_HISTORY_MONTHS"], 61)
        self.assertEqual(single.loc[0, "DOWN_PAYMENT"], 25000)
        self.assertTrue(pd.isna(single.loc[0, "PRI_UTILIZATION"]))

    def test_wilson_intervals_handle_zero_all_and_empty_events(self):
        low, high = wilson_interval([0, 10, 0], [10, 10, 0])
        self.assertAlmostEqual(low[0], 0)
        self.assertGreater(high[0], 0)
        self.assertLess(low[1], 1)
        self.assertAlmostEqual(high[1], 1)
        self.assertTrue(np.isnan(low[2]))
        midpoint_low, midpoint_high = wilson_interval(50, 100)
        self.assertAlmostEqual(float(midpoint_low), .4038315303659957)
        self.assertAlmostEqual(float(midpoint_high), .5961684696340043)

    def test_monitoring_psi_uses_reference_and_detects_missingness_or_shift(self):
        reference = np.arange(100, dtype=float)
        self.assertAlmostEqual(population_stability_index(reference, reference), 0)
        self.assertGreater(population_stability_index(reference, reference + 1000), 1)
        current = reference.copy()
        current[:50] = np.nan
        self.assertGreater(population_stability_index(reference, current), 0)

    def test_routing_boundaries_do_not_assign_automatic_lending_actions(self):
        routes = route_scores([.1, .2, .4, .6], .2, .6)
        self.assertEqual(list(routes), ["Lower-score queue", "Standard review", "Standard review", "Priority review"])
        with self.assertRaises(ValueError):
            route_scores([np.nan], .2, .6)
        with self.assertRaises(ValueError):
            route_scores([.3], .6, .2)

    def test_low_queue_threshold_uses_labels_and_minimum_volume_from_validation(self):
        scores = np.linspace(0, 1, 100)
        y = (scores > .3).astype(int)
        low = choose_low_threshold(y, scores, .7)
        mask = scores < low
        self.assertGreaterEqual(mask.sum(), 30)
        self.assertLessEqual(y[mask].mean(), .1)
        self.assertEqual(choose_low_threshold(np.ones(100), scores, .7), 0)

    def test_sensitivity_has_all_scenarios_and_obeys_each_capacity(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        scores = np.linspace(.1, .9, len(y))
        _, grid = choose_threshold(y, scores, Config(), 45, 8)
        scenarios = cost_sensitivity(grid, 100)
        self.assertEqual(len(scenarios), 45)
        self.assertTrue((scenarios.review_share <= scenarios.review_cap).all())

    def test_frozen_score_bootstrap_is_reproducible_and_bounded(self):
        y = np.array([0] * 50 + [1] * 50)
        scores = np.r_[np.linspace(.1, .7, 50), np.linspace(.3, .9, 50)]
        first = bootstrap_intervals(y, scores, .6, repetitions=20, seed=7)
        second = bootstrap_intervals(y, scores, .6, repetitions=20, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first.ci_low <= first.ci_high).all())
        self.assertTrue((first.ci_low >= 0).all() and (first.ci_high <= 1).all())


if __name__ == "__main__":
    unittest.main()
