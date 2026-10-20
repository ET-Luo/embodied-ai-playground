"""Check score semantics and diagnostic labels without fitting models."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from s3_3_model_uncertainty import confusion, disagreement, summarize


class UncertaintyTests(unittest.TestCase):
    def test_population_spread_units_and_denominator(self):
        predictions = np.array([[[0., 0.]], [[.002, 0.]]])
        mean, score = disagreement(predictions)
        np.testing.assert_allclose(mean, [[.001, 0]])
        np.testing.assert_allclose(score, [.001/np.sqrt(2)])

    def test_common_bias_and_permutation_leave_score_unchanged(self):
        p = np.random.default_rng(7).normal(0, .01, (5, 8, 2))
        _, score = disagreement(p)
        np.testing.assert_allclose(disagreement(p[[4, 1, 0, 3, 2]])[1], score, atol=1e-16)
        np.testing.assert_allclose(disagreement(p+[.02, 0])[1], score, atol=1e-16)

    def test_same_input_predictions_cannot_detect_changed_target(self):
        p = np.zeros((5, 4, 2))
        correct = summarize(p, np.zeros((4, 2)), .001)[0]
        shifted = summarize(p, np.tile([.01, 0], (4, 1)), .001)[0]
        self.assertEqual(correct['disagreement_median_m'], shifted['disagreement_median_m'])
        self.assertEqual(shifted['confusion']['fn'], 4)
        self.assertGreater(shifted['ensemble_mean']['rmse_m'], .007)
        self.assertIsNone(shifted['score_error_pearson'])

    def test_strict_diagnostic_threshold_and_full_denominator(self):
        result = confusion(np.array([.1, .2, .3, .4]), np.array([0, .006, 0, .006]), .2)
        self.assertEqual(result, {'tp': 1, 'fp': 1, 'fn': 1, 'tn': 1})
        self.assertEqual(sum(result.values()), 4)

    def test_bad_score_input_rejected(self):
        for p in (np.zeros((1, 5, 2)), np.zeros((2, 3)), np.full((2, 3, 2), np.nan)):
            with self.assertRaises(ValueError):
                disagreement(p)


if __name__ == '__main__':
    unittest.main()
