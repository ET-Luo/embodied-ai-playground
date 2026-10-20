"""Candidate search protocol checks without fitting a model."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from s3_6_model_based_planning import candidate_bank, rollout, select, costs, outcome, ideal_state, ACTION_LIMIT


class PlanningTests(unittest.TestCase):
    def test_nested_candidates_and_explicit_zero(self):
        small, large = candidate_bank(7, 32), candidate_bank(7, 128)
        np.testing.assert_array_equal(small, large[:32])
        np.testing.assert_array_equal(large[0], np.zeros((10, 2)))
        self.assertEqual(large.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(large)), ACTION_LIMIT+1e-8)

    def test_recursive_rollout_and_no_reset_mixing(self):
        actions = np.array([[[.01, 0], [.02, 0]], [[-.01, 0], [-.02, 0]]], dtype=np.float32)
        states = rollout(np.array([0, .1]), actions, lambda s, a: s+a)
        np.testing.assert_allclose(states[:, -1], [[.03, .1], [-.03, .1]], atol=1e-9)
        np.testing.assert_array_equal(states[:, 0], [[0, .1], [0, .1]])

    def test_selection_never_calls_environment_and_uses_argmin(self):
        candidates = np.zeros((2, 3, 2), dtype=np.float32)
        candidates[1, :, 0] = .03
        with patch('s3_6_model_based_planning.step', side_effect=AssertionError('true plant leaked into planner')):
            index, predicted, score = select(np.zeros(2), candidates, ideal_state)
        self.assertEqual(index, 1)
        self.assertEqual(index, int(np.argmin(costs(predicted, candidates))))
        np.testing.assert_array_equal(score, costs(predicted, candidates))

    def test_failure_stage_and_candidate_specific_boundary_history(self):
        self.assertEqual(outcome(np.array([[0, 0], [.09, 0]]))['failure_stage'], 'success')
        result = outcome(np.array([[0, 0], [0, .21], [.09, 0]]))
        self.assertFalse(result['success']); self.assertEqual(result['first_violation_transition'], 0)
        self.assertEqual(result['failure_stage'], 'boundary_violation')
        self.assertEqual(outcome(np.array([[0, 0], [0, 0]]))['failure_stage'], 'goal_not_reached')

    def test_nonfinite_predictor_is_rejected_before_argmin(self):
        with self.assertRaises(ValueError):
            select(np.zeros(2), np.zeros((2, 3, 2)), lambda s, a: np.full_like(s, np.nan))


if __name__ == '__main__':
    unittest.main()
