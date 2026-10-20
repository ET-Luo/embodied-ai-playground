"""Feedback and prefix timing protocol, independent of trained checkpoint."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from s3_7_mpc_replanning import run_episode, planning_bank


def identity_predictor(s, a):
    return s+a


class ReplanningTests(unittest.TestCase):
    def test_feedback_observes_prefix_end_and_short_final_prefix(self):
        def plant(s, a, t):
            result = s+a
            if t == 4:
                result[:, 1] += .08
            return result.astype(np.float32)
        initial = np.zeros(2, dtype=np.float32)
        state, commands, plans = run_episode(initial, 4, 8, 4, True, identity_predictor, plant)
        self.assertEqual([p['t'] for p in plans], [0, 4, 8])
        self.assertEqual([p['remaining_horizon'] for p in plans], [10, 6, 2])
        self.assertEqual([p['executed_prefix'] for p in plans], [4, 4, 2])
        self.assertEqual(commands.shape, (10, 2)); self.assertEqual(state.shape, (11, 2))
        for p in plans:
            np.testing.assert_array_equal(p['planning_state_m'], state[p['t']])

    def test_no_feedback_actions_ignore_plant_change(self):
        initial = np.zeros(2, dtype=np.float32)
        def plant(s, a, t):
            result = s+a
            result[:, 1] += .02
            return result.astype(np.float32)
        _, original, _ = run_episode(initial, 4, 8, 1, False, identity_predictor, lambda s, a, t: (s+a).astype(np.float32))
        _, disturbed, _ = run_episode(initial, 4, 8, 1, False, identity_predictor, plant)
        np.testing.assert_array_equal(original, disturbed)

    def test_full_prefix_equals_single_shot_despite_feedback_flag(self):
        initial = np.zeros(2, dtype=np.float32)
        plant = lambda s, a, t: (s+.65*a).astype(np.float32)
        first = run_episode(initial, 4, 8, 10, False, identity_predictor, plant)
        second = run_episode(initial, 4, 8, 10, True, identity_predictor, plant)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(len(first[2]), 1)

    def test_warm_start_is_only_unexecuted_suffix(self):
        warm = np.full((6, 2), .01, dtype=np.float32)
        bank = planning_bank(4, 4, 8, warm)
        np.testing.assert_array_equal(bank[-1], warm)
        np.testing.assert_array_equal(bank[0], np.zeros((6, 2)))
        with self.assertRaises(ValueError):
            planning_bank(4, 4, 8, np.zeros((10, 2)))


if __name__ == '__main__':
    unittest.main()
