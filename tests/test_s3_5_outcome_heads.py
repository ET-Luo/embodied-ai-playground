"""Outcome semantics, classification thresholds and multi-head loss checks."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from s3_5_outcome_heads import annotate, step, binary_metrics, sigmoid, loss_and_gradients, WALL_X


class OutcomeHeadTests(unittest.TestCase):
    def test_success_contact_and_reward_are_distinct(self):
        states = np.array([[.12, 0], [.12, .15], [-.15, 0]], dtype=np.float32)
        actions = np.array([[.04, 0], [.04, 0], [.04, 0]], dtype=np.float32)
        next_states, contact = step(states, actions)
        rewards, labels = annotate(states, next_states, contact)
        np.testing.assert_array_equal(labels[:, :2], [[True, True], [False, True], [False, False]])
        self.assertLess(rewards[0], 0)  # Goal occupancy can have negative shaped reward.
        self.assertGreater(rewards[2], 0)  # Progress is not success.
        self.assertFalse(labels[:, 2].any())  # Contact does not imply failure.

    def test_labels_refer_to_execution_result(self):
        state = np.array([[.03, 0], [0, .19]], dtype=np.float32)
        action = np.array([[.04, 0], [0, .05]], dtype=np.float32)
        next_state, contact = step(state, action)
        _, before = annotate(state, state, np.zeros(2, dtype=bool))
        _, after = annotate(state, next_state, contact)
        self.assertFalse(before[0, 0]); self.assertTrue(after[0, 0])
        self.assertFalse(before[1, 2]); self.assertTrue(after[1, 2])

    def test_wall_clips_position_but_not_all_contact_is_success(self):
        state = np.array([[.11, .18]], dtype=np.float32)
        next_state, contact = step(state, np.array([[.06, 0]], dtype=np.float32))
        self.assertTrue(contact[0]); self.assertEqual(next_state[0, 0], WALL_X)
        self.assertFalse(annotate(state, next_state, contact)[1][0, 0])

    def test_threshold_reduction_cannot_increase_false_negatives(self):
        p = np.array([.1, .2, .5, .9])
        target = np.array([False, True, False, True])
        low, high = binary_metrics(p, target, .2), binary_metrics(p, target, .5)
        self.assertEqual(low['tp'], 2); self.assertEqual(low['fn'], 0)
        self.assertLessEqual(low['fn'], high['fn']); self.assertGreaterEqual(low['fp'], high['fp'])
        self.assertEqual(sum(low[k] for k in ('tp','fp','fn','tn')), 4)
        self.assertEqual(low['brier'], high['brier'])

    def test_each_output_head_gradient_and_extreme_logits(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(6, 4))
        y = np.column_stack([rng.normal(size=6), rng.integers(0, 2, (6, 3))])
        weights = [rng.normal(0, .1, (4, 32)), np.zeros(32), rng.normal(0, .1, (32, 4)), np.zeros(4)]
        _, grad = loss_and_gradients(x, y, weights)
        for j in range(4):
            weights[3][j] += 1e-5
            plus = loss_and_gradients(x, y, weights)[0]
            weights[3][j] -= 2e-5
            minus = loss_and_gradients(x, y, weights)[0]
            weights[3][j] += 1e-5
            self.assertAlmostEqual((plus-minus)/2e-5, grad[3][j], places=7)
        weights[3][1:] = [1000, -1000, 1000]
        loss, grads = loss_and_gradients(x, y, weights)
        self.assertTrue(np.isfinite(loss)); self.assertTrue(all(np.isfinite(g).all() for g in grads))
        np.testing.assert_array_equal(sigmoid(np.array([-1000., 1000.])), [0, 1])


if __name__ == '__main__':
    unittest.main()
