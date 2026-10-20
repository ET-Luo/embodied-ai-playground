"""Protocol tests independent of any local trained checkpoint."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from s3_2_multistep_rollout import evaluate, extended_episodes, summarize
from s3_1_state_dynamics import dataset, transition


class RolloutProtocolTests(unittest.TestCase):
    def test_truth_cannot_leak_into_autoregression(self):
        truth = np.zeros((2, 6, 2), dtype=np.float32)
        actions = np.full((2, 5, 2), .01, dtype=np.float32)
        tf, ar = evaluate(lambda s, a: s+a, truth, actions)
        changed = truth.copy()
        changed[:, 2:] = 100
        changed_tf, changed_ar = evaluate(lambda s, a: s+a, changed, actions)
        np.testing.assert_array_equal(ar, changed_ar)
        np.testing.assert_array_equal(tf[:, :2], changed_tf[:, :2])
        self.assertGreater(np.max(np.abs(tf-changed_tf)), 99)

    def test_exact_oracle_and_horizon_metrics(self):
        states, actions = dataset()
        oracle = lambda s, a: transition(s.astype(np.float32), a.astype(np.float32))
        tf, ar = evaluate(oracle, states[:2], actions[:2])
        np.testing.assert_array_equal(tf, states[:2, 1:])
        np.testing.assert_array_equal(ar, states[:2])
        result = summarize(tf, ar, states[:2], [1, 5, 20])
        self.assertEqual(result['20']['prefix_transitions'], 40)
        self.assertEqual(result['20']['ar_endpoint_at_h']['rmse_m'], 0)

    def test_known_bias_accumulates_and_episode_resets(self):
        truth = np.zeros((3, 21, 2), dtype=np.float32)
        truth[:, :, 0] = np.arange(3)[:, None]
        actions = np.zeros((3, 20, 2), dtype=np.float32)
        bias = np.array([.001, 0])
        tf, ar = evaluate(lambda s, a: s+bias, truth, actions)
        np.testing.assert_allclose(tf-truth[:, 1:], np.broadcast_to(bias, tf.shape), atol=1e-15)
        np.testing.assert_allclose(ar[:, -1]-truth[:, 0], np.tile(20*bias, (3, 1)), atol=1e-14)
        self.assertEqual(summarize(tf, ar, truth, [20])['20']['episodes'], 3)

    def test_extension_preserves_source_and_every_prefix(self):
        states, actions = dataset()
        data = {'state_m': states, 'action_m': actions, 'validation_episode_ids': np.arange(30, 40)}
        s40, a40 = extended_episodes(data, 40)
        s100, a100 = extended_episodes(data, 100)
        np.testing.assert_array_equal(s40[:, :21], states[30:])
        np.testing.assert_array_equal(a40[:, :20], actions[30:])
        np.testing.assert_array_equal(s40, s100[:, :41])
        np.testing.assert_array_equal(a40, a100[:, :40])


if __name__ == '__main__':
    unittest.main()
