"""Representation/data-flow checks independent of trained dynamics artifacts."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from s3_1_state_dynamics import dataset
from s3_4_latent_state import SENSOR, observe, fit_encoder, encode, decode, physical_state, evaluate


class LatentStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states, cls.actions = dataset()
        cls.observations = observe(cls.states)

    def test_rank2_recovers_sensor_and_physical_state(self):
        encoder = fit_encoder(self.observations[:30].reshape(-1, 6), 2)
        recovered = decode(encode(self.observations[30:], encoder), encoder)
        np.testing.assert_allclose(recovered, self.observations[30:], atol=5e-8, rtol=0)
        np.testing.assert_allclose(physical_state(recovered), self.states[30:], atol=5e-8, rtol=0)
        np.testing.assert_allclose(encoder['components'].T@encoder['components'], np.eye(2), atol=1e-14)

    def test_rank1_has_physically_distinct_same_latent_states(self):
        encoder = fit_encoder(self.observations[:30].reshape(-1, 6), 1)
        direction = SENSOR.T@(encoder['components'][:, 0]/encoder['observation_std'])
        null = np.array([-direction[1], direction[0]])
        null /= np.linalg.norm(null)
        pair = np.stack([-.05*null, .05*null])
        latent = encode(pair@SENSOR.T, encoder)
        np.testing.assert_allclose(latent[0], latent[1], atol=1e-14)
        self.assertGreater(np.linalg.norm(pair[0]-pair[1]), .09)

    def test_autoregression_never_reads_future_observations(self):
        encoder = fit_encoder(self.observations[:30].reshape(-1, 6), 2)
        weights = [np.zeros((4, 32)), np.zeros(32), np.zeros((32, 2)), np.zeros(2)]
        stats = {'input_mean': np.zeros(4), 'input_std': np.ones(4),
                 'delta_mean': np.array([.001, -.001]), 'delta_std': np.ones(2)}
        truth, obs, action = self.states[30:], self.observations[30:], self.actions[30:]
        _, first = evaluate(truth, obs, action, encoder, weights, stats)
        changed = obs.copy()
        changed[:, 1:] += .1
        _, second = evaluate(truth, changed, action, encoder, weights, stats)
        np.testing.assert_array_equal(first['ar_latent'], second['ar_latent'])
        self.assertGreater(np.max(np.abs(first['target_next_latent']-second['target_next_latent'])), .1)
        np.testing.assert_array_equal(first['predicted_next_latent'].reshape(10, 20, 2)[:, 0], first['ar_latent'][:, 1])

    def test_encoder_rejects_degenerate_or_invalid_input(self):
        for observation, dimension in ((np.zeros((8, 6)), 2), (np.ones((8, 5)), 2),
                                       (np.full((8, 6), np.nan), 2), (np.ones((8, 6)), 3)):
            with self.assertRaises(ValueError):
                fit_encoder(observation, dimension)


if __name__ == '__main__':
    unittest.main()
