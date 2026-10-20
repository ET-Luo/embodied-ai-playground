"""Data separation, causal rollout, independent replays and integration semantics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import s4_2_world_model_verification as wm


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.states, self.actions, self.train, self.val = wm.training_episodes()
        self.model = wm.fit_model(self.states, self.actions, self.train)

    def test_holdout_never_changes_fit_or_statistics(self):
        self.assertFalse(set(self.train) & set(self.val))
        changed_states, changed_actions = self.states.copy(), self.actions.copy()
        changed_states[self.val] += 100
        changed_actions[self.val] *= -3
        changed = wm.fit_model(changed_states, changed_actions, self.train)
        for key in changed: np.testing.assert_array_equal(changed[key], self.model[key])
        prediction = wm.predict_next(self.states[self.val, :-1], self.actions[self.val], self.model)
        self.assertLess(np.sqrt(np.mean((prediction-self.states[self.val, 1:])**2)), 1e-7)

    def test_replay_resets_and_action_time_alignment(self):
        initial = np.array([[[.1, -.12], [.1, -.12]]], np.float32)
        actions = np.array([[[[.01, 0], [0, .02], [-.01, 0]],
                             [[-.02, 0], [0, 0], [0, -.01]]]], np.float32)
        trace = wm.plant_rollout(initial, actions, .65)
        self.assertEqual(trace.shape, (1, 2, 4, 2))
        np.testing.assert_array_equal(trace[..., 0, :], initial)
        np.testing.assert_allclose(trace[..., 1:, :]-trace[..., :-1, :], .65*actions, atol=1e-8)
        for k in range(2):
            separate = wm.plant_rollout(initial[:, k], actions[:, k], .65)
            np.testing.assert_array_equal(trace[:, k], separate)

    def test_gain_shift_is_not_an_input_to_prediction(self):
        initial = self.states[self.val, 0]
        actions = self.actions[self.val, :3]
        predicted = wm.imagine(initial, actions, self.model)
        nominal = wm.plant_rollout(initial, actions, 1.)
        shifted = wm.plant_rollout(initial, actions, .65)
        np.testing.assert_allclose(predicted, nominal, atol=1e-7)
        np.testing.assert_allclose(predicted-shifted,
            np.concatenate([np.zeros_like(initial[:, None]), .35*np.cumsum(actions, axis=-2)], axis=-2), atol=1e-7)
        tf = wm.predict_next(shifted[:, :-1], actions, self.model)
        np.testing.assert_allclose(tf-shifted[:, 1:], .35*actions, atol=1e-7)
        shifted[:] = 999  # Evaluation futures cannot mutate the already-computed AR prediction.
        np.testing.assert_array_equal(predicted, wm.imagine(initial, actions, self.model))

    def test_mismatched_model_contract_and_bad_scale(self):
        for field, value in [('embodiment', 'wall_xy'), ('unit', 'mm'), ('frame', 'camera'),
                             ('control_dt_s', .2), ('representation', 'absolute_position')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                wm.validate_model(self.model, dict(wm.MODEL_CONTRACT, **{field:value}))
        bad = {key:value.copy() for key,value in self.model.items()}
        bad['input_std'][0] = 0
        with self.assertRaises(ValueError): wm.validate_model(bad, wm.MODEL_CONTRACT)

    def test_source_contract_rejected_before_payload_load(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source/'contract.json').write_text(json.dumps(dict(wm.CONTRACT, unit='mm')))
            (source/'report.json').write_text('{}')
            (source/'proposals.npz').write_bytes(b'not a valid NPZ')
            with self.assertRaisesRegex(ValueError, 'semantics/embodiment'):
                wm.load_proposals(source)


if __name__ == '__main__': unittest.main()
