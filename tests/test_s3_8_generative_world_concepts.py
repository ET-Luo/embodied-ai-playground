"""Action and joint-distribution counterexamples, independent of external models."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from s3_8_generative_world_concepts import build, render, temporal_violation, sample_summary, component_count


class GenerativeConceptTests(unittest.TestCase):
    def test_identical_contexts_different_action_futures(self):
        data = build(1)
        truth = data['truth_state_m']
        self.assertEqual(truth.dtype, np.float64)
        images = render(truth)
        np.testing.assert_array_equal(images[:, 0, 0, 0], images[:, 1, 1, 0])
        np.testing.assert_array_equal(images[:, 0, 0, 0], images[:, 0, 1, 0])
        np.testing.assert_allclose(truth[:, 1, :, -1, 0]-truth[:, 0, :, -1, 0], .24, atol=1e-8)

    def test_persistent_mode_valid_but_resampling_breaks_temporal_joint(self):
        data = build(16)
        self.assertFalse(temporal_violation(data['persistent_state_m'],data['action_m'][:, :, None]).any())
        self.assertGreater(temporal_violation(data['frame_resampled_state_m'],data['action_m'][:, :, None]).mean(), .3)
        self.assertTrue(temporal_violation(data['mean_state_m'],data['action_m']).all())
        self.assertTrue(temporal_violation(data['action_ignored_state_m'],data['action_m'][:, :, None]).all())

    def test_more_samples_preserve_prefix_and_coverage_is_not_single_sample_accuracy(self):
        single, many = build(1), build(16)
        np.testing.assert_array_equal(single['persistent_state_m'],many['persistent_state_m'][:, :, :1])
        low = sample_summary(single['persistent_state_m'],single['truth_state_m'],single['action_m'])
        high = sample_summary(many['persistent_state_m'],many['truth_state_m'],many['action_m'])
        self.assertEqual(low['exact_support_coverage'], .5)
        self.assertEqual(high['exact_support_coverage'], 1.)
        self.assertAlmostEqual(low['all_sample_path_mse_m2'],high['all_sample_path_mse_m2'])
        self.assertLessEqual(high['best_of_k_path_mse_m2'],low['best_of_k_path_mse_m2'])

    def test_pixel_mean_is_not_a_valid_single_object_frame(self):
        images = render(build(1)['truth_state_m'])
        mean = images.astype(float).mean(2)
        self.assertEqual(component_count(images[0, 1, 0, -1]), 1)
        self.assertEqual(component_count(mean[0, 1, -1]), 2)


if __name__ == '__main__':
    unittest.main()
