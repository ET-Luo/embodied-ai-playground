"""Contract failures and nested candidate provenance at the integration boundary."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import s4_1_vla_proposal as proposal


class ProposalTests(unittest.TestCase):
    def test_semantic_mismatches(self):
        bank = np.zeros((5, 3, 2), np.float32)
        for key, value in [('unit', 'mm'), ('frame', 'camera_xy'), ('control_dt_s', .2),
                           ('representation', 'absolute_position'), ('embodiment', 'wall_xy')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                proposal.validate_proposals(bank, dict(proposal.CONTRACT, **{key: value}))

    def test_invalid_payloads(self):
        for bank in [np.zeros((3, 2), np.float32), np.zeros((0, 3, 2), np.float32),
                     np.zeros((1, 3, 2), np.float64), np.full((1, 3, 2), np.nan, np.float32),
                     np.full((1, 3, 2), .061, np.float32)]:
            with self.assertRaises(ValueError): proposal.validate_proposals(bank, proposal.CONTRACT)

    def test_prefix_and_clipping(self):
        base = np.array([[.059, -.059]]*3, np.float32)
        with patch.object(proposal.vla, 'predict', return_value=base):
            one = proposal.propose(None, None, None, None, None, 1)
            five = proposal.propose(None, None, None, None, None, 5)
        np.testing.assert_array_equal(one, five[:1])
        np.testing.assert_array_equal(five[0], base)
        self.assertEqual(five[1, 0, 0], np.float32(.06))
        np.testing.assert_array_equal(base, np.array([[.059, -.059]]*3, np.float32))


if __name__ == '__main__': unittest.main()
