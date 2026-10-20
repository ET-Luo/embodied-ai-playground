"""Primary priority, alternate choice, fallback safety and full rejection denominators."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import s4_4_rejection_alternate as gate


class GateTests(unittest.TestCase):
    def setUp(self): self.buffer=dict(risk=.005,success=.0155)

    def test_keep_primary_even_if_other_candidate_has_better_goal_score(self):
        scores=dict(risk=np.array([[-.02,-.02]]),success=np.array([[-.03,.008]]))
        chosen,rejected=gate.select(scores,self.buffer,1)
        self.assertEqual(chosen.tolist(),[0]);self.assertFalse(rejected.any())

    def test_reject_primary_choose_best_eligible_and_lowest_tie(self):
        scores=dict(risk=np.array([[.01,-.02,-.02,.03]]),success=np.array([[.009,.005,.005,.009]]))
        chosen,rejected=gate.select(scores,self.buffer,1)
        self.assertEqual(chosen.tolist(),[1]);self.assertEqual(rejected.tolist(),[[True,False,False,True]])
        no_alternate,_=gate.select(scores,self.buffer,1,alternate=False)
        self.assertEqual(no_alternate.tolist(),[-1])

    def test_fallback_minus_one_is_not_last_candidate(self):
        bank=np.full((2,5,3,2),.04,np.float32)
        selected=gate.selected_commands(bank,np.array([-1,4],np.int64))
        np.testing.assert_array_equal(selected[0],np.zeros((3,2),np.float32))
        np.testing.assert_array_equal(selected[1],bank[1,4])
        with self.assertRaises(ValueError):gate.selected_commands(bank,np.array([-2,4],np.int64))

    def test_success_requirement_can_force_all_fallbacks(self):
        scores=dict(risk=np.full((2,5),-.03),success=np.full((2,5),.01))
        loose,_=gate.select(scores,self.buffer,1)
        strict,rejected=gate.select(scores,self.buffer,1,require_success=True)
        self.assertEqual(loose.tolist(),[0,0]);self.assertEqual(strict.tolist(),[-1,-1])
        self.assertTrue(rejected.all())

    def test_risk_free_miss_is_not_a_good_action(self):
        counts=gate.rejection_counts(np.array([[True,True,False,True]]),
            np.array([[True,False,False,False]]),np.array([[True,True,True,False]]))
        self.assertEqual(counts['rejected_bad'],1)
        self.assertEqual(counts['rejected_risk_free'],2)
        self.assertEqual(counts['actual_good'],2)
        self.assertEqual(counts['false_rejected_good'],1)
        self.assertEqual(counts['allowed_good'],1)
        self.assertEqual(counts['actual_bad']+counts['actual_risk_free'],counts['candidates'])

    def test_zero_commands_can_stay_in_monitoring_disk(self):
        initial=gate.scoring.CENTER[None].astype(np.float32)
        trajectory=gate.replay(initial,np.zeros((1,3,2),np.float32),np.array([.65]))
        scores=gate.scoring.margins(trajectory[:,None],np.array([[.2,.2]]))
        self.assertTrue((scores['risk']>=0).all());self.assertTrue((scores['success']<0).all())

    def test_scores_only_selection_is_stable_and_rejects_nan(self):
        scores=dict(risk=np.array([[.01,-.02,-.02]]),success=np.array([[-.03,-.02,-.01]]))
        chosen,_=gate.select(scores,self.buffer,1)
        labels=np.array([[True,False,True]])
        gate.rejection_counts(np.array([[True,False,False]]),labels,~labels)
        # Swapping evaluator labels never enters select's input or changes its output.
        labels[:]=~labels
        np.testing.assert_array_equal(chosen,gate.select(scores,self.buffer,1)[0])
        scores['risk'][0,0]=np.nan
        with self.assertRaises(ValueError):gate.select(scores,self.buffer,1)


if __name__=='__main__':unittest.main()
