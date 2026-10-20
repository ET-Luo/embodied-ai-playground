"""Feedback causality, E-prefix alignment, evaluator isolation and rejection records."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import s4_5_receding_horizon as loop


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        coefficients=np.zeros((5,2),np.float64);coefficients[2,0]=1;coefficients[3,1]=1
        self.model=dict(coefficients=coefficients,input_mean=np.zeros(4),input_std=np.ones(4))
        self.buffers=dict(risk=.005,success=.0155)
        self.case=dict(case_id='test',condition='nominal',scene_id='test',centers_uv=[[4,4],[28,28]],
            text='move to red',initial_m=[0.,-.12],evaluation_goal_m=[-.24,.24],plant_gain=1.,
            kick_after_transition=2,kick_m=[0.,0.])
        self.bank=np.zeros((5,3,2),np.float32);self.bank[:,:,0]=.005

    def run_mock(self,case=None,execute=1,feedback=True,strict=False):
        with patch.object(loop.proposal,'propose',return_value=self.bank.copy()):
            return loop.run_episode(self.case if case is None else case,None,None,self.model,self.buffers,1.,execute,feedback,strict)

    def test_e3_prefixes_and_discarded_suffix(self):
        one=self.run_mock();three=self.run_mock(execute=3)
        self.assertEqual([p['t'] for p in one['plans']],list(range(8)))
        self.assertEqual([p['t'] for p in three['plans']],[0,3,6])
        self.assertEqual([p['executed_prefix'] for p in three['plans']],[3,3,2])
        self.assertEqual(three['state_m'].shape,(9,2));self.assertEqual(three['action_m'].shape,(8,2))
        self.assertEqual(np.array(three['plans'][-1]['predicted_bank_m']).shape,(5,4,2))
        np.testing.assert_array_equal(one['plans'][0]['candidate_bank_m'],three['plans'][0]['candidate_bank_m'])

    def test_observation_boundary_after_kick(self):
        case=deepcopy(self.case);case['kick_m']=[.12,0.]
        one=self.run_mock(case);three=self.run_mock(case,execute=3)
        np.testing.assert_array_equal(one['plans'][2]['planning_state_m'],one['state_m'][2].tolist())
        np.testing.assert_array_equal(three['plans'][1]['planning_state_m'],three['state_m'][3].tolist())
        # E3 executes slot2 from tick0's stale chunk, then observes at tick3.
        np.testing.assert_array_equal(three['action_m'][:3],self.bank[0])

    def test_prediction_only_does_not_read_gain_or_kick_observations(self):
        nominal=self.run_mock(feedback=False)
        case=deepcopy(self.case);case.update(plant_gain=.65,kick_m=[.12,0.])
        shifted=self.run_mock(case,feedback=False)
        np.testing.assert_array_equal(nominal['action_m'],shifted['action_m'])
        for a,b in zip(nominal['plans'],shifted['plans']):
            np.testing.assert_array_equal(a['planning_state_m'],b['planning_state_m'])
            np.testing.assert_array_equal(a['predicted_bank_m'],b['predicted_bank_m'])
        self.assertFalse(np.array_equal(nominal['state_m'],shifted['state_m']))

    def test_evaluator_goal_cannot_change_control_or_early_stop(self):
        first=self.run_mock()
        case=deepcopy(self.case);case['evaluation_goal_m']=first['state_m'][-1].tolist()
        second=self.run_mock(case)
        np.testing.assert_array_equal(first['action_m'],second['action_m'])
        np.testing.assert_array_equal(first['state_m'],second['state_m'])
        self.assertFalse(first['goal_success']);self.assertTrue(second['goal_success'])
        self.assertEqual(second['completed_steps'],8)

    def test_strict_fallback_is_executed_and_replanned(self):
        result=self.run_mock(execute=3,strict=True)
        self.assertEqual([p['chosen'] for p in result['plans']],[-1,-1,-1])
        np.testing.assert_array_equal(result['action_m'],np.zeros((8,2),np.float32))
        np.testing.assert_array_equal(result['state_m'],np.repeat(np.array([[0.,-.12]],np.float32),9,axis=0))

    def test_unknown_input_has_one_state_no_fake_zero_actions(self):
        case=deepcopy(self.case);case['text']='grab red block'
        result=loop.run_episode(case,None,loop.proposal.vla.fit_vocabulary(loop.proposal.vla.LANGUAGES),
            self.model,self.buffers,1.,1,True)
        self.assertEqual(result['completed_steps'],0);self.assertEqual(result['state_m'].shape,(1,2))
        self.assertEqual(result['action_m'].shape,(0,2));self.assertEqual(len(result['plans']),1)
        self.assertEqual(result['plans'][0]['executed_prefix'],0)
        self.assertEqual(result['failure_stage'],'policy_input_or_output_rejection')
        summary=loop.summarize([result]);self.assertIsNone(summary['executed_prediction_rmse_m'])
        self.assertEqual(summary['attempts'],1);self.assertEqual(summary['rejected_attempts'],1)

    def test_executed_error_uses_only_prefix_and_real_scalar_denominator(self):
        result=self.run_mock(execute=3)
        error=sum(np.square(np.array(p['actual_executed_m'])-np.array(p['predicted_executed_m'])).sum()
                  for p in result['plans'])
        summary=loop.summarize([result])
        self.assertEqual(summary['executed_prediction_scalar_denominator'],16)
        self.assertAlmostEqual(summary['executed_prediction_rmse_m'],np.sqrt(error/16))


if __name__=='__main__':unittest.main()
