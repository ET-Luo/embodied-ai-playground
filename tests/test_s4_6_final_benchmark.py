"""Matched protocol separation, deployable timing boundary and evaluator isolation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import s4_6_final_benchmark as bench


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.case=dict(case_id='test',seed=461,condition='nominal',scene_id='test',centers_uv=[[4,4],[28,28]],
            text='move to red',initial_m=[0.,-.12],evaluation_goal_m=[-.24,.24],plant_gain=1.,
            kick_after_transition=2,kick_m=[0.,0.])
        c=np.zeros((5,2));c[2,0]=c[3,1]=1
        self.model=dict(coefficients=c,input_mean=np.zeros(4),input_std=np.ones(4))
        self.buffers=dict(risk=.005,success=.0155)
        self.bank=np.zeros((5,3,2),np.float32);self.bank[:,:,0]=.005

    def proposer(self,rgb,state,text,weights,vocab,candidates):return self.bank[:candidates].copy()

    def run_mock(self,case=None,method='vla_only',execute=1,model=None):
        with patch.object(bench.loop.proposal,'propose',side_effect=self.proposer):
            return bench.run_episode(self.case if case is None else case,method,None,None,
                self.model if model is None else model,self.buffers,1.,False,execute)

    def test_protocol_excludes_train_validation_and_seed_overlap(self):
        xy=[(u,v) for v in (4,12,20,28) for u in (4,12,20,28)]
        layouts=[(xy[r],xy[b]) for r in range(16) for b in range(16) if r!=b]
        rows=[dict(centers_uv=[list(p) for p in pair]) for pair in layouts[:40]]
        contract=dict(split=dict(train=rows[:32],validation=rows[32:]))
        cases=bench.fresh_protocol(contract)
        self.assertEqual(cases,bench.fresh_protocol(contract));self.assertEqual(len(cases),128)
        self.assertEqual(len({c['case_id'] for c in cases}),128)
        per_seed={s:{tuple(map(tuple,c['centers_uv'])) for c in cases if c['seed']==s} for s in bench.SEEDS}
        self.assertEqual([len(per_seed[s]) for s in bench.SEEDS],[8,8])
        self.assertFalse(per_seed[461]&per_seed[462])
        self.assertFalse(set.union(*per_seed.values())&set(layouts[:40]))

    def test_vla_only_control_never_calls_world_model_or_goal_decoder(self):
        rgb=bench.loop.proposal.vla.render(np.array(self.case['centers_uv'],np.int64))
        with patch.object(bench.loop.proposal,'propose',side_effect=self.proposer), \
             patch.object(bench.loop.gate.scoring.wm,'imagine',side_effect=AssertionError('WM in baseline control')), \
             patch.object(bench.loop.gate.scoring,'observation_goals',side_effect=AssertionError('goal in baseline control')):
            result=bench.control(rgb,np.array([0.,-.12],np.float32),'move to red','vla_only',None,None,self.model,self.buffers,1.,False)
        self.assertEqual(result['chosen'],0);self.assertEqual(result['bank'].shape,(1,3,2))
        self.assertIsNone(result['prediction']);self.assertIsNone(result['world_model_ms']);self.assertIsNone(result['gate_ms'])

    def test_shadow_model_does_not_change_baseline_commands(self):
        first=self.run_mock()
        wrong={k:v.copy() for k,v in self.model.items()};wrong['coefficients']*=.1
        other=self.run_mock(model=wrong)
        np.testing.assert_array_equal(first['action_m'],other['action_m'])
        np.testing.assert_array_equal(first['state_m'],other['state_m'])
        self.assertNotEqual(first['plans'][0]['prediction_sse_m2'],other['plans'][0]['prediction_sse_m2'])
        self.assertIsNone(bench.summarize([first])['rejection'])

    def test_evaluator_goal_only_changes_outcome_diagnostics(self):
        for method in bench.METHODS:
            first=self.run_mock(method=method)
            case=deepcopy(self.case);case['evaluation_goal_m']=first['state_m'][-1].tolist()
            second=self.run_mock(case,method)
            np.testing.assert_array_equal(first['action_m'],second['action_m'])
            np.testing.assert_array_equal(first['state_m'],second['state_m'])
            self.assertFalse(first['success']);self.assertTrue(second['success'])

    def test_counterfactual_kick_alignment_and_independent_candidates(self):
        case=deepcopy(self.case);case['kick_m']=[.12,0.]
        start=np.array([0.,-.12],np.float32)
        trace=bench.actual_future(start,self.bank,case,0)
        self.assertEqual(trace.shape,(5,4,2))
        np.testing.assert_allclose(trace[:,2]-trace[:,1],self.bank[:,1]+[.12,0.],atol=1e-8)
        after=bench.actual_future(start,self.bank,case,2)
        np.testing.assert_allclose(after[:,2]-after[:,1],self.bank[:,1],atol=1e-8)
        np.testing.assert_array_equal(trace[0],trace[-1])

    def test_matched_execution_schedule_and_primary_when_gate_passes(self):
        first=self.run_mock(execute=3);second=self.run_mock(method='vla_wm',execute=3)
        for row in (first,second):
            self.assertEqual([p['t'] for p in row['plans']],[0,3,6])
            self.assertEqual([p['executed_prefix'] for p in row['plans']],[3,3,2])
        np.testing.assert_array_equal(first['action_m'],second['action_m'])
        np.testing.assert_array_equal(first['state_m'],second['state_m'])
        self.assertEqual(bench.summarize([second])['rejection']['candidates'],15)
        self.assertEqual(bench.summarize([second])['executed_prediction_scalar_denominator'],16)

    def test_refusal_and_missing_pair_denominators(self):
        case=deepcopy(self.case);case['text']='grab red block'
        result=bench.run_episode(case,'vla_only',None,bench.loop.proposal.vla.fit_vocabulary(bench.loop.proposal.vla.LANGUAGES),
            self.model,self.buffers,1.,False,1)
        self.assertEqual(result['state_m'].shape,(1,2));self.assertEqual(result['action_m'].shape,(0,2))
        summary=bench.summarize([result]);self.assertEqual(summary['attempts'],1)
        self.assertIsNone(summary['executed_prediction_rmse_m']);self.assertIsNone(summary['latency_ms']['control_ms'])
        with self.assertRaises(ValueError):bench.paired_counts([result])


if __name__=='__main__':unittest.main()
