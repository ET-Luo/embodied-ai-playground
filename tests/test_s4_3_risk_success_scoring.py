"""Score semantics, independent calibration, conservative tradeoffs and absent-class reporting."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import s4_3_risk_success_scoring as scoring


class ScoringTests(unittest.TestCase):
    def test_future_samples_only_and_no_continuous_collision_claim(self):
        # Segment crosses disk, but none of its sampled future endpoints is inside.
        trajectory=np.array([[[[-.05,-.09],[-.03,-.09],[.06,-.09],[.08,-.09]]]],np.float64)
        goal=np.array([[.08,-.09]],np.float64)
        scores=scoring.margins(trajectory,goal)
        self.assertLess(scores['risk'][0,0],0)
        self.assertGreater(scores['success'][0,0],0)
        trajectory[...,0,:]=scoring.CENTER  # Initial-only occupancy is deliberately excluded.
        changed=scoring.margins(trajectory,goal)
        for key in scores: np.testing.assert_array_equal(scores[key],changed[key])
        trajectory[...,-1,:]=scoring.CENTER
        both=scoring.margins(trajectory,scoring.CENTER[None])
        self.assertGreater(both['risk'][0,0],0)
        self.assertGreater(both['success'][0,0],0)  # Goal occupancy is independent of risk.

    def test_calibration_uses_group_max_and_not_heldout(self):
        pred={key:np.zeros((10,5)) for key in ('risk','success')}
        actual={key:np.column_stack([np.arange(10),np.zeros((10,4))]) for key in pred}
        buffers=scoring.calibrate(pred,actual)
        self.assertEqual(buffers,dict(risk=9.,success=9.))
        heldout={key:np.full((4,5),100.) for key in pred}
        scoring.evaluate(heldout,heldout,buffers,1.)
        self.assertEqual(buffers,dict(risk=9.,success=9.))

    def test_conservative_margin_tradeoffs_and_scale_zero(self):
        score={key:np.array([[-.02,-.005,0.,.005,.02]]) for key in ('risk','success')}
        buffer=dict(risk=.01,success=.01)
        raw=scoring.decisions(score,buffer,0)
        cautious=scoring.decisions(score,buffer,1)
        np.testing.assert_array_equal(raw['risk'],score['risk']>=0)
        np.testing.assert_array_equal(raw['success'],score['success']>=0)
        self.assertTrue(np.all(~raw['risk'] | cautious['risk']))
        self.assertTrue(np.all(~cautious['success'] | raw['success']))
        with self.assertRaises(ValueError): scoring.decisions(score,buffer,float('nan'))

    def test_absent_positive_class_and_full_denominator(self):
        c=scoring.confusion(np.array([[True,False,False]]),np.array([[False,False,False]]))
        self.assertEqual(c['total'],3)
        self.assertEqual(c['fp'],1)
        self.assertEqual(c['tn'],2)
        self.assertIsNone(c['false_negative_rate'])
        self.assertEqual(sum(c[k] for k in ('tp','fp','fn','tn')),3)

    def test_goal_is_decoded_from_current_observation_only(self):
        import s1_5_tiny_transformer_vla as vla
        rgb=vla.render(np.array([[4,12],[28,20]],np.int64))
        data=dict(image=np.stack([rgb,rgb]),text=np.array(vla.LANGUAGES))
        goals=scoring.observation_goals(data)
        np.testing.assert_allclose(goals,[[-.24,.08],[.24,-.08]],atol=1e-8)
        data['text']=np.array(['grab red','move to blue'])
        with self.assertRaises(ValueError): scoring.observation_goals(data)

    def test_probe_independence_and_stable_inputs(self):
        a=scoring.probe_contexts(431,64,'calibration')
        b=scoring.probe_contexts(432,32,'heldout')
        self.assertFalse(set(a['group_id']) & set(b['group_id']))
        self.assertEqual(a['action'].shape,(64,5,3,2))
        self.assertTrue(np.all(np.abs(a['action'])<=np.float32(.06)))
        self.assertTrue(np.all((a['gain']>=.8)&(a['gain']<1)))
        again=scoring.probe_contexts(431,64,'calibration')
        for key in a: np.testing.assert_array_equal(a[key],again[key])


if __name__=='__main__': unittest.main()
