"""Transparent geometric margins, group calibration and held-out FP/FN; no action selection."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require
from s1_7_closed_loop_evaluation import sha256
from s4_1_vla_proposal import CONTRACT
import s4_2_world_model_verification as wm

CENTER = np.array([.015, -.09], np.float64)
RADIUS = .02
TOLERANCE = .01
FILES = ('contract.json', 'report.json', 'model.npz', 'verification.npz')


def margins(trajectory, goal):
    """Positive risk = a sampled future state in disk; positive success = terminal L-infinity goal occupancy."""
    require(trajectory.ndim == 4 and trajectory.shape[-2:] == (4, 2)
            and np.isfinite(trajectory).all(), 'finite [group,candidate,4,2] trajectory')
    require(goal.shape == (len(trajectory), 2) and np.isfinite(goal).all(), 'current known goal [group,2]')
    future = trajectory[..., 1:, :].astype(np.float64)
    risk = RADIUS - np.linalg.norm(future-CENTER, axis=-1).min(axis=-1)
    success = TOLERANCE - np.abs(future[..., -1, :]-goal[:, None]).max(axis=-1)
    return dict(risk=risk, success=success)


def calibrate(predicted, actual):
    # One residual per independent reset group, not one per correlated candidate.
    return {key:float(np.quantile(np.abs(predicted[key]-actual[key]).max(axis=1), .9, method='higher'))
            for key in ('risk', 'success')}


def decisions(scores, buffer, scale):
    require(np.isfinite(scale) and scale >= 0, 'finite nonnegative margin scale')
    return dict(risk=scores['risk'] >= -scale*buffer['risk'],
                success=scores['success'] >= scale*buffer['success'])


def confusion(predicted, actual):
    require(predicted.shape == actual.shape and predicted.dtype == actual.dtype == np.bool_, 'boolean labels same shape')
    tp = int((predicted & actual).sum()); fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum()); tn = int((~predicted & ~actual).sum())
    return dict(total=predicted.size, positives=tp+fn, negatives=tn+fp, tp=tp, fp=fp, fn=fn, tn=tn,
                false_positive_rate=None if tn+fp == 0 else fp/(tn+fp),
                false_negative_rate=None if tp+fn == 0 else fn/(tp+fn))


def evaluate(scores, truth_scores, buffer, scale):
    flags = decisions(scores, buffer, scale)
    return {key:confusion(flags[key], truth_scores[key] >= 0) for key in flags}


def probe_contexts(seed, groups, prefix):
    """Synthetic diagnostic contexts, NOT additional VLA successes or training examples."""
    rng = np.random.default_rng(seed)
    initial = rng.uniform([-.08, -.14], [.08, .02], (groups, 2)).astype(np.float32)
    goal = (initial + rng.uniform(-.08, .08, (groups, 2))).astype(np.float32)
    # Goal is fixed BEFORE actions/truth. Purpose-built action probes provide positive and negative classes.
    direction = (goal-initial)[:, None, None]/3
    factor = rng.uniform(.4, 1.6, (groups, 5, 1, 1))
    noise = rng.normal(0, .008, (groups, 5, 3, 2))
    action = np.clip(direction*factor+noise, -.06, .06).astype(np.float32)
    gain = rng.uniform(.8, 1., groups)
    return dict(initial=initial, goal=goal, action=action, gain=gain,
                group_id=np.array([f'{prefix}_{i:03d}' for i in range(groups)]))


def replay_probe(context, model, shifted=False):
    initial = np.broadcast_to(context['initial'][:, None], (*context['action'].shape[:2], 2))
    prediction = wm.imagine(initial, context['action'], model)
    gain = np.full(len(initial), .65) if shifted else context['gain']
    truth = np.stack([wm.plant_rollout(initial[i], context['action'][i], float(g)) for i,g in enumerate(gain)])
    return prediction, truth


def observation_goals(data):
    # Explicit perfect goal decoder for THIS synthetic camera, separate from the learned policy.
    goals = []
    for rgb, text in zip(data['image'], data['text']):
        require(text in ('move to red', 'move to blue'), 'literal goal decoder only')
        channel = 0 if text == 'move to red' else 2
        mask = (rgb[..., channel] > 200) & (rgb[..., 2-channel] < 50)
        v, u = np.nonzero(mask)
        require(len(u) == 9, 'expected synthetic 3x3 object marker')
        goals.append([(u.mean()-16)*.02, (16-v.mean())*.02])
    return np.array(goals, np.float64)


def load_source(directory):
    require(all((directory/name).is_file() for name in FILES), 'need frozen S4.2 source files')
    contract = json.loads((directory/'contract.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    require(contract.get('schema_version') == 'candidate_future_verification/1.0'
            and contract['proposal_contract'] == CONTRACT, 'S4.2 schema/action semantics')
    require(report.get('schema_version') == 'candidate_future_verification_report/1.0', 'S4.2 report schema')
    conditions = contract['conditions']
    require(conditions['nominal_gain'] == 1. and np.isfinite(conditions['shifted_gain'])
            and 0 < conditions['shifted_gain'] < 1, 'declared plant gains')
    with np.load(directory/'model.npz', allow_pickle=False) as saved:
        model = {key:saved[key] for key in saved.files}
    wm.validate_model(model, contract['world_model_contract'])
    require(sha256(directory/'model.npz') == report['model']['sha256'], 'model provenance hash')
    proposal_dir = Path(report['source']['directory'])
    require(report['source']['sha256'] == {name:sha256(proposal_dir/name) for name in wm.SOURCE_FILES}, 'proposal source hashes')
    data, proposal_report = wm.load_proposals(proposal_dir)
    with np.load(directory/'verification.npz', allow_pickle=False) as saved:
        arrays = {key:saved[key] for key in saved.files}
    np.testing.assert_array_equal(arrays['timestamp_s'], np.arange(4, dtype=np.float64)*.1)
    require(arrays['timestamp_s'].dtype == np.float64, 'float64 simulation clock')
    action = arrays['action_m']; np.testing.assert_array_equal(action, data['action'])
    np.testing.assert_array_equal(arrays['initial_state_m'], data['state'])
    shape = (*action.shape[:2], 4, 2)
    require(list(action.shape) == contract['action_shape'] and list(shape) == contract['trajectory_shape'], 'declared shapes')
    for key, dtype in [('predicted_ar_m', np.float64), ('truth_nominal_m', np.float32), ('truth_shifted_m', np.float32)]:
        require(arrays[key].shape == shape and arrays[key].dtype == dtype and np.isfinite(arrays[key]).all(), 'future '+key)
    initial = np.broadcast_to(data['state'][:,None], (*action.shape[:2],2))
    np.testing.assert_allclose(wm.imagine(initial, action, model), arrays['predicted_ar_m'], rtol=0, atol=1e-14)
    for key,gain in [('truth_nominal_m',1.), ('truth_shifted_m',contract['conditions']['shifted_gain'])]:
        np.testing.assert_array_equal(arrays[key], wm.plant_rollout(initial, action, gain))
    for value in model.values(): value.flags.writeable = False
    return model, arrays, data, proposal_report, proposal_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', type=Path, default=Path('tmp/s4_2_verified'))
    parser.add_argument('--margin-scale', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=Path('tmp/s4_3'))
    args = parser.parse_args()
    require(not args.output.exists(), 'fresh output directory required')
    require(np.isfinite(args.margin_scale) and args.margin_scale >= 0, 'finite nonnegative margin scale')
    model, source, proposal, proposal_report, proposal_dir = load_source(args.verification)
    hashes = {name:sha256(args.verification/name) for name in FILES}
    proposal_hashes = {name:sha256(proposal_dir/name) for name in wm.SOURCE_FILES}
    calibration = probe_contexts(431, 64, 'calibration')
    heldout = probe_contexts(432, 32, 'heldout')
    require(set(calibration['group_id']).isdisjoint(heldout['group_id']), 'group overlap')
    cal_prediction, cal_truth = replay_probe(calibration, model)
    cal_scores, cal_actual = margins(cal_prediction, calibration['goal']), margins(cal_truth, calibration['goal'])
    buffer = calibrate(cal_scores, cal_actual)  # Frozen BEFORE heldout replay/labels.
    arrays = {}; summary = {}
    for key,value in calibration.items(): arrays['calibration_'+key] = value
    arrays.update(calibration_predicted_m=cal_prediction, calibration_truth_m=cal_truth)
    for key,value in heldout.items(): arrays['heldout_'+key] = value
    goals = observation_goals(proposal)
    cases = [('calibration',cal_prediction,cal_truth,calibration['goal'])]
    for name,shift in [('heldout_id',False), ('heldout_gain065',True)]:
        pred, truth = replay_probe(heldout, model, shift)
        cases.append((name,pred,truth,heldout['goal']))
    cases.extend([('vla_nominal',source['predicted_ar_m'],source['truth_nominal_m'],goals),
                  ('vla_gain_shift',source['predicted_ar_m'],source['truth_shifted_m'],goals)])
    for name,pred,truth,goal in cases:
        score, actual = margins(pred,goal), margins(truth,goal)
        summary[name] = dict(raw=evaluate(score,actual,buffer,0),
                            calibrated=evaluate(score,actual,buffer,args.margin_scale),
                            groups=len(pred), candidates_per_group=pred.shape[1],
                            constant_negative={key:confusion(np.zeros_like(value,dtype=bool),value>=0) for key,value in actual.items()},
                            constant_positive={key:confusion(np.ones_like(value,dtype=bool),value>=0) for key,value in actual.items()},
                            buffer_group_coverage={key:dict(covered=int((np.abs(score[key]-actual[key]).max(axis=1)<=buffer[key]).sum()),
                                groups=len(pred)) for key in score})
        arrays[name+'_prediction_m'] = pred; arrays[name+'_truth_m'] = truth; arrays[name+'_goal_m'] = goal
        flags = decisions(score,buffer,args.margin_scale)
        for key in score:
            arrays[name+'_'+key+'_score_m'] = score[key]
            arrays[name+'_'+key+'_actual_margin_m'] = actual[key]
            arrays[name+'_'+key+'_label'] = actual[key] >= 0
            arrays[name+'_'+key+'_flag'] = flags[key]
    times = []
    for _ in range(5): decisions(margins(source['predicted_ar_m'],goals),buffer,args.margin_scale)
    for _ in range(30):
        start = perf_counter_ns()
        decisions(margins(source['predicted_ar_m'],goals),buffer,args.margin_scale)
        times.append((perf_counter_ns()-start)/1e6)
    require(hashes == {name:sha256(args.verification/name) for name in FILES}, 'source mutation')
    require(proposal_hashes == {name:sha256(proposal_dir/name) for name in wm.SOURCE_FILES}, 'proposal mutation')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'scores.npz', **arrays)
    with np.load(args.output/'scores.npz', allow_pickle=False) as saved:
        for key,value in arrays.items(): np.testing.assert_array_equal(saved[key],value)
    contract = dict(schema_version='risk_success_margins/1.0', world_model_contract=wm.MODEL_CONTRACT,
        coordinate_frame='world_xy', unit='m', dt_s=.1, horizon=3,
        risk=dict(center_m=CENTER.tolist(), radius_m=RADIUS,
            definition='any sampled FUTURE state h1..3 in closed monitoring disk; no segment collision or plant constraint'),
        success=dict(tolerance_m=TOLERANCE, definition='final h3 L-infinity goal distance <= .01m, independent of risk'),
        scores='risk=radius-min_future_l2_to_center; success=tolerance-final_linf_to_goal; positive means event',
        decisions='risk_score>=-scale*risk_buffer; success_score>=scale*success_buffer; flags only, no execution gate',
        calibration='64 independent synthetic groups K5; per-group max absolute score error, q90 higher; gains uniform [.8,1)',
        heldout='32 new synthetic groups K5, independent seed; ID gains [.8,1), additional .65 mechanism shift',
        probe_seeds=dict(calibration=431,heldout=432),
        interpretation='empirical error buffers, not calibrated probabilities or a formal finite-sample coverage guarantee',
        goal_source='synthetic probe goals fixed before commands; VLA goals perfectly decoded from current literal text/RGB only')
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    report = dict(schema_version='risk_success_scoring_report/1.0',
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,numpy_path=np.__file__),
        verification=str(args.verification.resolve()), source_sha256=hashes, proposal_sha256=proposal_hashes,
        source_code_sha256={name:sha256(Path(__file__).parent/name) for name in
            ('s4_3_risk_success_scoring.py','s4_2_world_model_verification.py','s4_1_vla_proposal.py')},
        margin_scale=args.margin_scale, calibration_buffers_m=buffer, summary=summary,
        proposal_attempts=18, supported_inputs=16, input_rejections=proposal_report['rejected'],
        scoring_latency_ms=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),
            samples=30,warmup=5,batch_candidates=source['action_m'].shape[0]*source['action_m'].shape[1],
            scope='geometric score+flag only; excludes prediction, goal decode, plant, IO, transport'),
        trained_during_scoring=False,closed_loop_success=None,
        limitations='Synthetic monitoring geometry, discrete samples only; perfect goal grounding bypasses learned perception. Probe actions are hand-designed, not VLA outputs. Correlated K candidates, only 32 heldout groups. No calibrated event probability, task-level safe success, rejection, fallback, selection or new closed-loop evidence.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['set,method,event,total,positives,negatives,tp,fp,fn,tn']
    for name,group in summary.items():
        for method in ('raw','calibrated','constant_negative','constant_positive'):
            for event,c in group[method].items():
                lines.append(','.join(map(str,[name,method,event,*[c[k] for k in ('total','positives','negatives','tp','fp','fn','tn')]])))
    (args.output/'confusion.csv').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(calibration_buffers_m=buffer,margin_scale=args.margin_scale,summary=summary),indent=2))


if __name__ == '__main__': main()
