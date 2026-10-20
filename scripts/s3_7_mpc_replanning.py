"""Frozen S3.6 dynamics: single-shot vs repeat search vs observed-state replanning."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from s3_3_model_uncertainty import fit_statistics
from s3_5_outcome_heads import step
from s3_6_model_based_planning import HORIZON, candidate_bank, costs, outcome, predict_state, select


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(directory):
    names = ('contract.json', 'model.npz', 'planning.npz', 'report.json')
    hashes = {name: digest(directory/name) for name in names}
    contract = json.loads((directory/'contract.json').read_text())
    if contract.get('schema') != 's3_6_wall_planning/1.0' or contract.get('plant_profile') != 's3_5_wall_outcomes/1.0':
        raise ValueError('S3.6 wall-planning contract required')
    evidence = json.loads((directory/'report.json').read_text())
    if evidence.get('model_sha256') != hashes['model.npz']:
        raise ValueError('checkpoint does not match source report hash')
    with np.load(directory/'model.npz', allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    shapes = {'w1': (4, 32), 'b1': (32,), 'w2': (32, 2), 'b2': (2,),
              'x_mean': (4,), 'x_std': (4,), 'y_mean': (2,), 'y_std': (2,)}
    if set(model) != set(shapes):
        raise ValueError('checkpoint fields mismatch')
    for key, shape in shapes.items():
        if model[key].shape != shape or model[key].dtype.kind != 'f' or not np.isfinite(model[key]).all():
            raise ValueError(f'invalid checkpoint {key}')
    if not np.all(model['x_std'] > 0) or not np.all(model['y_std'] > 0):
        raise ValueError('nonpositive normalization scale')
    with np.load(directory/'planning.npz', allow_pickle=False) as archive:
        states, actions = archive['source_state_m'], archive['source_action_m']
        train_ids, ids = archive['train_episode_ids'], archive['validation_episode_ids']
        starts = archive['initial_state_m']
    if states.shape != (60, 21, 2) or actions.shape != (60, 20, 2) or states.dtype != np.float32 or actions.dtype != np.float32:
        raise ValueError('source data shape/dtype')
    if train_ids.shape != (40,) or ids.shape != (20,) or not np.array_equal(np.sort(np.r_[train_ids, ids]), np.arange(60)):
        raise ValueError('invalid episode partition')
    np.testing.assert_array_equal(starts, states[ids, 0])
    np.testing.assert_array_equal(step(states[:, :-1].reshape(-1, 2), actions.reshape(-1, 2))[0], states[:, 1:].reshape(-1, 2))
    x = np.concatenate([states[train_ids, :-1].reshape(-1, 2), actions[train_ids].reshape(-1, 2)], axis=1)
    for key, value in fit_statistics(x, states[train_ids, 1:].reshape(-1, 2)).items():
        np.testing.assert_array_equal(value, model[key])
    if any(digest(directory/name) != value for name, value in hashes.items()):
        raise ValueError('source changed during loading')
    return model, starts, ids, hashes, evidence['offline_validation']


def planning_bank(episode_id, t, count, warm_start=None):
    bank = candidate_bank(int(episode_id)+1000*t, count, HORIZON-t)
    if warm_start is not None:
        if warm_start.shape != bank[-1].shape:
            raise ValueError('warm-start time alignment')
        bank[-1] = warm_start
    return bank


def run_episode(initial, episode_id, count, execute, feedback, predictor, plant):
    if not 1 <= execute <= HORIZON:
        raise ValueError('execute must be within horizon')
    states, actions, plans = [initial.copy()], [], []
    belief, warm, t = initial.astype(np.float64), None, 0
    while t < HORIZON:
        # Actual state influences planning only at observation boundaries when feedback=True.
        planning_state = states[-1].copy() if feedback else belief.copy()
        start = perf_counter_ns()
        bank = planning_bank(episode_id, t, count, warm)
        generation_ms = (perf_counter_ns()-start)/1e6
        start = perf_counter_ns()
        index, predictions, scores = select(planning_state, bank, predictor)
        selection_ms = (perf_counter_ns()-start)/1e6
        length = min(execute, HORIZON-t)
        chosen = bank[index].copy()
        believed_endpoint = predictions[index, length].copy()
        plant_times = []
        for offset in range(length):
            start = perf_counter_ns()
            observed = plant(states[-1][None], chosen[offset:offset+1], t+offset)[0]
            plant_times.append((perf_counter_ns()-start)/1e6)
            if not np.isfinite(observed).all():
                raise ValueError('nonfinite plant state')
            states.append(observed.copy())
            actions.append(chosen[offset].copy())
        plans.append({'t': t, 'remaining_horizon': HORIZON-t, 'executed_prefix': length,
                      'planning_state_m': planning_state.tolist(), 'selected_index': index,
                      'selected_future_action_m': chosen.tolist(),
                      'selected_future_prediction_m': predictions[index].tolist(),
                      'candidate_costs': scores.tolist(),
                      'prefix_endpoint_prediction_error_m': float(np.linalg.norm(states[-1]-believed_endpoint)),
                      'generation_ms': generation_ms, 'selection_ms': selection_ms,
                      'plant_step_ms': plant_times, 'compute_budget_ms': length*.1*1000})
        belief = believed_endpoint
        warm = chosen[length:].copy()
        t += length
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32), plans


def plant_for(condition):
    def plant(state, action, t):
        gain = .65 if condition == 'weak_gain' else 1.
        next_state = step(state.astype(np.float32), (action*gain).astype(np.float32))[0]
        if condition == 'kick' and t == 4:
            next_state[:, 1] += np.float32(.08)
        return next_state
    return plant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('tmp/s3_6_final'))
    parser.add_argument('--execute', type=int, default=1)
    parser.add_argument('--candidates', type=int, default=128)
    parser.add_argument('--output', type=Path, default=Path('tmp/s3_7'))
    args = parser.parse_args()
    if not 1 <= args.execute <= 10 or not 2 <= args.candidates <= 512:
        parser.error('execute must be 1..10; candidates must be 2..512')
    model, starts, ids, hashes, offline = load_source(args.checkpoint)
    weights = [model[k] for k in ('w1', 'b1', 'w2', 'b2')]
    predictor = lambda s, a: predict_state(s, a, weights, model)
    args.output.mkdir(parents=True, exist_ok=False)
    methods = [('open_loop', 10, False), ('replan_no_feedback', args.execute, False), ('mpc_feedback', args.execute, True)]
    records, arrays = [], {'initial_state_m': starts, 'source_episode_ids': ids,
                           'timestamp_s': np.arange(11, dtype=np.float64)*.1}
    for condition in ('nominal', 'weak_gain', 'kick'):
        for method, execute, feedback in methods:
            raw_states, raw_actions = [], []
            for attempt, (episode_id, initial) in enumerate(zip(ids, starts)):
                actual, commands, plans = run_episode(initial, episode_id, args.candidates, execute, feedback,
                                                      predictor, plant_for(condition))
                result = outcome(actual)
                result.update(condition=condition, method=method, attempt=attempt, source_episode_id=int(episode_id),
                              calls=len(plans), actual_cost=float(costs(actual[None], commands[None])[0]),
                              total_generation_ms=sum(p['generation_ms'] for p in plans),
                              total_selection_ms=sum(p['selection_ms'] for p in plans), plans=plans)
                records.append(result)
                raw_states.append(actual)
                raw_actions.append(commands)
            arrays[condition+'_'+method+'_state_m'] = np.asarray(raw_states)
            arrays[condition+'_'+method+'_action_m'] = np.asarray(raw_actions)
    summary = {}
    for condition in ('nominal', 'weak_gain', 'kick'):
        summary[condition] = {}
        for method, _, _ in methods:
            subset = [r for r in records if r['condition'] == condition and r['method'] == method]
            plans = [p for r in subset for p in r['plans']]
            call_times = [p['generation_ms']+p['selection_ms'] for p in plans]
            total_times = [r['total_generation_ms']+r['total_selection_ms'] for r in subset]
            summary[condition][method] = {
                'attempts': len(subset), 'successes': sum(r['success'] for r in subset),
                'failure_stages': {stage: sum(r['failure_stage'] == stage for r in subset)
                                   for stage in ('boundary_violation', 'goal_not_reached')},
                'planning_calls_per_episode': subset[0]['calls'], 'total_planning_calls': len(plans),
                'mean_actual_cost': float(np.mean([r['actual_cost'] for r in subset])),
                'mean_final_distance_m': float(np.mean([r['final_distance_m'] for r in subset])),
                'per_call_generation_plus_selection_ms': {'p50': float(np.percentile(call_times, 50)),
                                                          'p95': float(np.percentile(call_times, 95))},
                'per_episode_generation_plus_selection_ms': {'p50': float(np.percentile(total_times, 50)),
                                                             'p95': float(np.percentile(total_times, 95))},
                'mean_prefix_endpoint_prediction_error_m': float(np.mean([p['prefix_endpoint_prediction_error_m'] for p in plans])),
                'compute_budget_overruns': sum(c > p['compute_budget_ms'] for c, p in zip(call_times, plans))}
    if any(digest(args.checkpoint/name) != value for name, value in hashes.items()):
        raise ValueError('frozen source modified')
    np.savez_compressed(args.output/'executions.npz', **arrays)
    report = {'schema': 's3_7_mpc_replanning/1.0', 'source_directory': str(args.checkpoint.resolve()),
              'source_sha256': hashes, 'offline_validation': offline,
              'execute': args.execute, 'candidates': args.candidates, 'task_steps': 10, 'dt_s': .1,
              'summary': summary, 'records': records,
              'timing': 'Actual CPU wall time, warmup0; generation+selection measured separately and summed per call/episode. Plant timing separate in plans. No sensor/I/O/robot delay.',
              'limitations': 'Finite-task shrinking-horizon MPC-style shooting on noiseless synthetic observations; no model adaptation or risk guarantee.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    contract = {'schema': 's3_7_wall_feedback/1.0', 'source_profile': 's3_6_wall_planning/1.0',
                'model': 'frozen S3.6 226-param wall dynamics and train-only statistics; no training or gain identification',
                'state_action': 'float32 world XY position/commanded displacement[m]; dt=.1s; raw arrays [20,11,2]/[20,10,2]',
                'observation': 'perfect current plant XY available at replan boundaries only to mpc_feedback',
                'horizon': 'fixed10-step task, shrink planning horizon=10-t; execute min(E,10-t), final prefix may be shorter',
                'candidates': 'S3.6 bank(episode_id+1000*t,N,10-t); t0 identical; later last candidate replaced by unexecuted previous selected suffix',
                'cost': 'same S3.6 known cost applied to remaining segment, with path/effort means over remaining length; final actual cost over full10',
                'control': 'no early success stop, model update or oracle future; all20 starts retained per condition/method',
                'conditions': 'nominal; gain .65 hidden inside plant; kick y+=.08m after transition4->5; planner not told perturbations',
                'feedback_ablation': 'replan_no_feedback reuses only predicted prefix endpoint as belief, never refreshed actual state after t0',
                'success': 'S3.6 final goal occupancy and no violation in executed future states; boundary vs goal failure stages',
                'timing_boundary': 'instantaneous simulated plant; measured compute budgets not actual asynchronous deadline experiment',
                'reconstruction': 'report stores all selected plans/scores; full banks reproducible from t/episode/N and previous unexecuted suffix'}
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    for condition, values in summary.items():
        for method, value in values.items():
            print(condition, method, f"{value['successes']}/20", value['failure_stages'],
                  'calls/episode',value['planning_calls_per_episode'],
                  'total compute p50 ms',round(value['per_episode_generation_plus_selection_ms']['p50'], 3))


if __name__ == '__main__':
    main()
