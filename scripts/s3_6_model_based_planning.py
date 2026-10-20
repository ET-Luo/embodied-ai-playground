"""Single-shot candidate shooting with learned wall dynamics; no MPC feedback."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from s3_1_state_dynamics import forward, metrics
from s3_3_model_uncertainty import fit_statistics, train
from s3_5_outcome_heads import GOAL, GOAL_RADIUS, WALL_X, dataset, step

HORIZON = 10
ACTION_LIMIT = .06


def candidate_bank(episode_id, count, horizon=HORIZON):
    bank = np.zeros((count, horizon, 2), dtype=np.float32)
    for index in range(1, count):
        # Candidate-specific RNG gives nested sets when only count changes.
        rng = np.random.default_rng(360000+int(episode_id)*1000+index)
        duration = int(rng.integers(1, horizon+1))
        switch = int(rng.integers(1, duration+1))
        first = rng.uniform(-ACTION_LIMIT, ACTION_LIMIT, 2)
        second = np.clip(first+rng.normal(0, .02, 2), -ACTION_LIMIT, ACTION_LIMIT)
        bank[index, :switch] = first
        bank[index, switch:duration] = second
    return bank


def predict_state(state, action, weights, stats):
    x = np.concatenate([state, action], axis=-1).astype(np.float32)
    delta = forward((x-stats['x_mean'])/stats['x_std'], weights)[0]
    return state+delta*stats['y_std']+stats['y_mean']


def ideal_state(state, action):
    result = state+action
    result[:, 0] = np.minimum(result[:, 0], WALL_X)
    return result


def rollout(initial_state, candidates, predictor):
    states = np.empty((len(candidates), candidates.shape[1]+1, 2), dtype=np.float64)
    states[:, 0] = initial_state
    for t in range(candidates.shape[1]):
        states[:, t+1] = predictor(states[:, t], candidates[:, t])
    if not np.isfinite(states).all():
        raise ValueError('nonfinite rollout; refusing candidate selection')
    return states


def violations(states):
    return (states[..., 0] < -.20) | (np.abs(states[..., 1]) > .20)


def costs(states, candidates):
    """Dimensionless declared task objective; no learned outcome-head score."""
    future = states[:, 1:]
    distance2 = np.sum(((future-GOAL)/.1)**2, axis=-1)
    return (distance2[:, -1]+.05*distance2.mean(1)
            +5*violations(future).sum(1)+.01*np.mean((candidates/ACTION_LIMIT)**2, axis=(1, 2)))


def select(initial_state, candidates, predictor):
    prediction = rollout(initial_state, candidates, predictor)
    score = costs(prediction, candidates)
    if not np.isfinite(score).all():
        raise ValueError('nonfinite costs; refusing selection')
    return int(np.argmin(score)), prediction, score


def outcome(states):
    bad = violations(states[1:])
    goal = np.linalg.norm(states[-1]-GOAL) <= GOAL_RADIUS
    stage = 'boundary_violation' if bad.any() else ('success' if goal else 'goal_not_reached')
    return {'success': bool(goal and not bad.any()), 'failure_stage': stage,
            'first_violation_transition': int(np.flatnonzero(bad)[0]) if bad.any() else None,
            'final_distance_m': float(np.linalg.norm(states[-1]-GOAL))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=Path('tmp/s3_6'))
    args = parser.parse_args()
    if not 2 <= args.candidates <= 512 or not 1 <= args.epochs <= 10000:
        parser.error('candidates must be 2..512; epochs must be 1..10000')
    args.output.mkdir(parents=True, exist_ok=False)
    states, actions, _, _ = dataset()
    ids = np.random.default_rng(351).permutation(60)
    train_ids, val_ids = ids[:40], ids[40:]
    tx = np.concatenate([states[train_ids, :-1].reshape(-1, 2), actions[train_ids].reshape(-1, 2)], axis=1)
    target = states[train_ids, 1:].reshape(-1, 2)
    stats = fit_statistics(tx, target)
    weights, losses = train((tx-stats['x_mean'])/stats['x_std'],
                            (target-tx[:, :2]-stats['y_mean'])/stats['y_std'], args.epochs, 360)
    np.savez_compressed(args.output/'model.npz', **stats,
                        **{k: w for k, w in zip(('w1', 'b1', 'w2', 'b2'), weights)})
    model_hash = hashlib.sha256((args.output/'model.npz').read_bytes()).hexdigest()
    vx = np.concatenate([states[val_ids, :-1].reshape(-1, 2), actions[val_ids].reshape(-1, 2)], axis=1)
    learned = lambda s, a: predict_state(s, a, weights, stats)
    offline = metrics(learned(vx[:, :2], vx[:, 2:]), states[val_ids, 1:].reshape(-1, 2))
    starts = states[val_ids, 0]
    arrays = {'initial_state_m': starts, 'validation_episode_ids': val_ids,
              'train_episode_ids': train_ids, 'source_state_m': states, 'source_action_m': actions,
              'timestamp_s': np.arange(HORIZON+1, dtype=np.float64)*.1}
    collections = {key: [] for key in ('candidate_action_m', 'learned_prediction_m', 'learned_score',
                                       'ideal_prediction_m', 'ideal_score', 'nominal_true_state_m', 'weak_gain_true_state_m')}
    records, planning_times, generation_times = [], [], []
    for row, (episode_id, initial) in enumerate(zip(val_ids, starts)):
        start = perf_counter_ns()
        candidates = candidate_bank(episode_id, args.candidates)
        generation_times.append((perf_counter_ns()-start)/1e6)
        start = perf_counter_ns()
        selected, prediction, score = select(initial, candidates, learned)
        planning_times.append((perf_counter_ns()-start)/1e6)
        ideal_index, ideal_prediction, ideal_score = select(initial, candidates, ideal_state)
        for key, value in [('candidate_action_m', candidates), ('learned_prediction_m', prediction),
                           ('learned_score', score), ('ideal_prediction_m', ideal_prediction), ('ideal_score', ideal_score)]:
            collections[key].append(value)
        # All learned/ideal choices are fixed BEFORE any environment future is evaluated.
        for condition, gain in [('nominal', 1.), ('weak_gain', .65)]:
            plant = lambda s, a: step(s.astype(np.float32), (a*gain).astype(np.float32))[0]
            actual = rollout(initial, candidates, plant)
            actual_score = costs(actual, candidates)
            oracle_index = int(np.argmin(actual_score))
            candidate_success = ((np.linalg.norm(actual[:, -1]-GOAL, axis=1) <= GOAL_RADIUS)
                                 & ~np.any(violations(actual[:, 1:]), axis=1))
            collections[condition+'_true_state_m'].append(actual)
            for method, index in [('learned', selected), ('ideal', ideal_index),
                                  ('zero', 0), ('random', 1), ('oracle', oracle_index)]:
                value = outcome(actual[index])
                predicted = prediction[index] if method == 'learned' else ideal_prediction[index] if method == 'ideal' else None
                record = {'condition': condition, 'method': method, 'attempt': row,
                          'source_episode_id': int(episode_id), 'candidate_index': index,
                          'candidate_success_count': int(candidate_success.sum()),
                          'actual_cost': float(actual_score[index]),
                          'candidate_set_regret': float(actual_score[index]-actual_score[oracle_index]), **value}
                if predicted is not None:
                    record.update(predicted_success=outcome(predicted)['success'],
                                  predicted_cost=float(score[index] if method == 'learned' else ideal_score[index]),
                                  endpoint_prediction_error_m=float(np.linalg.norm(predicted[-1]-actual[index, -1])))
                records.append(record)
    arrays.update({key: np.asarray(value) for key, value in collections.items()})
    np.savez_compressed(args.output/'planning.npz', **arrays)
    summary = {}
    for condition in ('nominal', 'weak_gain'):
        summary[condition] = {}
        for method in ('learned', 'ideal', 'zero', 'random', 'oracle'):
            subset = [r for r in records if r['condition'] == condition and r['method'] == method]
            result = {'attempts': len(subset), 'successes': sum(r['success'] for r in subset),
                      'failure_stages': {stage: sum(r['failure_stage'] == stage for r in subset)
                                         for stage in ('boundary_violation', 'goal_not_reached')},
                      'no_success_candidate_attempts': sum(r['candidate_success_count'] == 0 for r in subset),
                      'misses_despite_success_candidate': sum(not r['success'] and r['candidate_success_count'] > 0 for r in subset),
                      'mean_actual_cost': float(np.mean([r['actual_cost'] for r in subset])),
                      'mean_candidate_set_regret': float(np.mean([r['candidate_set_regret'] for r in subset]))}
            if method in ('learned', 'ideal'):
                result.update(predicted_successes=sum(r['predicted_success'] for r in subset),
                              false_success_predictions=sum(r['predicted_success'] and not r['success'] for r in subset),
                              mean_endpoint_prediction_error_m=float(np.mean([r['endpoint_prediction_error_m'] for r in subset])))
            summary[condition][method] = result
    report = {'schema': 's3_6_model_based_planning/1.0', 'candidates': args.candidates, 'horizon': HORIZON,
              'dt_s': .1, 'epochs': args.epochs, 'model_parameters': 226, 'model_sha256': model_hash,
              'normalized_train_losses': losses, 'offline_validation': offline, 'summary': summary, 'records': records,
              'latency_ms': {'learned_rollout_score_select': {'p50': float(np.percentile(planning_times, 50)),
                                                              'p95': float(np.percentile(planning_times, 95))},
                             'candidate_generation': {'p50': float(np.percentile(generation_times, 50)),
                                                      'p95': float(np.percentile(generation_times, 95))},
                             'samples': 20, 'warmup': 0},
              'limitations': 'Single-shot open-loop synthetic wall plant, no MPC or real robot; oracle uses privileged environment futures.'}
    assert hashlib.sha256((args.output/'model.npz').read_bytes()).hexdigest() == model_hash
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    contract = {'schema': 's3_6_wall_planning/1.0', 'plant_profile': 's3_5_wall_outcomes/1.0',
                'training': 'same seed350 dataset /351 episode split, train40/800 only, model seed360; 226-param residual MLP',
                'normalization': 'x and next_state-state mean/std fit train800 only; never refit for candidates',
                'candidate_actions': f'float32 [20,{args.candidates},10,2], commanded world XY displacement[m], per-axis <=.06m',
                'prediction': f'float64 [20,{args.candidates},11,2], world XY[m], first state true start then recursive predictions',
                'truth': 'float64 storage of float32 plant transitions; each candidate/reset independently initialized',
                'time': '10 transitions at dt=.1s, 11 state slots; full chunk executes once, no state feedback/replanning',
                'candidate_seed': '360000+source_episode_id*1000+candidate_index; index0 zero, index1 random comparator',
                'nominal': 'S3.5 wall step, fixed goal [.09,0]m and radius.045m',
                'weak_gain': 'unmodeled actuation gain .65 applied inside plant before same wall step; command semantics unchanged',
                'cost': 'terminal squared goal-distance/.1m + .05 mean path squared goal-distance/.1m +5 sum future boundary violations +.01 mean(action/.06m)^2',
                'success': 'final state in goal AND no boundary violation in 10 executed future states; all attempts retained',
                'oracle': 'same candidate set, actual environment rollout costs; privileged diagnostic only, not global optimum',
                'no_future_leakage': 'learned/ideal choose indices before any true candidate rollout; true futures only evaluator/oracle',
                'heads': 'known task cost; S3.5 learned reward/risk heads and S3.4 latent not integrated',
                'modality': 'position/action only, no image/language/force/velocity; raw states/actions preserved'}
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    print('offline one-step RMSE [mm]', offline['rmse_m']*1000)
    for condition, values in summary.items():
        for method, value in values.items():
            print(condition, method, f"{value['successes']}/{value['attempts']}", 'failures',value['failure_stages'],
                  'mean regret',round(value['mean_candidate_set_regret'], 5))
    print('latency',report['latency_ms'])


if __name__ == '__main__':
    main()
