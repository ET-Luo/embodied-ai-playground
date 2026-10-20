"""Frozen S3.1 model: truth-conditioned one-step vs autoregressive rollout."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from s3_1_state_dynamics import forward, metrics, transition


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(directory):
    paths = [directory/name for name in ('contract.json', 'model.npz', 'episodes.npz')]
    hashes = {p.name: digest(p) for p in paths}
    contract = json.loads(paths[0].read_text())
    require(contract['schema'] == 's3_1_xy_command_dynamics/1.0', 'unsupported source contract')
    # This evaluator supports this exact synthetic producer, not arbitrary XY-shaped files.
    expected = {'action': 'float32 [episode,20,2], commanded world XY displacement [m]; NOT realized displacement',
                'dynamics': 's_next=s+(0.70+0.15*tanh(s/0.25))*a+0.08*reverse_xy(a)',
                'time': 'float64 [21], relative seconds, dt=0.1; action[t] acts from state[t] to state[t+1]'}
    require(all(contract.get(k) == v for k, v in expected.items()), 'source action/dynamics/time mismatch')
    with np.load(paths[1], allow_pickle=False) as archive:
        model = {k: archive[k] for k in archive.files}
    shapes = {'w1': (4, 32), 'b1': (32,), 'w2': (32, 2), 'b2': (2,),
              'x_mean': (4,), 'x_std': (4,), 'y_mean': (2,), 'y_std': (2,)}
    require(set(model) == set(shapes), 'model fields')
    for key, shape in shapes.items():
        require(model[key].shape == shape and model[key].dtype.kind == 'f'
                and np.isfinite(model[key]).all(), f'invalid model {key}')
    require(np.all(model['x_std'] > 0) and np.all(model['y_std'] > 0), 'nonpositive scale')
    with np.load(paths[2], allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    s, a = data['state_m'], data['action_m']
    require(s.shape == (40, 21, 2) and a.shape == (40, 20, 2), 'source shape')
    require(s.dtype == a.dtype == np.float32 and np.isfinite(s).all()
            and np.isfinite(a).all(), 'source dtype/finite')
    require(data['timestamp_s'].dtype == np.float64, 'clock dtype')
    np.testing.assert_allclose(data['timestamp_s'], np.arange(21)*.1, rtol=0, atol=1e-12)
    train, val = data['train_episode_ids'], data['validation_episode_ids']
    require(train.shape == (30,) and val.shape == (10,), 'split sizes')
    require(train.dtype.kind in 'iu' and val.dtype.kind in 'iu', 'split integer ids')
    require(np.array_equal(np.sort(np.concatenate([train, val])), np.arange(40)), 'split partition')
    np.testing.assert_array_equal(transition(s[:, :-1].reshape(-1, 2), a.reshape(-1, 2)),
                                  s[:, 1:].reshape(-1, 2))
    x = np.concatenate([s[train, :-1].reshape(-1, 2), a[train].reshape(-1, 2)], axis=1)
    y = (s[train, 1:] - s[train, :-1]).reshape(-1, 2)
    for key, expected_value in [('x_mean', x.mean(0)), ('x_std', x.std(0)),
                                ('y_mean', y.mean(0)), ('y_std', y.std(0))]:
        np.testing.assert_array_equal(model[key], expected_value)
    require(all(digest(p) == hashes[p.name] for p in paths), 'source changed during load')
    return model, data, hashes


def predict(state, action, model):
    x = np.concatenate([state, action], axis=-1).astype(np.float32)
    weights = [model[k] for k in ('w1', 'b1', 'w2', 'b2')]
    delta = forward((x-model['x_mean'])/model['x_std'], weights)[0]
    return state + delta*model['y_std'] + model['y_mean']


def extended_episodes(data, steps):
    ids = data['validation_episode_ids']
    actions = np.empty((len(ids), steps, 2), dtype=np.float32)
    actions[:, :20] = data['action_m'][ids]
    for row, episode in enumerate(ids):
        # Each episode has its own stream; extending horizon preserves every prefix.
        rng = np.random.default_rng(3200 + int(episode))
        actions[row, 20:] = rng.uniform(-.04, .04, (steps-20, 2)).astype(np.float32)
    states = np.empty((len(ids), steps+1, 2), dtype=np.float32)
    states[:, :21] = data['state_m'][ids]
    for t in range(20, steps):
        states[:, t+1] = transition(states[:, t], actions[:, t])
    return states, actions


def evaluate(predictor, truth, actions):
    """TF sees truth[t]; AR sees only initial truth and previous prediction."""
    count, steps, _ = actions.shape
    tf = np.empty((count, steps, 2), dtype=np.float64)
    ar = np.empty((count, steps+1, 2), dtype=np.float64)
    ar[:, 0] = truth[:, 0]
    for t in range(steps):
        tf[:, t] = predictor(truth[:, t].astype(np.float64), actions[:, t].astype(np.float64))
        ar[:, t+1] = predictor(ar[:, t], actions[:, t].astype(np.float64))
    require(np.isfinite(tf).all() and np.isfinite(ar).all(), 'nonfinite prediction; evaluation failed')
    np.testing.assert_array_equal(tf[:, 0], ar[:, 1])
    return tf, ar


def summarize(tf, ar, truth, horizons):
    result = {}
    for h in horizons:
        result[str(h)] = {
            'episodes': len(truth), 'prefix_transitions': len(truth)*h,
            'tf_local_at_h': metrics(tf[:, h-1], truth[:, h]),
            'ar_endpoint_at_h': metrics(ar[:, h], truth[:, h]),
            'tf_prefix': metrics(tf[:, :h], truth[:, 1:h+1]),
            'ar_prefix': metrics(ar[:, 1:h+1], truth[:, 1:h+1])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('tmp/s3_1_final'))
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--bias-mm', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=Path('tmp/s3_2'))
    args = parser.parse_args()
    if not 21 <= args.steps <= 500 or not np.isfinite(args.bias_mm) or abs(args.bias_mm) > 10:
        parser.error('steps must be 21..500; bias-mm must be finite within [-10,10]')
    model, data, hashes = load_source(args.checkpoint)
    truth, actions = extended_episodes(data, args.steps)
    bias = np.array([args.bias_mm/1000, 0.])
    predictors = {'persistence': lambda s, a: s,
                  'ideal_addition': lambda s, a: s+a,
                  'mlp': lambda s, a: predict(s, a, model),
                  'mlp_plus_bias': lambda s, a: predict(s, a, model)+bias,
                  'oracle': lambda s, a: transition(s.astype(np.float32), a.astype(np.float32))}
    horizons = [1, 5, 20, args.steps]
    report = {'schema': 's3_2_multistep_rollout/1.0', 'source_directory': str(args.checkpoint.resolve()),
              'source_sha256': hashes, 'steps': args.steps, 'dt_s': .1, 'bias_mm': args.bias_mm,
              'protocol': 'All 10 validation episodes, same fixed actions; TF truth each step; AR truth only at t0.',
              'evaluation': {}}
    arrays = {'truth_state_m': truth, 'action_m': actions,
              'timestamp_s': np.arange(args.steps+1, dtype=np.float64)*.1,
              'validation_episode_ids': data['validation_episode_ids']}
    curves = ['model,step,tf_local_rmse_m,ar_endpoint_rmse_m']
    for name, predictor in predictors.items():
        tf, ar = evaluate(predictor, truth, actions)
        arrays[name+'_tf_next_state_m'], arrays[name+'_ar_state_m'] = tf, ar
        report['evaluation'][name] = summarize(tf, ar, truth, horizons)
        for h in range(1, args.steps+1):
            curves.append(f"{name},{h},{metrics(tf[:, h-1], truth[:, h])['rmse_m']},"
                          f"{metrics(ar[:, h], truth[:, h])['rmse_m']}")
    # Original S3.1 aggregate validation MSE uses all 200 one-step predictions.
    report['s3_1_replay'] = metrics(arrays['mlp_tf_next_state_m'][:, :20], truth[:, 1:21])
    train_states = data['state_m'][data['train_episode_ids'], :-1].reshape(-1, 2)
    lo, hi = train_states.min(0), train_states.max(0)
    report['training_state_box_m'] = {'min': lo.tolist(), 'max': hi.tolist()}
    report['outside_training_state_box'] = {}
    for h in horizons:
        fraction = lambda s: float(np.mean(np.any((s < lo) | (s > hi), axis=-1)))
        report['outside_training_state_box'][str(h)] = {
            'truth_input_fraction': fraction(truth[:, :h]),
            'mlp_ar_input_fraction': fraction(arrays['mlp_ar_state_m'][:, :h])}
    # At zero command the true state is constant; add known bias to exact dynamics.
    probe_truth = np.zeros((10, args.steps+1, 2), dtype=np.float32)
    probe_actions = np.zeros_like(actions)
    probe_tf, probe_ar = evaluate(lambda s, a: s+bias, probe_truth, probe_actions)
    np.testing.assert_allclose(probe_ar[:, -1], np.tile(args.steps*bias, (10, 1)), atol=1e-12)
    report['analytic_zero_action_bias_probe'] = {
        'tf_final_l2_m': float(np.linalg.norm(probe_tf[0, -1])),
        'ar_final_l2_m': float(np.linalg.norm(probe_ar[0, -1])),
        'meaning': 'Exact identity dynamics plus constant bias; not a learned-model claim.'}
    # Full trajectory timing, batch10; excludes truth generation, metrics and file I/O.
    rollout = lambda: evaluate(predictors['mlp'], truth, actions)
    for _ in range(3):
        rollout()
    times = []
    for _ in range(20):
        start = perf_counter_ns()
        rollout()
        times.append((perf_counter_ns()-start)/1e6)
    report['tf_and_ar_batch10_latency_ms'] = {'p50': float(np.percentile(times, 50)),
                                             'p95': float(np.percentile(times, 95)),
                                             'warmup': 3, 'samples': 20}
    require(all(digest(args.checkpoint/k) == value for k, value in hashes.items()), 'source mutated')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'rollouts.npz', **arrays)
    (args.output/'curves.csv').write_text('\n'.join(curves)+'\n')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    contract = {'schema': 's3_2_rollout/1.0', 'source_profile': 's3_1_xy_command_dynamics/1.0',
                'truth_state_m': f'float32 [10,{args.steps+1},2], world XY, dt=.1s',
                'action_m': f'float32 [10,{args.steps},2], commanded world XY displacement [m]',
                'prediction': 'float64; *_tf_next_state_m [10,H,2]; *_ar_state_m [10,H+1,2]',
                'time': 'float64 [H+1], seconds; no action at terminal state',
                'extension': 'First20 source validation transitions unchanged; new commands seed3200+episode_id',
                'boundary': 'H transitions per independent reset; time truncation, no success labels',
                'evidence': 'No fitting; frozen model/statistics; no policy, feedback controller or robot success.'}
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    print('model: TF local / AR endpoint RMSE [mm], at horizons', horizons)
    for name in predictors:
        print(name, [(h, round(report['evaluation'][name][str(h)]['tf_local_at_h']['rmse_m']*1000, 4),
                      round(report['evaluation'][name][str(h)]['ar_endpoint_at_h']['rmse_m']*1000, 4))
                     for h in horizons])
    print('S3.1 replay:', report['s3_1_replay'])
    print('outside train box:', report['outside_training_state_box'])
    print('analytic bias:', report['analytic_zero_action_bias_probe'])


if __name__ == '__main__':
    main()
