"""CPU NumPy MLP: current world XY + literal language -> displacement."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require, toy_episode, validate
from s0_5_action_representation import round_trip

LANGUAGES = ('move left', 'move right')
GOALS = np.array([[-.24, 0], [.24, 0]], dtype=np.float32)
STEP = .06


def encode(state, language):
    require(language in LANGUAGES, 'unknown literal instruction')
    state = np.asarray(state)
    require(state.shape == (2,) and state.dtype == np.float32 and np.isfinite(state).all(), 'finite float32 world XY')
    return np.concatenate((state / .3, np.eye(2, dtype=np.float32)[LANGUAGES.index(language)]))


def expert(state, index):
    return np.clip(GOALS[index] - state, -STEP, STEP)


def dataset(output):
    rng = np.random.default_rng(11)
    starts = rng.uniform([-.18, -.15], [.18, .15], (30, 2)).astype(np.float32)
    rows = {'train': [], 'validation': []}
    for group, start in enumerate(starts):
        split = 'train' if group < 24 else 'validation'
        for index, language in enumerate(LANGUAGES):
            states = [start.copy()]; actions = []
            for _ in range(10):
                action = expert(states[-1], index)
                actions.append(action); states.append(states[-1] + action)
            arrays, meta = toy_episode(1)
            arrays.update(proprioception=np.asarray(states, dtype=np.float32), action=np.asarray(actions, dtype=np.float32),
                          timestamp=np.arange(11, dtype=np.float64)*.1, terminated=np.zeros(10, dtype=bool),
                          truncated=np.array([False]*9+[True]))
            np.testing.assert_allclose(arrays['proprioception'][1:], arrays['proprioception'][:-1]+arrays['action'])
            require(np.max(np.abs(states[-1]-GOALS[index])) <= .01, 'expert final goal')
            meta.update(episode_id=f's1_1_g{group}_{index}', language_instruction=language,
                        task_name='literal_world_xy_goal', success=True,
                        success_definition='final max XY goal error <= 0.01 m', embodiment='ideal_world_xy_point_v1')
            meta['provenance'].update(source='synthetic S1.1 expert', group_id=f'reset_{group}', split=split)
            for key, value in arrays.items():
                meta['modalities'][key]['shape'] = list(value.shape)
            validate(arrays, meta)
            arrays, meta = round_trip(output/'episodes'/meta['episode_id'], arrays, meta)
            for t in range(10):
                rows[split].append((encode(arrays['proprioception'][t], language), arrays['action'][t], group))
    batches = {}
    for split, values in rows.items():
        x = np.stack([r[0] for r in values]); y = np.stack([r[1] for r in values])
        groups = np.array([r[2] for r in values])
        np.savez_compressed(output/(split+'.npz'), x=x, action_m=y, group=groups)
        batches[split] = (x, y, groups)
    require(not set(batches['train'][2]) & set(batches['validation'][2]), 'reset overlap')
    return batches, starts[24:]


def forward(x, weights):
    w1, b1, w2, b2 = weights
    hidden = np.tanh(x @ w1 + b1)
    return hidden @ w2 + b2, hidden


def gradients(x, y, weights):
    predicted, hidden = forward(x, weights)
    delta = 2*(predicted-y)/y.size
    back = (delta @ weights[2].T)*(1-hidden*hidden)
    return [x.T @ back, back.sum(0), hidden.T @ delta, delta.sum(0)]


def train(x, y, epochs, seed):
    rng = np.random.default_rng(seed)
    weights = [rng.normal(0, .3, (4, 32)).astype(np.float32), np.zeros(32, np.float32),
               rng.normal(0, .1, (32, 2)).astype(np.float32), np.zeros(2, np.float32)]
    first = [np.zeros_like(w) for w in weights]; second = [w.copy() for w in first]
    history = []
    for step in range(1, epochs+1):
        grad = gradients(x, y/STEP, weights)
        for i in range(4):
            first[i] = .9*first[i] + .1*grad[i]
            second[i] = .999*second[i] + .001*grad[i]**2
            weights[i] -= .01*(first[i]/(1-.9**step))/(np.sqrt(second[i]/(1-.999**step))+1e-8)
        if step == 1 or step % 100 == 0 or step == epochs:
            history.append(dict(epoch=step, train_mse_m2=float(np.mean((forward(x, weights)[0]*STEP-y)**2))))
    return weights, history


def predict(state, language, weights, ablated=False):
    x = encode(state, language)
    if ablated:
        x[2:] = 0
    return np.clip(forward(x[None], weights)[0][0]*STEP, -STEP, STEP)


def evaluate(starts, weights, ablated):
    results = []; latencies = []
    for group, start in enumerate(starts, 24):
        for index, language in enumerate(LANGUAGES):
            state = start.copy(); trace = [state.tolist()]; commands = []
            for _ in range(10):
                before = perf_counter_ns()
                action = predict(state, language, weights, ablated)
                latencies.append((perf_counter_ns()-before)/1e6)
                state = state+action; trace.append(state.tolist()); commands.append(action.tolist())
            error = float(np.max(np.abs(state-GOALS[index])))
            results.append(dict(group=group, language=language, success=error <= .01, final_error_m=error,
                                failure_stage=None if error <= .01 else 'final goal error after 10-step budget',
                                truncated=True, trace=trace, commands=commands))
    return dict(successes=sum(r['success'] for r in results), denominator=len(results), episodes=results,
                latency_ms=dict(median=float(np.median(latencies)), p95=float(np.percentile(latencies, 95)),
                                samples=len(latencies), scope='encode + NumPy MLP + clipping; excludes environment, no deployment claim'))


def gradient_check():
    rng = np.random.default_rng(2)
    weights = [rng.normal(size=(4, 3))*.1, np.zeros(3), rng.normal(size=(3, 2))*.1, np.zeros(2)]
    x = rng.normal(size=(5, 4)); y = rng.normal(size=(5, 2))
    analytic = gradients(x, y, weights)
    worst = 0.
    for parameter, grad in zip(weights, analytic):
        for index in np.ndindex(parameter.shape):
            old = parameter[index]; eps = 1e-5
            parameter[index] = old+eps; plus = np.mean((forward(x, weights)[0]-y)**2)
            parameter[index] = old-eps; minus = np.mean((forward(x, weights)[0]-y)**2)
            parameter[index] = old
            worst = max(worst, abs((plus-minus)/(2*eps)-grad[index]))
    require(worst < 1e-8, 'manual gradient check')
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('tmp/s1_1'))
    args = parser.parse_args()
    require(args.epochs > 0 and not args.output.exists(), 'positive epochs and fresh output required')
    check = gradient_check()
    args.output.mkdir(parents=True)
    batches, starts = dataset(args.output)
    rejected = []
    for language, state in [('go left', np.zeros(2, np.float32)), ('move left', np.zeros(2, np.float64)),
                            ('move left', np.full(2, np.nan, np.float32))]:
        try:
            encode(state, language)
        except ValueError:
            rejected.append(True)
        else:
            raise RuntimeError('invalid policy input accepted')
    report = dict(rejected_input_cases=len(rejected), environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, numpy_path=np.__file__),
                  seed=args.seed, epochs=args.epochs, gradient_max_absolute_error=check, models={})
    for name, ablated in [('conditioned', False), ('no_language', True)]:
        x, y, _ = batches['train']; x = x.copy()
        if ablated:
            x[:, 2:] = 0
        weights, history = train(x, y, args.epochs, args.seed)
        np.savez_compressed(args.output/(name+'_weights.npz'), **dict(zip(('w1', 'b1', 'w2', 'b2'), weights)))
        with np.load(args.output/(name+'_weights.npz'), allow_pickle=False) as saved:
            loaded = [saved[key] for key in ('w1', 'b1', 'w2', 'b2')]
            np.testing.assert_array_equal(forward(x, weights)[0], forward(x, loaded)[0])
        metrics = {}
        for split, (features, labels, _) in batches.items():
            features = features.copy()
            if ablated:
                features[:, 2:] = 0
            physical = forward(features, weights)[0]*STEP
            metrics[split] = dict(samples=len(labels), raw_mse_m2=float(np.mean((physical-labels)**2)),
                                  executed_mse_m2=float(np.mean((np.clip(physical, -STEP, STEP)-labels)**2)))
        pair = [predict(np.zeros(2, np.float32), language, weights, ablated).tolist() for language in LANGUAGES]
        report['models'][name] = dict(metrics=metrics, history=history, same_state_actions_m=pair,
                                     rollout=evaluate(starts, weights, ablated))
    report['limitations'] = 'Fixed literal one-hot, static two goals, ideal point, six held-out resets; no vision, paraphrase, real robot or broad generalization evidence. Fixed epochs, no validation tuning.'
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name:dict(metrics=value['metrics'], pair=value['same_state_actions_m'],
                               success=f"{value['rollout']['successes']}/{value['rollout']['denominator']}")
                      for name, value in report['models'].items()}, indent=2))
    print(f'PASS gradient check={check:.3g}, raw contract/round-trip, reset split, model reload; {args.output}')


if __name__ == '__main__':
    main()
