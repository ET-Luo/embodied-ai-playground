"""Single-step CPU dynamics learning; no policy, planning or multi-step rollout."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np


def transition(state, action):
    """World XY position [m], commanded displacement [m], dt=0.1s."""
    gain = .70 + .15 * np.tanh(state / .25)
    return (state + gain * action + .08 * action[:, ::-1]).astype(np.float32)


def dataset():
    rng = np.random.default_rng(31)
    states, actions = [], []
    for _ in range(40):
        trajectory = [rng.uniform(-.3, .3, 2).astype(np.float32)]
        commands = rng.uniform(-.04, .04, (20, 2)).astype(np.float32)
        for action in commands:
            trajectory.append(transition(trajectory[-1][None], action[None])[0])
        states.append(trajectory)
        actions.append(commands)
    return np.asarray(states, dtype=np.float32), np.asarray(actions)


def forward(x, weights):
    w1, b1, w2, b2 = weights
    h = np.tanh(x @ w1 + b1)
    return h @ w2 + b2, h


def gradients(x, y, weights):
    pred, h = forward(x, weights)
    d = 2 * (pred - y) / y.size
    dh = (d @ weights[2].T) * (1 - h * h)
    return [x.T @ dh, dh.sum(0), h.T @ d, d.sum(0)]


def metrics(pred, target):
    error = pred - target
    return {'mse_m2': float(np.mean(error ** 2)),
            'rmse_m': float(np.sqrt(np.mean(error ** 2))),
            'p95_l2_m': float(np.percentile(np.linalg.norm(error, axis=-1), 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=Path('tmp/s3_1'))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {
        'schema': 's3_1_xy_command_dynamics/1.0',
        'source': 'synthetic deterministic generator; not measured robot data',
        'embodiment': 'fully observed memoryless world XY point with position-dependent gain',
        'state': 'float32 [episode,21,2], world XY measured position [m]',
        'action': 'float32 [episode,20,2], commanded world XY displacement [m]; NOT realized displacement',
        'time': 'float64 [21], relative seconds, dt=0.1; action[t] acts from state[t] to state[t+1]',
        'boundary': '20 transitions per independent reset episode, time truncation, no success label',
        'modalities': 'position and action only; no image, language, force or velocity',
        'dynamics': 's_next=s+(0.70+0.15*tanh(s/0.25))*a+0.08*reverse_xy(a)',
        'split': '40 independent episodes; seed32 permutation; first30 train, remaining10 validation',
        'normalization': 'per-channel mean/std fit only 600 train transitions; target is s_next-s',
        'compute': 'raw float32, NumPy MLP/Adam float64, CPU; batch evaluation float64',
    }
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    states, actions = dataset()
    # Entire independent reset episodes are split before flattening transitions.
    ids = np.random.default_rng(32).permutation(40)
    train_ids, val_ids = ids[:30], ids[30:]
    assert not set(train_ids) & set(val_ids)
    np.savez_compressed(args.output/'episodes.npz', state_m=states, action_m=actions,
                        timestamp_s=np.arange(21, dtype=np.float64)*.1,
                        train_episode_ids=train_ids, validation_episode_ids=val_ids)
    batches = {}
    for name, indices in [('train', train_ids), ('validation', val_ids)]:
        s = states[indices, :-1].reshape(-1, 2)
        a = actions[indices].reshape(-1, 2)
        target = states[indices, 1:].reshape(-1, 2)
        batches[name] = (np.concatenate([s, a], axis=1), target - s, target)
    raw_x, raw_y, _ = batches['train']
    xm, xs = raw_x.mean(0), raw_x.std(0)
    ym, ys = raw_y.mean(0), raw_y.std(0)
    assert np.all(xs > 0) and np.all(ys > 0)
    x, y = (raw_x-xm)/xs, (raw_y-ym)/ys
    rng = np.random.default_rng(33)
    weights = [rng.normal(0, .2, (4, 32)), np.zeros(32),
               rng.normal(0, .15, (32, 2)), np.zeros(2)]
    # Central difference checks an active coordinate in every parameter tensor.
    analytic = gradients(x[:8], y[:8], weights)
    gradient_errors = []
    for w, grad in zip(weights, analytic):
        index = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
        original = w[index]
        w[index] = original + 1e-5
        plus = np.mean((forward(x[:8], weights)[0]-y[:8])**2)
        w[index] = original - 1e-5
        minus = np.mean((forward(x[:8], weights)[0]-y[:8])**2)
        w[index] = original
        gradient_errors.append(float(abs((plus-minus)/2e-5-grad[index])))
    assert max(gradient_errors) < 1e-7
    m, v = [np.zeros_like(w) for w in weights], [np.zeros_like(w) for w in weights]
    losses = []
    for step in range(1, args.epochs+1):
        for i, grad in enumerate(gradients(x, y, weights)):
            m[i] = .9*m[i] + .1*grad
            v[i] = .999*v[i] + .001*grad*grad
            weights[i] -= .01*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
        if step == 1 or step % 100 == 0 or step == args.epochs:
            losses.append([step, float(np.mean((forward(x, weights)[0]-y)**2))])
    np.savez_compressed(args.output/'model.npz', w1=weights[0], b1=weights[1],
                        w2=weights[2], b2=weights[3], x_mean=xm, x_std=xs, y_mean=ym, y_std=ys)
    report = {'schema': 's3_1_state_dynamics/1.0', 'epochs': args.epochs,
              'seed_data_split_model': [31, 32, 33], 'parameters': 226,
              'train_episodes_transitions': [30, 600], 'validation_episodes_transitions': [10, 200],
              'gradient_absolute_errors': gradient_errors, 'normalized_train_loss': losses,
              'evaluation': {}}
    for name, (rx, _, target) in batches.items():
        pred = rx[:, :2] + forward((rx-xm)/xs, weights)[0]*ys+ym
        report['evaluation'][name] = {
            'persistence': metrics(rx[:, :2], target),
            'ideal_addition': metrics(rx[:, :2]+rx[:, 2:], target),
            'mlp': metrics(pred, target)}
        if name == 'validation':
            np.savez_compressed(args.output/'validation_predictions.npz', input_m=rx,
                                target_next_state_m=target, predicted_next_state_m=pred)
    rx = batches['validation'][0][:1]
    for _ in range(30):
        forward((rx-xm)/xs, weights)
    times = []
    for _ in range(200):
        start = perf_counter_ns()
        prediction = rx[:, :2] + forward((rx-xm)/xs, weights)[0]*ys+ym
        times.append((perf_counter_ns()-start)/1e6)
    report['batch1_latency_ms'] = {'p50': float(np.percentile(times, 50)),
                                    'p95': float(np.percentile(times, 95)), 'warmup': 30, 'samples': 200}
    report['limitations'] = 'Synthetic deterministic fully observed XY; one-step only; no robot success or OOD claim.'
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
