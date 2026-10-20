"""CPU reward/success/contact/failure heads with explicit one-transition labels."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from s3_1_state_dynamics import forward, transition

GOAL = np.array([.09, 0.], dtype=np.float32)
WALL_X = np.float32(.12)
GOAL_RADIUS = .045
HEADS = ('success_next', 'contact_transition', 'failure_next')


def step(state, action):
    """Commanded displacement -> attempted nominal position -> unilateral x wall."""
    attempted = transition(state, action)
    contact = attempted[..., 0] > WALL_X
    next_state = attempted.copy()
    next_state[..., 0] = np.minimum(next_state[..., 0], WALL_X)
    return next_state, contact


def annotate(state, next_state, contact):
    distance_before = np.linalg.norm(state-GOAL, axis=-1)
    distance_after = np.linalg.norm(next_state-GOAL, axis=-1)
    success = distance_after <= GOAL_RADIUS
    failure = (next_state[..., 0] < -.20) | (np.abs(next_state[..., 1]) > .20)
    reward = ((distance_before-distance_after)/.04-.2*contact-.5*failure).astype(np.float32)
    return reward, np.stack([success, contact, failure], axis=-1)


def dataset():
    rng = np.random.default_rng(350)
    states, actions = [], []
    for episode in range(60):
        trajectory = [rng.uniform([-.18, -.22], [.115, .22]).astype(np.float32)]
        commands = []
        side = 1 if trajectory[0][1] >= 0 else -1
        for _ in range(20):
            mode = episode % 4
            if mode == 0:
                action = rng.uniform(-.06, .06, 2)
            elif mode == 1:
                action = np.clip(GOAL-trajectory[-1], -.04, .04)+rng.uniform(-.005, .005, 2)
            elif mode == 2:
                action = np.array([rng.uniform(.02, .06), rng.uniform(-.02, .02)])
            else:
                action = np.array([rng.uniform(-.04, .04), side*.05])
            action = action.astype(np.float32)
            next_state, _ = step(trajectory[-1][None], action[None])
            trajectory.append(next_state[0])
            commands.append(action)
        states.append(trajectory)
        actions.append(commands)
    states, actions = np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)
    flat_next, contact = step(states[:, :-1].reshape(-1, 2), actions.reshape(-1, 2))
    np.testing.assert_array_equal(flat_next, states[:, 1:].reshape(-1, 2))
    rewards, labels = annotate(states[:, :-1], states[:, 1:], contact.reshape(60, 20))
    return states, actions, rewards, labels


def inputs(state, action):
    """Causal current state and candidate command only; no labels or next state."""
    return np.concatenate([state, action], axis=-1)


def sigmoid(logits):
    e = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1/(1+e), e/(1+e))


def loss_and_gradients(x, y, weights):
    output, h = forward(x, weights)
    reward_loss = np.mean((output[:, 0]-y[:, 0])**2)
    logits, truth = output[:, 1:], y[:, 1:]
    classification_loss = np.mean(np.logaddexp(0, logits)-truth*logits)
    d = np.empty_like(output)
    d[:, 0] = 2*(output[:, 0]-y[:, 0])/len(x)
    d[:, 1:] = (sigmoid(logits)-truth)/truth.size
    back = (d@weights[2].T)*(1-h*h)
    grads = [x.T@back, back.sum(0), h.T@d, d.sum(0)]
    return float(reward_loss+classification_loss), grads


def fit(x, rewards, labels, epochs):
    stats = {'input_mean': x.mean(0), 'input_std': x.std(0),
             'reward_mean': rewards.mean(), 'reward_std': rewards.std()}
    if not np.all(stats['input_std'] > 0) or stats['reward_std'] <= 0:
        raise ValueError('degenerate training statistics')
    nx = (x-stats['input_mean'])/stats['input_std']
    y = np.column_stack([(rewards-stats['reward_mean'])/stats['reward_std'], labels]).astype(np.float64)
    rng = np.random.default_rng(352)
    weights = [rng.normal(0, .2, (4, 32)), np.zeros(32), rng.normal(0, .15, (32, 4)), np.zeros(4)]
    _, analytic = loss_and_gradients(nx[:12], y[:12], weights)
    errors = []
    for w, grad in zip(weights, analytic):
        index = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
        original = w[index]
        w[index] = original+1e-5
        plus = loss_and_gradients(nx[:12], y[:12], weights)[0]
        w[index] = original-1e-5
        minus = loss_and_gradients(nx[:12], y[:12], weights)[0]
        w[index] = original
        errors.append(float(abs((plus-minus)/2e-5-grad[index])))
    assert max(errors) < 1e-7
    m, v = [np.zeros_like(w) for w in weights], [np.zeros_like(w) for w in weights]
    losses = []
    for iteration in range(1, epochs+1):
        _, grads = loss_and_gradients(nx, y, weights)
        for i, grad in enumerate(grads):
            m[i] = .9*m[i]+.1*grad
            v[i] = .999*v[i]+.001*grad*grad
            weights[i] -= .01*(m[i]/(1-.9**iteration))/(np.sqrt(v[i]/(1-.999**iteration))+1e-8)
        if iteration == 1 or iteration % 100 == 0 or iteration == epochs:
            losses.append([iteration, loss_and_gradients(nx, y, weights)[0]])
    return weights, stats, losses, errors


def predict(x, weights, stats):
    output, _ = forward((x-stats['input_mean'])/stats['input_std'], weights)
    return output[:, 0]*stats['reward_std']+stats['reward_mean'], sigmoid(output[:, 1:])


def binary_metrics(probability, target, threshold):
    decision = probability >= threshold
    target = target.astype(bool)
    tp, fp = int(np.sum(decision & target)), int(np.sum(decision & ~target))
    fn, tn = int(np.sum(~decision & target)), int(np.sum(~decision & ~target))
    ratio = lambda a, b: float(a/b) if b else None
    p = np.clip(probability, 1e-12, 1-1e-12)
    return {'positive_labels': int(target.sum()), 'predicted_positive': int(decision.sum()),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'accuracy': float(np.mean(decision == target)),
            'precision': ratio(tp, tp+fp), 'recall': ratio(tp, tp+fn),
            'false_positive_rate': ratio(fp, fp+tn), 'false_negative_rate': ratio(fn, fn+tp),
            'brier': float(np.mean((probability-target)**2)),
            'bce': float(-np.mean(target*np.log(p)+(~target)*np.log(1-p)))}


def evaluate(pred_reward, probability, rewards, labels, threshold):
    return {'samples': len(rewards),
            'reward_mae': float(np.mean(np.abs(pred_reward-rewards))),
            'reward_rmse': float(np.sqrt(np.mean((pred_reward-rewards)**2))),
            'classification': {name: binary_metrics(probability[:, i], labels[:, i], threshold)
                               for i, name in enumerate(HEADS)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--output', type=Path, default=Path('tmp/s3_5'))
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10000 or not np.isfinite(args.threshold) or not 0 < args.threshold < 1:
        parser.error('epochs must be 1..10000; threshold must be finite in (0,1)')
    args.output.mkdir(parents=True, exist_ok=False)
    states, actions, rewards, labels = dataset()
    ids = np.random.default_rng(351).permutation(60)
    train_ids, val_ids = ids[:40], ids[40:]
    assert not set(train_ids) & set(val_ids)
    x = inputs(states[:, :-1], actions)
    tx, tr, ty = x[train_ids].reshape(-1, 4), rewards[train_ids].reshape(-1), labels[train_ids].reshape(-1, 3)
    weights, stats, losses, errors = fit(tx, tr, ty, args.epochs)
    np.savez_compressed(args.output/'model.npz', **stats,
                        **{k: w for k, w in zip(('w1', 'b1', 'w2', 'b2'), weights)})
    # Episode outcome is a separate result field; never used as a one-step head label or input.
    episode_success = labels[:, -1, 0] & ~np.any(labels[:, :, 2], axis=1)
    np.savez_compressed(args.output/'episodes.npz', state_m=states, action_m=actions, reward=rewards,
                        event_labels=labels, episode_success=episode_success,
                        train_episode_ids=train_ids, validation_episode_ids=val_ids,
                        behavior_mode=np.arange(60)%4, timestamp_s=np.arange(21, dtype=np.float64)*.1,
                        truncated=np.tile(np.arange(20) == 19, (60, 1)), terminated=np.zeros((60, 20), dtype=bool))
    vx = x[val_ids].reshape(-1, 4)
    ox = vx.copy()
    ox[:, 2:] *= np.float32(3)
    onext, oc = step(ox[:, :2], ox[:, 2:])
    oreward, olabels = annotate(ox[:, :2], onext, oc)
    groups = {'train': (tx, tr, ty),
              'validation': (vx, rewards[val_ids].reshape(-1), labels[val_ids].reshape(-1, 3)),
              'action_scale3': (ox, oreward, olabels)}
    priors = ty.mean(0)
    report = {'schema': 's3_5_outcome_heads/1.0', 'parameters': 292, 'epochs': args.epochs,
              'threshold': args.threshold, 'seeds_data_split_model': [350, 351, 352],
              'binary_head_order': list(HEADS), 'gradient_absolute_errors': errors,
              'train_losses': losses, 'train_label_priors': priors.tolist(), 'evaluation': {}}
    arrays = {}
    for name, (rx, rr, ry) in groups.items():
        prediction, probability = predict(rx, weights, stats)
        constant = evaluate(np.full(len(rr), tr.mean()), np.broadcast_to(priors, ry.shape), rr, ry, args.threshold)
        zero_x = rx.copy()
        zero_x[:, 2:] = 0
        zero_r, zero_p = predict(zero_x, weights, stats)
        report['evaluation'][name] = {'learned': evaluate(prediction, probability, rr, ry, args.threshold),
                                      'train_constant': constant,
                                      'zero_action_ablation': evaluate(zero_r, zero_p, rr, ry, args.threshold)}
        if name != 'train':
            for key, value in [('input_m', rx), ('reward', rr), ('labels', ry),
                               ('predicted_reward', prediction), ('probability', probability)]:
                arrays[name+'_'+key] = value
    for _ in range(20):
        predict(vx[:1], weights, stats)
    times = []
    for _ in range(200):
        start = perf_counter_ns()
        predict(vx[:1], weights, stats)
        times.append((perf_counter_ns()-start)/1e6)
    report['batch1_heads_latency_ms'] = {'p50': float(np.percentile(times, 50)),
                                        'p95': float(np.percentile(times, 95)), 'warmup': 20, 'samples': 200}
    report['limitations'] = 'Synthetic clipped point wall, no contact forces or robot task policy; logits not calibrated risk.'
    np.savez_compressed(args.output/'evaluation.npz', **arrays)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    contract = {'schema': 's3_5_wall_outcomes/1.0',
                'state': 'float32 [60,21,2], world XY metres; initial x<=.115; no force/velocity',
                'action': 'float32 [60,20,2], commanded world XY displacement[m], dt=.1s',
                'dynamics': 'S3.1 attempted nominal transition, then next_x=min(attempted_x,.12m)',
                'reward': 'float32 [60,20], dimensionless; (distance_before-distance_after)/.04m-.2*contact-.5*failure_next',
                'success_next': 'bool goal occupancy at t+1: L2(next_state-[.09,0]m)<=.045m; NOT eventual task success',
                'contact_transition': 'bool attempted_next_x>.12m; geometric active wall constraint, no force/contact solver',
                'failure_next': 'bool next_x<-.20m OR abs(next_y)>.20m; one-step boundary violation, NOT eventual failure risk',
                'event_order': list(HEADS), 'label_relationship': 'separate definitions, not statistically independent',
                'episode_success': 'last success_next AND no failure_next in episode; result only, never input or head target',
                'boundary': '20 transitions/21 observations, timestamp float64 dt=.1; time truncation only, events do not terminate',
                'input': 'float32 [N,4] current state and command ONLY; normalized with original train800 statistics',
                'output': 'float64 [N] reward and [N,3] sigmoid scores; mean/std fit train only; no calibration guarantee',
                'split': 'seed351 whole episodes: train40/validation20; behavior modes random/goal/wall/outward, all failures kept',
                'ood': 'action_scale3 is 400 paired counterfactual single-step tuples, not continuous rollout',
                'threshold': 'fixed CLI diagnostic threshold, default .5, not optimized on validation/OOD',
                'modality': 'state/action only; no language, images, force/tactile or future observations'}
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    for name in ('validation', 'action_scale3'):
        v = report['evaluation'][name]['learned']
        print(name, 'reward RMSE', v['reward_rmse'])
        for head in HEADS:
            c = v['classification'][head]
            print(head, 'positives/TP/FP/FN/TN', [c[k] for k in ('positive_labels', 'tp', 'fp', 'fn', 'tn')])
    print('gradient errors', errors)


if __name__ == '__main__':
    main()
