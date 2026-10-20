"""CPU episode-bootstrap ensemble: disagreement is a score, not calibrated risk."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from s3_1_state_dynamics import dataset, forward, gradients, metrics, transition


def calibration_episodes():
    rng = np.random.default_rng(330)
    states, actions = [], []
    for _ in range(10):
        s = [rng.uniform(-.3, .3, 2).astype(np.float32)]
        a = rng.uniform(-.04, .04, (20, 2)).astype(np.float32)
        for command in a:
            s.append(transition(s[-1][None], command[None])[0])
        states.append(s)
        actions.append(a)
    return np.asarray(states, dtype=np.float32), np.asarray(actions)


def flatten(states, actions):
    x = np.concatenate([states[:, :-1].reshape(-1, 2), actions.reshape(-1, 2)], axis=1)
    return x, states[:, 1:].reshape(-1, 2)


def fit_statistics(x, target):
    delta = target-x[:, :2]
    return {'x_mean': x.mean(0), 'x_std': x.std(0),
            'y_mean': delta.mean(0), 'y_std': delta.std(0)}


def train(x, y, epochs, seed):
    rng = np.random.default_rng(seed)
    weights = [rng.normal(0, .2, (4, 32)), np.zeros(32),
               rng.normal(0, .15, (32, 2)), np.zeros(2)]
    m, v = [np.zeros_like(w) for w in weights], [np.zeros_like(w) for w in weights]
    losses = []
    for step in range(1, epochs+1):
        for i, grad in enumerate(gradients(x, y, weights)):
            m[i] = .9*m[i]+.1*grad
            v[i] = .999*v[i]+.001*grad*grad
            weights[i] -= .01*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
        if step == 1 or step % 100 == 0 or step == epochs:
            losses.append([step, float(np.mean((forward(x, weights)[0]-y)**2))])
    return weights, losses


def predict_members(x, weights, stats):
    normalized = (x-stats['x_mean'])/stats['x_std']
    return np.stack([x[:, :2]+forward(normalized, w)[0]*stats['y_std']+stats['y_mean']
                     for w in weights])


def disagreement(predictions):
    """Population spread over K members and D coordinates, physical metres."""
    if predictions.ndim != 3 or predictions.shape[0] < 2 or not np.isfinite(predictions).all():
        raise ValueError('finite predictions [K>=2,N,D] required')
    mean = predictions.mean(0)
    score = np.sqrt(np.mean((predictions-mean[None])**2, axis=(0, 2)))
    return mean, score


def confusion(score, error, threshold, error_limit=.005):
    flagged, bad = score > threshold, error > error_limit
    return {'tp': int(np.sum(flagged & bad)), 'fp': int(np.sum(flagged & ~bad)),
            'fn': int(np.sum(~flagged & bad)), 'tn': int(np.sum(~flagged & ~bad))}


def summarize(predictions, target, threshold):
    mean, score = disagreement(predictions)
    error = np.linalg.norm(mean-target, axis=-1)
    corr = None
    if np.std(score) > 1e-15 and np.std(error) > 1e-15:
        corr = float(np.corrcoef(score, error)[0, 1])
    return {'samples': len(target), 'ensemble_mean': metrics(mean, target),
            'members': [metrics(p, target) for p in predictions],
            'disagreement_median_m': float(np.median(score)),
            'disagreement_p95_m': float(np.percentile(score, 95)),
            'score_error_pearson': corr, 'flagged': int(np.sum(score > threshold)),
            'high_error_l2_over_5mm': int(np.sum(error > .005)),
            'confusion': confusion(score, error, threshold)}, mean, score, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--shift-mm', type=float, default=10.)
    parser.add_argument('--output', type=Path, default=Path('tmp/s3_3'))
    args = parser.parse_args()
    if not 2 <= args.members <= 10 or not 1 <= args.epochs <= 10000:
        parser.error('members must be 2..10; epochs must be 1..10000')
    if not np.isfinite(args.shift_mm) or not 0 <= args.shift_mm <= 50:
        parser.error('shift-mm must be finite within [0,50]')
    args.output.mkdir(parents=True, exist_ok=False)
    states, actions = dataset()
    ids = np.random.default_rng(32).permutation(40)
    train_ids, val_ids = ids[:30], ids[30:]
    cs, ca = calibration_episodes()
    tx, target = flatten(states[train_ids], actions[train_ids])
    vx, vtarget = flatten(states[val_ids], actions[val_ids])
    cx, ctarget = flatten(cs, ca)
    stats = fit_statistics(tx, target)
    assert all(np.all(stats[k] > 0) for k in ('x_std', 'y_std'))
    weights, boot_ids, losses = [], [], []
    for member in range(args.members):
        # Draw 30 whole episodes with replacement, preserving all 20 transitions each.
        chosen = np.random.default_rng(3300+member).choice(train_ids, 30, replace=True)
        assert set(chosen) <= set(train_ids) and not set(chosen) & set(val_ids)
        x, y = flatten(states[chosen], actions[chosen])
        w, loss = train((x-stats['x_mean'])/stats['x_std'],
                        (y-x[:, :2]-stats['y_mean'])/stats['y_std'], args.epochs, 3400+member)
        weights.append(w)
        boot_ids.append(chosen)
        losses.append(loss)
    checkpoint = {k: np.stack([w[i] for w in weights]) for i, k in enumerate(('w1', 'b1', 'w2', 'b2'))}
    np.savez_compressed(args.output/'ensemble.npz', **checkpoint, **stats,
                        bootstrap_episode_ids=boot_ids)
    np.savez_compressed(args.output/'episodes.npz', state_m=states, action_m=actions,
                        train_episode_ids=train_ids, validation_episode_ids=val_ids,
                        calibration_state_m=cs, calibration_action_m=ca,
                        timestamp_s=np.arange(21, dtype=np.float64)*.1)
    cpred = predict_members(cx, weights, stats)
    _, cscore = disagreement(cpred)
    # Fixed independent nominal score percentile; never fit on test labels or OOD.
    threshold = float(np.quantile(cscore, .95))
    sx = vx.copy()
    sx[:, 0] += np.float32(.8)
    ax = vx.copy()
    ax[:, 2:] *= np.float32(3)
    shifted_target = vtarget.copy()
    shifted_target[:, 0] += np.float32(args.shift_mm/1000)
    groups = {'id': (vx, vtarget),
              'state_shift': (sx, transition(sx[:, :2], sx[:, 2:])),
              'action_scale': (ax, transition(ax[:, :2], ax[:, 2:])),
              'hidden_mechanism_shift': (vx.copy(), shifted_target)}
    report = {'schema': 's3_3_model_uncertainty/1.0', 'members': args.members, 'epochs': args.epochs,
              'parameters_total': args.members*226, 'hidden_shift_mm': args.shift_mm,
              'seeds': {'data': 31, 'split': 32, 'calibration': 330,
                        'bootstrap': list(range(3300, 3300+args.members)),
                        'initialization': list(range(3400, 3400+args.members))},
              'normalized_bootstrap_train_losses': losses,
              'score_threshold_m': threshold, 'threshold_source': 'independent nominal calibration scores q95',
              'high_error_definition': 'ensemble mean next-state L2 > 0.005 m, diagnostic only',
              'calibration': summarize(cpred, ctarget, threshold)[0], 'evaluation': {}}
    arrays = {'calibration_input_m': cx, 'calibration_target_m': ctarget,
              'calibration_members_next_state_m': cpred}
    lines = ['group,index,error_l2_m,disagreement_m,flagged']
    for name, (x, y) in groups.items():
        pred = predict_members(x, weights, stats)
        summary, mean, score, error = summarize(pred, y, threshold)
        summary['baselines'] = {'persistence': metrics(x[:, :2], y),
                                'ideal_addition': metrics(x[:, :2]+x[:, 2:], y)}
        report['evaluation'][name] = summary
        for key, value in [('input_m', x), ('target_m', y), ('members_next_state_m', pred),
                           ('mean_next_state_m', mean), ('disagreement_m', score), ('error_l2_m', error)]:
            arrays[name+'_'+key] = value
        lines.extend(f'{name},{i},{e},{d},{int(d > threshold)}' for i, (e, d) in enumerate(zip(error, score)))
    # These paired probes must receive exactly identical input-dependent scores.
    np.testing.assert_array_equal(arrays['id_members_next_state_m'], arrays['hidden_mechanism_shift_members_next_state_m'])
    np.testing.assert_array_equal(arrays['id_disagreement_m'], arrays['hidden_mechanism_shift_disagreement_m'])
    # A copied member provides an explicit correlated-error negative control.
    clones = np.repeat(arrays['id_members_next_state_m'][:1], args.members, axis=0)
    report['cloned_member_control'] = summarize(clones, shifted_target, threshold)[0]
    for _ in range(20):
        disagreement(predict_members(vx[:1], weights, stats))
    times = []
    for _ in range(200):
        start = perf_counter_ns()
        disagreement(predict_members(vx[:1], weights, stats))
        times.append((perf_counter_ns()-start)/1e6)
    report['batch1_ensemble_latency_ms'] = {'p50': float(np.percentile(times, 50)),
                                          'p95': float(np.percentile(times, 95)), 'warmup': 20, 'samples': 200}
    report['limitations'] = 'Deterministic synthetic single-step scores; no calibrated risk, aleatoric estimate or robot success.'
    np.savez_compressed(args.output/'evaluation.npz', **arrays)
    (args.output/'samples.csv').write_text('\n'.join(lines)+'\n')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    contract = {'schema': 's3_3_ensemble_xy/1.0', 'nominal_dynamics': 'same as s3_1_xy_command_dynamics/1.0',
                'raw': 'float32 positions and commanded displacements in world XY [m], dt=.1s; 21 states/20 actions per episode',
                'training': '30 whole source episodes; 600 transitions; bootstrap 30 episodes/member with replacement',
                'statistics': 'shared x/delta mean/std fit original train600 only; no calibration/test/OOD fitting',
                'calibration': '10 independently generated nominal episodes (seed330); threshold q95 scores only',
                'evaluation': '10 held-out source episodes / 200 paired one-step tuples per condition',
                'ood': 'state x+.8m; action times3; same-input hidden mechanism next-x shift in mm',
                'ood_boundary': 'counterfactual one-step tuples, NOT continuous OOD trajectories',
                'hidden_shift': 'explicit changed transition mechanism; identical action semantics, no hidden state input',
                'input': 'float32 [N,4] current XY and command; train-only normalization',
                'output': 'float64 [K,N,2] next-state members, [N,2] mean, [N] disagreement [m]',
                'score': 'sqrt(mean over members and coordinates of centered prediction squared), ddof=0',
                'boundary': 'nominal episodes time-truncated after20 steps, no success labels; no policy/control',
                'modality': 'position/action only, no language/image/velocity/contact/noise model'}
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    print('threshold [mm]', threshold*1000)
    for name, values in report['evaluation'].items():
        print(name, 'RMSE/median disagreement [mm]', values['ensemble_mean']['rmse_m']*1000,
              values['disagreement_median_m']*1000, 'flags/high-error/confusion',
              values['flagged'], values['high_error_l2_over_5mm'], values['confusion'])
    print('cloned member median disagreement [m]', report['cloned_member_control']['disagreement_median_m'])


if __name__ == '__main__':
    main()
