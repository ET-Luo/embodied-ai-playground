"""S0.2: file-backed episodes -> grouped split -> causal padded sequences, CPU only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from s0_1_data_contract import require, toy_episode, validate


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def generate_episodes(directory):
    """Six independent reset groups, each with paired left/right episodes."""
    directory.mkdir()
    for group in range(6):
        for direction in (-1, 1):
            arrays, meta = toy_episode(direction)
            length = group + 1
            action = np.tile(np.array([direction*.01, 0], dtype=np.float32), (length, 1))
            start = np.array([[group*.1, 0]], dtype=np.float32)
            arrays.update(action=action,
                          proprioception=np.vstack([start, start+np.cumsum(action, axis=0)]),
                          timestamp=np.arange(length+1, dtype=np.float64)*.1,
                          terminated=np.zeros(length, dtype=bool), truncated=np.zeros(length, dtype=bool))
            arrays['truncated'][-1] = True  # Fixed horizon, not necessarily task success.
            name = f"group{group}_{'right' if direction == 1 else 'left'}"
            meta.update(episode_id=name, success=bool(direction*(arrays['proprioception'][-1, 0]-start[0, 0]) >= .03-1e-7),
                        success_definition='signed displacement from initial x >= 0.03 m; tolerance 1e-7 m')
            meta['provenance'].update(source='synthetic S0.2 paired reset', group_id=f'reset{group}')
            for key, x in arrays.items():
                meta['modalities'][key]['shape'] = list(x.shape)
            validate(arrays, meta)
            np.savez_compressed(directory/f'{name}.npz', **arrays)
            (directory/f'{name}.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')


def load_episodes(directory):
    """Validate raw file pairs before any window creation; retain terminal observation."""
    json_paths = sorted(directory.glob('*.json'))
    require(bool(json_paths), 'no episodes')
    require({p.stem for p in json_paths} == {p.stem for p in directory.glob('*.npz')}, 'unpaired episode files')
    episodes, seen = [], set()
    for path in json_paths:
        meta = json.loads(path.read_text(encoding='utf-8'))
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        validate(arrays, meta)
        require(meta['episode_id'] not in seen, 'duplicate episode id')
        seen.add(meta['episode_id'])
        # This lesson only windows state+language, never silently ignores other sensors.
        require(set(arrays) == {'timestamp', 'proprioception', 'action', 'terminated', 'truncated'},
                'S0.2 state-only reader: optional modalities require an explicit extension')
        episodes.append((arrays, meta))
    return episodes


def split_episodes(episodes, seed):
    """Split reset groups before windowing: related paired episodes stay together."""
    groups = sorted({m['provenance']['group_id'] for _, m in episodes})
    require(len(groups) >= 2 and 'unknown' not in groups, 'need at least two known independent groups')
    shuffled = np.random.default_rng(seed).permutation(groups)
    n_val = max(1, len(groups)//3)
    val_groups = set(shuffled[:n_val])
    return {name: [(a, m) for a, m in episodes if (m['provenance']['group_id'] in val_groups) == is_val]
            for name, is_val in [('train', False), ('validation', True)]}


def make_windows(episodes, history, horizon):
    """One anchor per executed action, causal left history and right action padding."""
    require(type(history) is int and history > 0 and type(horizon) is int and horizon > 0, 'positive L/H')
    require(bool(episodes), 'empty split')
    signature = None
    samples = []
    for arrays, meta in episodes:
        # Equal dimensions alone do not imply compatible embodiments or controllers.
        current = (meta['embodiment'], meta['control_dt'], meta['action_representation'],
                   meta['modalities']['proprioception']['fields'], meta['modalities']['action']['fields'])
        if signature is None:
            signature = current
        require(current == signature, 'incompatible episode semantics')
        t_total, a_dim = arrays['action'].shape
        p_dim = arrays['proprioception'].shape[1]
        for t in range(t_total):  # o_T is preserved in raw data but never a policy anchor.
            obs = np.zeros((history, p_dim), dtype=np.float32)
            action = np.zeros((horizon, a_dim), dtype=np.float32)
            obs_idx = np.full(history, -1, dtype=np.int64)
            action_idx = np.full(horizon, -1, dtype=np.int64)
            start = max(0, t-history+1)
            n_obs, n_action = t-start+1, min(horizon, t_total-t)
            obs[-n_obs:] = arrays['proprioception'][start:t+1]
            obs_idx[-n_obs:] = np.arange(start, t+1)
            action[:n_action] = arrays['action'][t:t+n_action]
            action_idx[:n_action] = np.arange(t, t+n_action)
            samples.append(dict(observation=obs, action=action, observation_mask=obs_idx >= 0,
                                action_mask=action_idx >= 0, observation_index=obs_idx, action_index=action_idx,
                                anchor=t, episode_id=meta['episode_id'], group_id=meta['provenance']['group_id'],
                                language_instruction=meta['language_instruction']))
    return {key: np.asarray([sample[key] for sample in samples]) for key in samples[0]}


def masked_mse(prediction, target, valid):
    """Mean squared error per valid action scalar; True means REAL, not padding."""
    require(prediction.shape == target.shape and target.ndim == 3, 'prediction/target (N,H,A)')
    require(valid.shape == target.shape[:2] and valid.dtype == np.bool_, 'mask (N,H) bool')
    require(bool(valid.any()), 'no valid action targets')
    difference = prediction[valid].astype(np.float64)-target[valid].astype(np.float64)
    require(np.isfinite(difference).all(), 'nonfinite valid prediction/target')
    return float(np.mean(difference**2))


def audit_windows(windows, episodes, history, horizon):
    """Independent source-index oracle for every anchor; catches cross-reset samples."""
    source = {meta['episode_id']: arrays for arrays, meta in episodes}
    require(len(windows['anchor']) == sum(len(a['action']) for a, _ in episodes), 'anchor count')
    for i, (episode_id, t) in enumerate(zip(windows['episode_id'], windows['anchor'])):
        raw = source[episode_id]
        for j in range(history):
            index = int(t)-history+1+j
            real = index >= 0
            require(bool(windows['observation_mask'][i, j]) == real, 'history mask')
            require(windows['observation_index'][i, j] == (index if real else -1), 'history index')
            expected = raw['proprioception'][index] if real else np.zeros(raw['proprioception'].shape[1])
            require(np.array_equal(windows['observation'][i, j], expected), 'history crosses boundary/future')
        for j in range(horizon):
            index = int(t)+j
            real = index < len(raw['action'])
            require(bool(windows['action_mask'][i, j]) == real, 'action mask')
            require(windows['action_index'][i, j] == (index if real else -1), 'action index')
            expected = raw['action'][index] if real else np.zeros(raw['action'].shape[1])
            require(np.array_equal(windows['action'][i, j], expected), 'action crosses boundary')


def experiment(episodes, splits, windows, history, horizon, seed):
    train_groups = {m['provenance']['group_id'] for _, m in splits['train']}
    val_groups = {m['provenance']['group_id'] for _, m in splits['validation']}
    require(not train_groups & val_groups, 'group leakage')
    require(not set(windows['train']['episode_id']) & set(windows['validation']['episode_id']), 'episode leakage')
    repeated = split_episodes(episodes, seed)
    require([m['episode_id'] for _, m in repeated['train']] == [m['episode_id'] for _, m in splits['train']], 'split repeatability')
    for name in windows:
        audit_windows(windows[name], splits[name], history, horizon)
    # Always test padding, including when learner chooses L=H=1.
    probe = make_windows(episodes[:1], 4, 4)
    audit_windows(probe, episodes[:1], 4, 4)
    corrupt = {key: x.copy() for key, x in probe.items()}
    corrupt['action'][0, -1] = episodes[-1][0]['action'][0]  # bad naive concatenation across reset
    try:
        audit_windows(corrupt, episodes[:1], 4, 4)
    except ValueError:
        pass
    else:
        raise RuntimeError('cross-boundary corruption accepted')
    target, mask = probe['action'], probe['action_mask']
    predicted = target.copy()
    predicted[mask] += 1
    predicted[~mask] = 1000  # padding must not dominate loss
    masked = masked_mse(predicted, target, mask)
    unmasked = float(np.mean((predicted.astype(np.float64)-target)**2))
    require(np.isclose(masked, 1), 'masked loss oracle')
    predicted[~mask] = np.nan
    require(masked_mse(predicted, target, mask) == masked, 'padded values affect loss')
    try:
        masked_mse(predicted, target, np.zeros_like(mask))
    except ValueError:
        pass
    else:
        raise RuntimeError('empty loss mask accepted')
    # Bad timestep split: adjacent windows from every T>=2 episode land on both sides.
    all_windows = make_windows(episodes, history, horizon)
    # Alternate anchors WITHIN each episode: an intentionally invalid split.
    row_train = set(all_windows['episode_id'][all_windows['anchor'] % 2 == 0])
    row_val = set(all_windows['episode_id'][all_windows['anchor'] % 2 == 1])
    leaked_ids = sorted(row_train & row_val)
    require(len(leaked_ids) == 10, 'bad row split overlap oracle')
    changed_terminal = copy.deepcopy(episodes[:1])
    changed_terminal[0][0]['proprioception'][-1] += 999
    require(np.array_equal(make_windows(changed_terminal, 4, 4)['observation'], probe['observation']),
            'terminal observation leaked into policy input')
    # Held-out sentinel cannot alter statistics computed only on train transition starts.
    train_mean = np.concatenate([a['proprioception'][:-1] for a, _ in splits['train']]).mean(0)
    shifted = copy.deepcopy(splits)
    for a, _ in shifted['validation']:
        a['proprioception'][:, 0] += 100
    train_after = np.concatenate([a['proprioception'][:-1] for a, _ in shifted['train']]).mean(0)
    require(np.array_equal(train_mean, train_after), 'train statistics contaminated')
    all_before = np.concatenate([a['proprioception'][:-1] for name in splits for a, _ in splits[name]]).mean(0)
    all_after = np.concatenate([a['proprioception'][:-1] for name in shifted for a, _ in shifted[name]]).mean(0)
    report = dict(masked_mse=masked, unmasked_mse=unmasked, bad_timestep_split_shared_episodes=len(leaked_ids),
                  correct_split_shared_episodes=0, correct_split_shared_groups=0,
                  train_mean_before=train_mean.tolist(), train_mean_after_val_shift=train_after.tolist(),
                  wrong_all_data_mean_shift=(all_after-all_before).tolist())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=positive_int, default=3)
    parser.add_argument('--horizon', type=positive_int, default=4)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('tmp/s0_2'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    generate_episodes(args.output/'episodes')
    episodes = load_episodes(args.output/'episodes')
    splits = split_episodes(episodes, args.seed)
    windows = {name: make_windows(items, args.history, args.horizon) for name, items in splits.items()}
    report = experiment(episodes, splits, windows, args.history, args.horizon, args.seed)
    report.update(history=args.history, horizon=args.horizon, seed=args.seed, raw_normalized=False, splits={})
    for name, batch in windows.items():
        np.savez_compressed(args.output/f'{name}_windows.npz', **batch)
        with np.load(args.output/f'{name}_windows.npz', allow_pickle=False) as saved:
            require(all(np.array_equal(saved[key], batch[key]) for key in batch), 'window round-trip')
        report['splits'][name] = dict(episode_ids=[m['episode_id'] for _, m in splits[name]],
            group_ids=sorted({m['provenance']['group_id'] for _, m in splits[name]}),
            windows=len(batch['anchor']), observation_shape=list(batch['observation'].shape),
            action_shape=list(batch['action'].shape), valid_history=int(batch['observation_mask'].sum()),
            valid_actions=int(batch['action_mask'].sum()))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'PASS raw reload / grouped split / all-window alignment / padding / masked loss / leakage counterexamples / round-trip; output={args.output}')


if __name__ == '__main__':
    main()
