"""S0.1: explicit episode contract, toy generation and CPU validation (no policy)."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

SCHEMA = 'embodied_episode/1.0'
REQUIRED = {'timestamp', 'proprioception', 'action', 'terminated', 'truncated'}
OPTIONAL = {'next_state', 'image', 'force', 'tactile'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(arrays, meta):
    """Validate structure/declared semantics; cannot prove real sensor causality."""
    require(meta.get('schema_version') == SCHEMA, 'unsupported schema_version')
    for key in ('episode_id', 'task_name', 'embodiment', 'language_instruction', 'success_definition'):
        require(isinstance(meta.get(key), str) and bool(meta[key].strip()), f'missing {key}')
    require(type(meta.get('success')) is bool, 'success must be bool')
    provenance = meta.get('provenance', {})
    for key in ('source', 'source_schema', 'source_commit', 'controller', 'sampling', 'group_id'):
        require(isinstance(provenance.get(key), str) and bool(provenance[key]), f'provenance.{key}')
    dt = meta.get('control_dt')
    require(type(dt) in (float, int) and np.isfinite(dt) and dt > 0, 'control_dt > 0')
    require(REQUIRED <= arrays.keys() <= REQUIRED | OPTIONAL, 'array keys')
    a = arrays['action']
    require(a.ndim == 2 and min(a.shape) > 0, 'action must be nonempty (T,A)')
    t = a.shape[0]
    p = arrays['proprioception']
    require(p.ndim == 2 and p.shape[0] == t + 1 and p.shape[1] > 0, 'T actions need T+1 observations')
    expected = {'timestamp': ((t+1,), 'float64'), 'proprioception': (p.shape, 'float32'),
                'action': (a.shape, 'float32'), 'terminated': ((t,), 'bool'),
                'truncated': ((t,), 'bool')}
    for key in OPTIONAL & arrays.keys():
        x = arrays[key]
        if key == 'image':
            require(x.ndim == 4 and x.shape[0] == t+1 and x.shape[-1] == 3
                    and min(x.shape[1:3]) > 0, 'image (T+1,H,W,3)')
            expected[key] = (x.shape, 'uint8')
        elif key == 'next_state':
            expected[key] = ((t, p.shape[1]), 'float32')
        else:
            require(x.ndim == 2 and x.shape[0] == t+1 and x.shape[1] > 0, f'{key} synchronized shape')
            expected[key] = (x.shape, 'float32')
    declarations = meta.get('modalities', {})
    require(declarations.keys() == arrays.keys(), 'modalities must match NPZ keys')
    for key, (shape, dtype) in expected.items():
        x, desc = arrays[key], declarations[key]
        require(x.shape == shape and x.dtype == np.dtype(dtype), f'{key} shape/dtype')
        require(np.isfinite(x).all(), f'{key} nonfinite')
        require(desc.get('shape') == list(shape) and desc.get('dtype') == dtype, f'{key} declaration')
        for field in ('semantic', 'temporal'):
            require(isinstance(desc.get(field), str) and bool(desc[field]), f'{key}.{field}')
        if x.ndim == 2:
            fields = desc.get('fields', [])
            require(len(fields) == shape[1], f'{key} channel count')
            for channel in fields:
                for field in ('name', 'unit', 'frame'):
                    require(isinstance(channel.get(field), str) and bool(channel[field]), f'{key}.{field}')
        elif key == 'image':
            require(desc.get('color_space') == 'RGB' and bool(desc.get('frame'))
                    and bool(desc.get('calibration')) and bool(desc.get('source')), 'image metadata')
        else:
            require(desc.get('unit') == ('s' if key == 'timestamp' else '1'), f'{key} unit')
    require(np.allclose(arrays['timestamp'], np.arange(t+1)*dt, atol=1e-9, rtol=1e-7), 'clock alignment')
    end, limit = arrays['terminated'], arrays['truncated']
    require(not (end[:-1] | limit[:-1]).any() and bool(end[-1]) != bool(limit[-1]), 'episode boundary')
    rep = meta.get('action_representation', {})
    require(rep.get('type') in ('joint_absolute', 'joint_delta', 'cartesian_absolute', 'cartesian_delta'), 'action type')
    require(rep.get('normalized') is False, 'raw action must not be normalized')
    for field in ('anchor', 'execution'):
        require(isinstance(rep.get(field), str) and bool(rep[field]), f'action {field}')
    if 'next_state' in arrays:
        require(np.array_equal(arrays['next_state'], p[1:]), 'next_state must equal observation successors')


def toy_episode(direction):
    """An ideal point changes measured world position once per interval."""
    actions = np.tile(np.array([direction*.01, 0], dtype=np.float32), (3, 1))
    positions = np.vstack([np.zeros((1, 2), dtype=np.float32), np.cumsum(actions, axis=0)])
    arrays = dict(timestamp=np.arange(4, dtype=np.float64)*.1, proprioception=positions,
                  action=actions, terminated=np.array([False, False, True]),
                  truncated=np.zeros(3, dtype=bool))
    fields = [dict(name=axis, unit='m', frame='world') for axis in ('x', 'y')]
    desc = {}
    for key, x in arrays.items():
        desc[key] = dict(shape=list(x.shape), dtype=str(x.dtype),
                         semantic={'timestamp': 'episode-relative clock', 'proprioception': 'measured point position',
                                   'action': 'executed displacement', 'terminated': 'natural task end',
                                   'truncated': 'time limit'}[key],
                         temporal='observation at t, including terminal T' if len(x) == 4 else 'transition t -> t+1')
        if x.ndim == 2:
            desc[key]['fields'] = copy.deepcopy(fields)
        else:
            desc[key]['unit'] = 's' if key == 'timestamp' else '1'
    side = 'right' if direction == 1 else 'left'
    meta = dict(schema_version=SCHEMA, episode_id=f'toy_{side}_001', task_name='point_displacement',
                embodiment='ideal_world_xy_point_v1', language_instruction=f'move {side}',
                success=bool(direction*positions[-1, 0] >= .03 - 1e-7), control_dt=.1,
                success_definition='signed final x >= 0.03 m (tolerance 1e-7 m)',
                action_representation=dict(type='cartesian_delta', anchor='measured position at t',
                                           execution='ideal displacement applied once per control interval', normalized=False),
                modalities=desc, provenance=dict(source='synthetic S0.1', source_schema=SCHEMA,
                    source_commit='not applicable: standalone toy generator', controller='ideal displacement',
                    sampling='synchronous fixed 10Hz', group_id='paired_initial_state_001'))
    return arrays, meta


def rejection_checks(arrays, meta):
    cases = []
    for name in ('missing_terminal', 'bad_clock', 'bad_dtype', 'early_boundary', 'nonfinite'):
        bad = {key: x.copy() for key, x in arrays.items()}
        if name == 'missing_terminal': bad['proprioception'] = bad['proprioception'][:-1]
        if name == 'bad_clock': bad['timestamp'][1] += .01
        if name == 'bad_dtype': bad['action'] = bad['action'].astype(np.float64)
        if name == 'early_boundary': bad['terminated'][0] = True
        if name == 'nonfinite': bad['action'][0, 0] = np.nan
        cases.append((name, bad, meta))
    for name in ('unknown_schema', 'normalized_raw', 'missing_unit'):
        bad = copy.deepcopy(meta)
        if name == 'unknown_schema': bad['schema_version'] = 'embodied_episode/2.0'
        if name == 'normalized_raw': bad['action_representation']['normalized'] = True
        if name == 'missing_unit': del bad['modalities']['action']['fields'][0]['unit']
        cases.append((name, arrays, bad))
    for name, bad_arrays, bad_meta in cases:
        try:
            validate(bad_arrays, bad_meta)
        except ValueError:
            print(f'REJECT {name}')
        else:
            raise RuntimeError(f'invalid case accepted: {name}')
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/s0_1'))
    parser.add_argument('--direction', choices=('right', 'left'), default='right')
    args = parser.parse_args()
    arrays, meta = toy_episode(1 if args.direction == 'right' else -1)
    validate(arrays, meta)
    # Real saved-data round-trip. Refuse overwriting a previous run.
    args.output.mkdir(parents=True, exist_ok=False)
    stem = args.output / meta['episode_id']
    np.savez_compressed(stem.with_suffix('.npz'), **arrays)
    stem.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
        loaded = {key: saved[key] for key in saved.files}
    loaded_meta = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
    validate(loaded, loaded_meta)
    require(loaded_meta == meta, 'JSON round-trip')
    for key in arrays:
        require(np.array_equal(loaded[key], arrays[key]), f'{key} round-trip')
    # Toy-only dynamics audit; this identity is NOT assumed for real robot servos.
    np.testing.assert_allclose(loaded['proprioception'][1:],
                               loaded['proprioception'][:-1]+loaded['action'], atol=1e-8, rtol=0)
    left, _ = toy_episode(-1)
    right, _ = toy_episode(1)
    require(np.array_equal(left['proprioception'][0], right['proprioception'][0]), 'paired start')
    require(not np.array_equal(left['action'][0], right['action'][0]), 'different language labels')
    count = rejection_checks(loaded, loaded_meta)
    print(f"language={meta['language_instruction']}; o_0={loaded['proprioception'][0].tolist()}; a_0={loaded['action'][0].tolist()}")
    print(f"T=3; observation=(4,2) float32; action=(3,2) float32; terminal={loaded['proprioception'][-1].tolist()}")
    print(f'PASS round-trip / toy alignment / paired language labels / {count} rejection checks; output={args.output}')


if __name__ == '__main__':
    main()
