"""Explicit loaded-RLDS concept profile; NOT a TFDS reader or general OXE adapter."""
import copy

import numpy as np

from scripts.s0_1_data_contract import require, validate

PROFILE = 'synthetic_loaded_rlds_xy/1.0'


def convert_episode(source):
    """Convert only the documented synthetic XY profile; no inferred robot semantics."""
    require(source.get('profile') == PROFILE, 'unsupported dataset-specific profile')
    require(type(source.get('invalid')) is bool and not source['invalid'], 'invalid/incomplete episode')
    steps = source.get('steps')
    require(isinstance(steps, list) and len(steps) >= 2, 'need initial and final observations')
    meta = copy.deepcopy(source['contract_metadata'])
    require(meta['embodiment'] == 'ideal_world_xy_point_v1', 'unsupported embodiment')
    expected_fields = [dict(name=axis, unit='m', frame='world') for axis in ('x', 'y')]
    for name in ('action', 'proprioception'):
        require(meta['modalities'][name]['fields'] == expected_fields, 'XY world meters required')
    require(meta['action_representation'] == dict(type='cartesian_delta', anchor='measured position at t',
            execution='ideal displacement applied once per control interval', normalized=False), 'unverified action semantics')
    require(meta['control_dt'] == .1, 'profile control_dt must be .1 s')
    require(set(steps[0]) == {'observation', 'action', 'timestamp', 'is_first', 'is_last',
                            'is_terminal', 'reward', 'discount'}, 'unsupported step fields')
    count = len(steps)
    for i, step in enumerate(steps):
        require(set(step) == set(steps[0]), 'inconsistent step fields')
        for key in ('is_first', 'is_last', 'is_terminal'):
            require(type(step[key]) is bool, f'{key} must be bool')
        require(step['is_first'] == (i == 0), 'initial flag alignment')
        require(step['is_last'] == (i == count-1), 'last observation missing or multiple boundaries')
        require(not step['is_terminal'] or i == count-1, 'early terminal')
        observation = step['observation']
        require(set(observation) == {'state', 'natural_language_instruction'}, 'unsupported observation fields')
        require(observation['natural_language_instruction'] == meta['language_instruction'], 'language changed/missing')
        state = np.asarray(observation['state'], dtype=np.float32)
        require(state.shape == (2,) and np.isfinite(state).all(), 'invalid state observation')
        require(type(step['timestamp']) in (int, float), 'timestamp must be numeric')
        if i < count-1:
            action = np.asarray(step['action'], dtype=np.float32)
            require(action.shape == (2,) and np.isfinite(action).all(), 'invalid real action')
    # Critical alignment: S_i has o_i and outgoing a_i. Last action is invalid even if nonzero.
    terminal = steps[-1]['is_terminal']
    arrays = dict(timestamp=np.asarray([s['timestamp'] for s in steps], dtype=np.float64),
                  proprioception=np.asarray([s['observation']['state'] for s in steps], dtype=np.float32),
                  action=np.asarray([s['action'] for s in steps[:-1]], dtype=np.float32),
                  terminated=np.zeros(count-1, dtype=bool), truncated=np.zeros(count-1, dtype=bool))
    arrays['terminated'][-1] = terminal
    arrays['truncated'][-1] = not terminal
    require(set(meta['modalities']) == set(arrays), 'profile only supports state/action/clock/boundaries')
    for key, x in arrays.items():
        meta['modalities'][key]['shape'] = list(x.shape)
    meta['provenance'].update(source='synthetic loaded-RLDS concept fixture', source_schema=PROFILE)
    meta['adapter'] = dict(profile=PROFILE, alignment='loaded RLDS: o_i -> a_i -> o_(i+1)',
                          discarded='last action; all reward/discount (not represented by embodied_episode/1.0)',
                          timestamp_origin='explicit synthetic relative clock; not guaranteed by RLDS',
                          success_origin='explicit source metadata; never inferred from boundary/reward')
    validate(arrays, meta)
    return arrays, meta
