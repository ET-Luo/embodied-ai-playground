"""S0.3: tiny synthetic loaded-RLDS fixture -> local files -> S0.2 windows."""
import argparse
import copy
import json
from pathlib import Path
import sys

# Direct repository imports; no sibling repo or TensorFlow dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from data.adapters.rlds_toy import PROFILE, convert_episode
from scripts.s0_1_data_contract import require, toy_episode, validate
from s0_2_sequence_dataset import audit_windows, make_windows


def fixture(ending):
    arrays, meta = toy_episode(1)
    steps = []
    for i in range(4):
        steps.append(dict(observation=dict(state=arrays['proprioception'][i].tolist(),
                natural_language_instruction=meta['language_instruction']),
            action=arrays['action'][i].tolist() if i < 3 else [999., 999.],
            timestamp=float(arrays['timestamp'][i]), is_first=i == 0, is_last=i == 3,
            is_terminal=i == 3 and ending == 'terminal', reward=0. if i < 3 else 999., discount=1.))
    return dict(profile=PROFILE, invalid=False, contract_metadata=meta, steps=steps)


def rejection_checks(source):
    cases = []
    for name in ('invalid_episode', 'missing_last_observation', 'early_last', 'early_terminal',
                 'language_change', 'bad_clock', 'unknown_units', 'wrong_profile', 'missing_success', 'bad_real_action'):
        bad = copy.deepcopy(source)
        if name == 'invalid_episode': bad['invalid'] = True
        if name == 'missing_last_observation': bad['steps'].pop()
        if name == 'early_last': bad['steps'][1]['is_last'] = True
        if name == 'early_terminal': bad['steps'][1]['is_terminal'] = True
        if name == 'language_change': bad['steps'][1]['observation']['natural_language_instruction'] = 'move left'
        if name == 'bad_clock': bad['steps'][1]['timestamp'] = .9
        if name == 'unknown_units': bad['contract_metadata']['modalities']['action']['fields'][0]['unit'] = 'unknown'
        if name == 'wrong_profile': bad['profile'] = 'real_oxe_dataset'
        if name == 'missing_success': del bad['contract_metadata']['success']
        if name == 'bad_real_action': bad['steps'][0]['action'] = [float('nan'), 0.]
        cases.append((name, bad))
    for name, bad in cases:
        try:
            convert_episode(bad)
        except ValueError:
            print(f'REJECT {name}')
        else:
            raise RuntimeError(f'bad source accepted: {name}')
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ending', choices=('terminal', 'truncated'), default='terminal')
    parser.add_argument('--output', type=Path, default=Path('tmp/s0_3'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = fixture(args.ending)
    source_path = args.output/'synthetic_loaded_rlds.json'
    source_path.write_text(json.dumps(source, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    arrays, meta = convert_episode(json.loads(source_path.read_text(encoding='utf-8')))
    expected, _ = toy_episode(1)
    for key in ('timestamp', 'proprioception', 'action'):
        require(np.array_equal(arrays[key], expected[key]), f'{key} alignment oracle')
    require(arrays['action'].shape == (3, 2) and meta['success'] is True, 'T and independent success')
    altered = copy.deepcopy(source)
    altered['steps'][-1]['action'] = [float('nan')]  # invalid terminal payload must never become a label
    changed, _ = convert_episode(altered)
    require(np.array_equal(changed['action'], arrays['action']), 'last invalid payload affected labels')
    for ending in ('terminal', 'truncated'):
        check, info = convert_episode(fixture(ending))
        require(bool(check['terminated'][-1]) == (ending == 'terminal'), 'termination mapping')
        require(bool(check['truncated'][-1]) == (ending == 'truncated'), 'truncation mapping')
        require(info['success'] is True, 'success must not be inferred from ending')
    unsuccessful = copy.deepcopy(source)
    unsuccessful['contract_metadata']['success'] = False
    _, unsuccessful_meta = convert_episode(unsuccessful)
    require(unsuccessful_meta['success'] is False, 'termination incorrectly implies success')
    count = rejection_checks(source)
    np.savez_compressed(args.output/'episode.npz', **arrays)
    (args.output/'episode.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    with np.load(args.output/'episode.npz', allow_pickle=False) as saved:
        loaded = {key: saved[key] for key in saved.files}
    loaded_meta = json.loads((args.output/'episode.json').read_text(encoding='utf-8'))
    validate(loaded, loaded_meta)
    require(loaded_meta == meta, 'metadata round-trip')
    for key in arrays:
        require(np.array_equal(arrays[key], loaded[key]), f'{key} round-trip')
    windows = make_windows([(loaded, loaded_meta)], 3, 4)
    audit_windows(windows, [(loaded, loaded_meta)], 3, 4)
    report = dict(source='synthetic only; no real robot sample', source_steps=4, local_transitions=3,
                  observation_shape=list(arrays['proprioception'].shape), action_shape=list(arrays['action'].shape),
                  last_source_action_ignored=source['steps'][-1]['action'],
                  terminated=arrays['terminated'].tolist(), truncated=arrays['truncated'].tolist(),
                  success=meta['success'], ending=args.ending, rejected_cases=count,
                  s0_2_windows=len(windows['anchor']), valid_action_tokens=int(windows['action_mask'].sum()),
                  numeric_payload_bytes=sum(x.nbytes for x in arrays.values()))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('PASS alignment / invalid last action ignored / terminal-vs-truncated / independent success / rejection / round-trip / S0.2 windows')


if __name__ == '__main__':
    main()
