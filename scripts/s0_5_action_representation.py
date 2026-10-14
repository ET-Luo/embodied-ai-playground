"""S0.5: explicit continuous action views of a synthetic two-joint servo trace."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

from s0_1_data_contract import SCHEMA, require, validate
from s0_2_sequence_dataset import audit_windows, make_windows

LINKS = np.array([.3, .2], dtype=np.float64)
REFERENCES = np.array([[.2, .3], [.4, .5], [.3, .6], [.5, .2]], dtype=np.float32)


def fk(q, lengths=LINKS):
    """Planar XY end point in fixed base/world frame; no orientation channel."""
    return np.stack([lengths[0]*np.cos(q[..., 0]) + lengths[1]*np.cos(q.sum(axis=-1)),
                     lengths[0]*np.sin(q[..., 0]) + lengths[1]*np.sin(q.sum(axis=-1))], axis=-1)


def servo(initial, references, gain):
    require(np.isfinite(gain) and 0 < gain <= 1, 'gain in (0,1]')
    require(references.ndim == 2 and references.shape[1] == 2 and len(references) > 0, 'references (T,2)')
    require(np.isfinite(references).all() and np.max(np.abs(references)) < np.pi, 'finite references within toy joint limits')
    q = np.empty((len(references)+1, 2), dtype=np.float32)
    q[0] = initial
    for t, reference in enumerate(references):
        q[t+1] = q[t] + gain * (reference-q[t])
    return q


def episode(index, offset, split, gain):
    reference = REFERENCES + np.asarray(offset, dtype=np.float32)
    initial = np.array([.1, .4], dtype=np.float32) + offset
    q = servo(initial, reference, gain)
    arrays = dict(timestamp=np.arange(5, dtype=np.float64)*.1, proprioception=q, action=reference,
                  terminated=np.zeros(4, dtype=bool), truncated=np.array([False, False, False, True]))
    descriptions = {}
    for key, value in arrays.items():
        descriptions[key] = dict(shape=list(value.shape), dtype=str(value.dtype),
            semantic={'timestamp': 'relative synthetic clock', 'proprioception': 'measured joint positions',
                      'action': 'joint position references sent to toy servo', 'terminated': 'natural boundary',
                      'truncated': 'four-command budget boundary'}[key],
            temporal='observation at t, including final T' if key in ('timestamp', 'proprioception') else 'transition t -> t+1')
        if value.ndim == 2:
            descriptions[key]['fields'] = [dict(name=name, unit='rad', frame='joint axis') for name in ('q1', 'q2')]
        else:
            descriptions[key]['unit'] = 's' if key == 'timestamp' else '1'
    meta = dict(schema_version=SCHEMA, episode_id=f'servo_{index}', split=split,
        task_name='four_reference_tracking', embodiment='synthetic_planar_2r_links_0.3_0.2_m',
        language_instruction='track the four joint references',
        success=bool(np.max(np.abs(q[-1]-reference[-1])) <= .02),
        success_definition='toy max final joint tracking error <= 0.02 rad; independent of budget boundary',
        control_dt=.1, action_representation=dict(type='joint_absolute', anchor='joint coordinate zero',
            execution=f'one discrete servo update q_next=q+{gain}*(reference-q) per interval', normalized=False),
        modalities=descriptions, provenance=dict(source='synthetic S0.5', source_schema=SCHEMA,
            source_commit='not applicable: explicit toy generator', controller=f'discrete proportional servo gain={gain}',
            sampling='synthetic synchronous 10Hz', group_id=f'independent_reset_{index}'))
    validate(arrays, meta)
    return arrays, meta


def views(arrays, meta):
    q, r = arrays['proprioception'], arrays['action']
    # r_-1=q_0 is an explicit episode initialization, not a generic assumption.
    previous = np.vstack([q[0], r[:-1]])
    p = fk(q.astype(np.float64)).astype(np.float32)
    target = fk(r.astype(np.float64)).astype(np.float32)
    values = dict(joint_absolute=r.copy(), joint_delta_measured=r-q[:-1],
                  joint_delta_previous_reference=r-previous, cartesian_absolute=target,
                  cartesian_delta_measured=target-p[:-1])
    signatures = {}
    for name in values:
        cartesian = name.startswith('cartesian')
        anchor = {'joint_absolute': 'joint coordinate zero', 'joint_delta_measured': 'measured q_t',
                  'joint_delta_previous_reference': 'previous command r_(t-1), initial r_-1=q_0',
                  'cartesian_absolute': 'fixed world origin', 'cartesian_delta_measured': 'measured FK(q_t)'}[name]
        signatures[name] = dict(representation=name, anchor=anchor,
            unit='m' if cartesian else 'rad', frame='fixed base = world' if cartesian else 'ordered joint axes',
            fields=['x', 'y'] if cartesian else ['q1', 'q2'], normalized=False,
            embodiment=meta['embodiment'], links_m=LINKS.tolist(), dt_s=meta['control_dt'],
            controller=meta['action_representation']['execution'],
            execution='derived XY target only; no IK/controller conversion supplied' if cartesian else 'decode reference, then use same toy servo')
    return values, signatures, p, previous


def fit_stats(train, name):
    require(bool(train) and all(meta['split'] == 'train' for _, meta in train), 'normalizer accepts training episodes only')
    values, signature = [], None
    for arrays, meta in train:
        current, descriptions, _, _ = views(arrays, meta)
        if signature is None:
            signature = descriptions[name]
        require(signature == descriptions[name], 'normalizer source semantics mismatch')
        values.append(current[name])
    x = np.concatenate(values).astype(np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    # For an exactly constant channel, use one source unit; keep the real std=0.
    scale = np.where(std == 0, 1., std)
    return dict(schema_version='action_zscore/1.0', signature=signature, mean=mean.tolist(),
                std=std.tolist(), scale=scale.tolist(), constant_channels=(std == 0).tolist(),
                ddof=0, fit_episode_ids=[meta['episode_id'] for _, meta in train], fit_rows=len(x),
                rule='training raw actions once each; no validation, padding or overlapping chunk copies; no clipping')


def transform(x, stats, signature, inverse=False):
    require(stats['signature'] == signature, 'normalizer target semantics mismatch')
    require(x.shape[-1] == len(stats['mean']) and np.isfinite(x).all(), 'normalizer input shape/finite')
    mean, scale = np.asarray(stats['mean']), np.asarray(stats['scale'])
    require(np.isfinite(scale).all() and (scale > 0).all(), 'normalizer scale')
    return ((x*scale+mean) if inverse else ((x-mean)/scale)).astype(np.float32)


def round_trip(directory, arrays, meta):
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory/'episode.npz', **arrays)
    (directory/'episode.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    with np.load(directory/'episode.npz', allow_pickle=False) as saved:
        loaded = {key: saved[key] for key in saved.files}
    loaded_meta = json.loads((directory/'episode.json').read_text(encoding='utf-8'))
    validate(loaded, loaded_meta)
    require(loaded_meta == meta, 'metadata round-trip')
    require(all(np.array_equal(arrays[key], loaded[key]) for key in arrays), 'raw round-trip')
    return loaded, loaded_meta


def expect_rejection(name, function):
    try:
        function()
    except ValueError:
        return name
    raise RuntimeError(f'invalid input accepted: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gain', type=float, default=.5)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('tmp/s0_5'))
    args = parser.parse_args()
    require(np.isfinite(args.gain) and 0 < args.gain <= 1, 'gain in (0,1]')
    require(args.horizon > 0, 'positive horizon')
    require(not args.output.exists(), 'choose a new output directory')
    episodes = [episode(i, np.array(offset, dtype=np.float32), split, args.gain)
                for i, offset, split in [(0, [0, 0], 'train'), (1, [.1, 0], 'train'),
                                        (2, [-.1, .05], 'train'), (3, [.8, -.1], 'validation')]]
    args.output.mkdir(parents=True, exist_ok=False)
    episodes = [round_trip(args.output/'episodes'/meta['episode_id'], arrays, meta) for arrays, meta in episodes]
    train, validation = episodes[:3], episodes[3:]
    require(not {m['provenance']['group_id'] for _, m in train} & {m['provenance']['group_id'] for _, m in validation}, 'reset group overlap')
    raw, meta = train[0]
    value, signature, position, previous = views(raw, meta)
    stats = {name: fit_stats(train, name) for name in value}
    normalized = {}
    for name, x in value.items():
        normalized[name] = transform(x, stats[name], signature[name])
        np.testing.assert_allclose(transform(normalized[name], stats[name], signature[name], inverse=True), x, atol=1e-7, rtol=0)
    r, q = raw['action'], raw['proprioception']
    np.testing.assert_allclose(value['joint_delta_measured']+q[:-1], r, atol=1e-7, rtol=0)
    np.testing.assert_allclose(np.cumsum(value['joint_delta_previous_reference'], axis=0)+q[0], r, atol=1e-7, rtol=0)
    np.testing.assert_allclose(value['cartesian_delta_measured']+position[:-1], value['cartesian_absolute'], atol=1e-7, rtol=0)
    np.testing.assert_allclose(np.diff(q, axis=0), args.gain*value['joint_delta_measured'], atol=1e-7, rtol=0)
    # Execute both delta interpretations with their correct anchors.
    measured_replay = np.empty_like(q); measured_replay[0] = q[0]
    previous_references = np.cumsum(value['joint_delta_previous_reference'], axis=0)+q[0]
    for t, delta in enumerate(value['joint_delta_measured']):
        decoded_reference = measured_replay[t]+delta
        measured_replay[t+1] = measured_replay[t]+args.gain*(decoded_reference-measured_replay[t])
    np.testing.assert_allclose(measured_replay, q, atol=1e-7, rtol=0)
    np.testing.assert_allclose(servo(q[0], previous_references, args.gain), q, atol=1e-7, rtol=0)
    wrong_replay = servo(q[0], value['joint_delta_measured'], args.gain)
    # Link lengths, frames and ordering change meaning even with equal dimensions.
    rotation = np.array([[0., -1.], [1., 0.]])  # fixed local axes at +90 degrees; not the moving end-effector frame
    world_delta = value['cartesian_delta_measured'].astype(np.float64)
    local_delta = world_delta @ rotation  # column convention: delta_world = R delta_local
    np.testing.assert_allclose(local_delta @ rotation.T, world_delta, atol=1e-12)
    other_position = fk(r.astype(np.float64), np.array([.2, .3]))
    require(np.max(np.abs(other_position-value['cartesian_absolute'])) > .001, 'link length mismatch probe')
    # Hold out synthetic episode 3 before fitting. Perturbing it cannot affect train stats.
    altered_val = copy.deepcopy(validation)
    altered_val[0][0]['action'][:, 0] += .2
    for name in stats:
        require(fit_stats(train, name) == stats[name], 'train stats changed')
    train_mean = np.array(stats['joint_absolute']['mean'])
    wrong_all_mean = np.concatenate([a['action'] for a, _ in episodes]).mean(axis=0, dtype=np.float64)
    wrong_altered_mean = np.concatenate([a['action'] for a, _ in train+altered_val]).mean(axis=0, dtype=np.float64)
    require(not np.array_equal(wrong_all_mean, wrong_altered_mean), 'leakage stress probe ineffective')
    val_views, val_signature, _, _ = views(*validation[0])
    val_normalized = transform(val_views['joint_absolute'], stats['joint_absolute'], val_signature['joint_absolute'])
    # Reuse S0.2 causal windows; normalize REAL slots only and preserve padding zeros.
    windows = make_windows(train, 1, args.horizon)
    audit_windows(windows, train, 1, args.horizon)
    mask = windows['action_mask']
    normalized_chunk = np.zeros_like(windows['action'])
    normalized_chunk[mask] = transform(windows['action'][mask], stats['joint_absolute'], signature['joint_absolute'])
    require(not normalized_chunk[~mask].any(), 'padding must stay masked zero')
    recovered = transform(normalized_chunk[mask], stats['joint_absolute'], signature['joint_absolute'], inverse=True)
    np.testing.assert_allclose(recovered, windows['action'][mask], atol=1e-7, rtol=0)
    # A future measured-anchor delta needs the FUTURE measured anchor; frozen q_t is wrong.
    stale_reference = q[0]+value['joint_delta_measured']
    stale_error = np.max(np.abs(stale_reference-r), axis=0)
    require(stale_error.max() > .01, 'stale anchor error probe')
    rejected = [expect_rejection('validation_in_fit', lambda: fit_stats(train+validation, 'joint_absolute')),
                expect_rejection('wrong_normalizer_representation', lambda: transform(value['joint_delta_measured'], stats['joint_absolute'], signature['joint_delta_measured'])),
                expect_rejection('nonfinite_normalized_input', lambda: transform(np.full((1,2), np.nan), stats['joint_absolute'], signature['joint_absolute']))]
    normalized_meta = copy.deepcopy(meta)
    normalized_meta['action_representation']['normalized'] = True
    rejected.append(expect_rejection('normalized_raw_episode', lambda: validate(raw, normalized_meta)))
    bad_scale = copy.deepcopy(stats['joint_absolute']); bad_scale['scale'][0] = 0
    rejected.append(expect_rejection('zero_normalizer_scale', lambda: transform(r, bad_scale, signature['joint_absolute'])))
    for name in ('embodiment', 'channel_order', 'unit', 'frame', 'controller', 'anchor'):
        incompatible = copy.deepcopy(train[1])
        bad = incompatible[1]
        if name == 'embodiment': bad['embodiment'] = 'different_link_lengths'
        if name == 'channel_order': bad['modalities']['action']['fields'].reverse()
        if name == 'unit': bad['modalities']['action']['fields'][0]['unit'] = 'degree'
        if name == 'frame': bad['modalities']['action']['fields'][0]['frame'] = 'different_axis'
        if name == 'controller': bad['action_representation']['execution'] = 'different servo'
        if name == 'anchor': bad['action_representation']['anchor'] = 'measured q_t'
        rejected.append(expect_rejection(name, lambda: make_windows([train[0], incompatible], 1, args.horizon)))
    # Constant channel policy demonstrated on a distinct named toy training fixture.
    constant_train = copy.deepcopy(train)
    for arrays, info in constant_train:
        info['episode_id'] += '_constant_probe'
        arrays['action'][:, 1] = .25
        arrays['proprioception'] = servo(arrays['proprioception'][0], arrays['action'], args.gain)
        info['success'] = bool(np.max(np.abs(arrays['proprioception'][-1]-arrays['action'][-1])) <= .02)
        validate(arrays, info)
    constant_stats = fit_stats(constant_train, 'joint_absolute')
    require(constant_stats['std'][1] == 0 and constant_stats['scale'][1] == 1, 'constant channel policy')
    require(transform(np.array([[.2,.25]], dtype=np.float32), constant_stats, signature['joint_absolute'])[0,1] == 0, 'constant channel normalization')
    payload = dict(measured_joint=q, measured_xy=position, previous_reference=previous,
                   measured_joint_displacement=np.diff(q, axis=0), command_delta_rate=value['joint_delta_measured']/meta['control_dt'],
                   local_frame_cartesian_delta=local_delta.astype(np.float32), **value,
                   **{name+'_normalized': x for name, x in normalized.items()})
    np.savez_compressed(args.output/'representations.npz', **payload)
    with np.load(args.output/'representations.npz', allow_pickle=False) as saved:
        require(all(np.array_equal(saved[k], x) for k, x in payload.items()), 'representation round-trip')
    np.savez_compressed(args.output/'train_chunks.npz', **windows, action_normalized=normalized_chunk)
    with np.load(args.output/'train_chunks.npz', allow_pickle=False) as saved:
        require(all(np.array_equal(saved[k], x) for k, x in windows.items()), 'chunk round-trip')
        require(np.array_equal(saved['action_normalized'], normalized_chunk), 'normalized chunk round-trip')
    (args.output/'normalizers.json').write_text(json.dumps(stats, indent=2)+'\n', encoding='utf-8')
    require(json.loads((args.output/'normalizers.json').read_text()) == stats, 'normalizer JSON round-trip')
    descriptions = {}
    extra = dict(measured_joint=('rad', 'ordered joint axes', 'measured q_i at observations i=0..T'),
                 measured_xy=('m', 'fixed base = world', 'FK(q_i) at observations i=0..T; derived, not a camera'),
                 previous_reference=('rad', 'ordered joint axes', 'r_(t-1) with r_-1=q_0'),
                 measured_joint_displacement=('rad', 'ordered joint axes', 'q_(t+1)-q_t; result of transition, not a policy input'),
                 command_delta_rate=('rad/s', 'ordered joint axes', '(r_t-q_t)/dt; command error per interval, not measured angular velocity'),
                 local_frame_cartesian_delta=('m', 'fixed local axes rotated +90 degrees to world', 'R^T*(FK(r_t)-FK(q_t)); derived target delta'))
    for name, x in payload.items():
        if name in extra:
            unit, frame, temporal = extra[name]
        else:
            normalized_name = name.endswith('_normalized')
            base_name = name.removesuffix('_normalized')
            unit = '1' if normalized_name else signature[base_name]['unit']
            frame = signature[base_name]['frame']
            temporal = 'derived outgoing command view at transition t -> t+1; signature states anchor'
        descriptions[name] = dict(shape=list(x.shape), dtype=str(x.dtype), unit=unit, frame=frame, temporal=temporal)
    report = dict(schema_version='continuous_action_experiment/1.0', gain=args.gain, dt_s=.1,
        environment=dict(python=sys.version.split()[0], executable=sys.executable, numpy=np.__version__, numpy_path=np.__file__),
        source='synthetic servo only; not Bridge conversion or a learned policy', split=dict(train=[m['episode_id'] for _, m in train], validation=[m['episode_id'] for _, m in validation]),
        representation_signatures=signature, derived_arrays=descriptions,
        episode_results=[dict(episode_id=m['episode_id'], split=m['split'],
            final_tracking_error_rad=(a['action'][-1]-a['proprioception'][-1]).tolist(),
            toy_success=m['success'], truncated=True) for a,m in episodes],
        first_step={name:x[0].tolist() for name,x in value.items()}, measured_joint=q.tolist(), reference=r.tolist(),
        final_tracking_error_rad=(r[-1]-q[-1]).tolist(), toy_success=meta['success'], ending='truncated: four-command budget',
        delta_anchor_difference_max_rad=float(np.max(np.abs(value['joint_delta_measured']-value['joint_delta_previous_reference']))),
        command_vs_measured_displacement_max_rad=float(np.max(np.abs(value['joint_delta_measured']-np.diff(q,axis=0)))),
        correct_delta_replays_match=True, wrong_delta_as_absolute_final_error_rad=(wrong_replay[-1]-q[-1]).tolist(),
        stale_chunk_anchor_max_error_rad=stale_error.tolist(), link_length_mismatch_max_xy_error_m=float(np.max(np.abs(other_position-value['cartesian_absolute']))),
        wrong_local_delta_as_world_max_error_m=float(np.max(np.abs(local_delta-world_delta))),
        normalization=dict(fit_rows=12, train_mean_rad=train_mean.tolist(), wrong_all_data_mean_rad=wrong_all_mean.tolist(),
            wrong_mean_after_validation_perturbation_rad=wrong_altered_mean.tolist(), validation_zscore_max_abs=float(np.abs(val_normalized).max()),
            max_roundtrip_error=float(max(np.max(np.abs(transform(normalized[n], stats[n], signature[n], inverse=True)-value[n])) for n in value)),
            constant_channel_probe=constant_stats, no_clipping=True),
        chunk=dict(horizon=args.horizon, shape=list(windows['action'].shape), anchors=len(windows['anchor']),
            valid_tokens=int(mask.sum()), first_indices=windows['action_index'][0].tolist(), last_indices=windows['action_index'][-1].tolist(),
            last_mask=mask[-1].tolist(), normalization_uses_real_slots_only=True), rejected_cases=rejected,
        limitations='Discrete servo toy, planar position-only FK, independent synthetic resets, no contact/inertia/IK policy, no real robot success or learned-model evaluation.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('gain','first_step','final_tracking_error_rad','toy_success','delta_anchor_difference_max_rad','command_vs_measured_displacement_max_rad','chunk','rejected_cases')},indent=2))
    print(f'PASS raw/view/chunk round-trip / correct delta replay / frame+embodiment probes / train-only normalization / rejection; output={args.output}')


if __name__ == '__main__':
    main()
