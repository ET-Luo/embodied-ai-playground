"""S0.6: train-range scalar bins, channel-specific IDs, center decode, CPU toy replay."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

from s0_1_data_contract import require
from s0_5_action_representation import episode, expect_rejection, round_trip, servo, views


def fit_tokenizer(train, bins):
    require(type(bins) is int and 2 <= bins <= 65536, 'bins integer in [2,65536]')
    require(bool(train) and all(m['split'] == 'train' for _, m in train), 'fit training episodes only')
    signatures = [views(a, m)[1]['joint_absolute'] for a, m in train]
    require(all(s == signatures[0] for s in signatures), 'fit semantics mismatch')
    x = np.concatenate([a['action'] for a, _ in train]).astype(np.float64)
    require(np.isfinite(x).all(), 'finite training actions')
    low, high = x.min(axis=0), x.max(axis=0)
    require((high > low).all(), 'constant channel needs an explicit codec extension')
    return dict(schema_version='scalar_action_bins/1.0', bins=bins, channels=2,
                low=low.tolist(), high=high.tolist(), signature=signatures[0],
                fit_episode_ids=[m['episode_id'] for _, m in train], fit_rows=len(x),
                range_rule='per-channel training raw min/max; no validation/padding/window repetition',
                boundary_rule='internal edges go to right bin; both outer endpoints are in range',
                decode_rule='bin center', id_rule='channel*bins+bin_index',
                serialization='time major: q1_t,q2_t,q1_(t+1),q2_(t+1); no special text tokens')


def grid(spec, signature):
    require(spec.get('schema_version') == 'scalar_action_bins/1.0', 'tokenizer schema')
    require(spec.get('signature') == signature, 'tokenizer semantics mismatch')
    b = spec.get('bins')
    require(type(b) is int and 2 <= b <= 65536, 'bins integer in [2,65536]')
    require(spec.get('channels') == 2, 'two-channel codec only')
    low, high = np.asarray(spec['low'], dtype=np.float64), np.asarray(spec['high'], dtype=np.float64)
    require(low.shape == high.shape == (2,) and np.isfinite(low).all() and np.isfinite(high).all()
            and (high > low).all(), 'valid per-channel range')
    require(spec.get('boundary_rule') == 'internal edges go to right bin; both outer endpoints are in range'
            and spec.get('decode_rule') == 'bin center' and spec.get('id_rule') == 'channel*bins+bin_index', 'unsupported codec rules')
    edges = np.stack([np.linspace(low[j], high[j], b+1) for j in range(2)], axis=1)
    return edges, (high-low)/b


def encode(actions, spec, signature, overflow='clip'):
    edges, _ = grid(spec, signature)
    require(actions.ndim == 2 and actions.shape[1] == 2 and len(actions) > 0, 'actions (T,2)')
    require(np.issubdtype(actions.dtype, np.floating) and np.isfinite(actions).all(), 'finite floating actions')
    require(overflow in ('clip', 'reject'), 'overflow clip or reject')
    x = actions.astype(np.float64)
    saturated = (x < edges[0]) | (x > edges[-1])
    require(overflow != 'reject' or not saturated.any(), 'out-of-range action')
    clipped = np.clip(x, edges[0], edges[-1])
    ids = np.empty(actions.shape, dtype=np.int64)
    for j in range(2):
        ids[:, j] = j*spec['bins'] + np.searchsorted(edges[1:-1, j], clipped[:, j], side='right')
    return ids, saturated, clipped


def decode(ids, spec, signature):
    edges, _ = grid(spec, signature)
    require(ids.ndim == 2 and ids.shape[1] == 2 and len(ids) > 0 and ids.dtype == np.int64, 'IDs (T,2) int64')
    b = spec['bins']
    bins = ids - np.arange(2)*b
    require(((bins >= 0) & (bins < b)).all(), 'wrong channel namespace or invalid ID')
    centers = (edges[:-1] + edges[1:])/2
    return np.stack([centers[bins[:,j],j] for j in range(2)], axis=1).astype(np.float32)


def boundary_probe(spec, signature):
    edges, width = grid(spec, signature)
    rows = []
    for j in range(2):
        k = spec['bins']//2
        edge = edges[k,j]
        samples = [edges[0,j]-.1, edges[0,j], np.nextafter(edge, -np.inf), edge,
                   np.nextafter(edge, np.inf), edges[-1,j], edges[-1,j]+.1]
        x = np.tile(edges[0], (len(samples),1)); x[:,j] = samples
        ids, saturated, _ = encode(x, spec, signature)
        indices = ids[:,j]-j*spec['bins']
        require(indices.tolist() == [0,0,k-1,k,k,spec['bins']-1,spec['bins']-1], 'boundary assignment oracle')
        require(saturated[:,j].tolist() == [True,False,False,False,False,False,True], 'boundary saturation oracle')
        rows.append(dict(channel=j, samples=samples, bin_indices=indices.tolist(),
                         saturation=saturated[:,j].tolist(), decoded=decode(ids,spec,signature)[:,j].tolist()))
    return dict(width_rad=width.tolist(), in_range_half_width_bound_rad=(width/2).tolist(),
                numeric_tolerance_rad=1e-7, channels=rows)


def reconstruction_error(raw, decoded, clipped, saturated, width):
    quantization = np.abs(decoded.astype(np.float64)-clipped)
    require((quantization <= width/2+1e-7).all(), 'center quantization exceeds half-bin bound')
    total = np.abs(decoded.astype(np.float64)-raw)
    return dict(per_channel_mae_rad=total.mean(axis=0).tolist(), per_channel_max_error_rad=total.max(axis=0).tolist(),
                saturated_scalars=int(saturated.sum()), total_scalars=raw.size,
                saturation_fraction=float(saturated.mean()),
                per_channel_clipping_max_error_rad=np.abs(clipped-raw).max(axis=0).tolist(),
                per_channel_quantization_max_error_rad=quantization.max(axis=0).tolist(),
                in_range_bound_passed=bool((total[~saturated] <= np.broadcast_to(width/2+1e-7, raw.shape)[~saturated]).all()))


def replay(arrays, decoded, gain):
    """Counterfactual toy execution only; original raw measurements stay unchanged."""
    raw_reference = arrays['action']
    baseline = servo(arrays['proprioception'][0], raw_reference, gain)
    quantized = servo(arrays['proprioception'][0], decoded, gain)
    baseline_error = float(np.max(np.abs(raw_reference[-1]-baseline[-1])))
    quantized_error = float(np.max(np.abs(raw_reference[-1]-quantized[-1])))
    return dict(gain=gain, final_target='original physical reference, not decoded target', threshold_rad=.02,
                baseline_final_error_rad=baseline_error, quantized_final_error_rad=quantized_error,
                baseline_success=baseline_error <= .02, quantized_success=quantized_error <= .02,
                max_trace_component_difference_rad=float(np.max(np.abs(quantized-baseline))),
                failure_stage=None if quantized_error <= .02 else 'final joint tracking threshold',
                baseline_trace=baseline.tolist(), quantized_trace=quantized.tolist(),
                ending='four-command budget truncated; success independent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bins', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('tmp/s0_6'))
    args = parser.parse_args()
    require(type(args.bins) is int and 2 <= args.bins <= 65536, 'bins integer in [2,65536]')
    require(not args.output.exists(), 'choose a new output directory')
    episodes = [episode(i,np.array(offset,dtype=np.float32),split,.5) for i,offset,split in
                [(0,[0,0],'train'),(1,[.1,0],'train'),(2,[-.1,.05],'train'),(3,[.8,-.1],'validation')]]
    train = episodes[:3]
    spec = fit_tokenizer(train,args.bins); signature = spec['signature']
    probe = boundary_probe(spec,signature)
    edges, width = grid(spec,signature)
    # Independent hand-sized codec: IDs and center error are easy to inspect.
    manual = copy.deepcopy(spec); manual.update(bins=4, low=[0.,0.], high=[1.,1.])
    toy = np.array([[0.,.25],[.5,1.]],dtype=np.float64)
    toy_ids,_,_ = encode(toy,manual,signature)
    require(toy_ids.tolist() == [[0,5],[2,7]], 'manual ID oracle')
    np.testing.assert_array_equal(decode(toy_ids,manual,signature),[[.125,.375],[.625,.875]])
    # Changing held-out data never refits train range. Wrong all-data fit would move it.
    changed = copy.deepcopy(episodes[3]); changed[0]['action'][:,0] += .2
    require(fit_tokenizer(train,args.bins) == spec, 'train-only range changed')
    wrong_high = np.concatenate([a['action'] for a,_ in episodes]).max(axis=0)
    changed_high = np.concatenate([a['action'] for a,_ in train+[changed]]).max(axis=0)
    require(not np.array_equal(wrong_high,changed_high), 'range leakage probe ineffective')
    rejected = [expect_rejection('validation_fit',lambda:fit_tokenizer(episodes,args.bins)),
                expect_rejection('bins_one',lambda:fit_tokenizer(train,1)),
                expect_rejection('nan_action',lambda:encode(np.full((1,2),np.nan),spec,signature)),
                expect_rejection('wrong_action_width',lambda:encode(np.zeros((1,3)),spec,signature)),
                expect_rejection('float_ids',lambda:decode(np.zeros((1,2)),spec,signature)),
                expect_rejection('negative_id',lambda:decode(np.array([[-1,args.bins]],dtype=np.int64),spec,signature)),
                expect_rejection('wrong_channel_namespace',lambda:decode(np.array([[0,0]],dtype=np.int64),spec,signature)),
                expect_rejection('out_of_range_reject',lambda:encode(episodes[3][0]['action'],spec,signature,'reject'))]
    for key in ('anchor','unit','frame','fields','embodiment','controller','dt_s'):
        wrong = copy.deepcopy(signature)
        wrong[key] = 'incompatible'
        rejected.append(expect_rejection('signature_'+key,lambda:decode(toy_ids,spec,wrong)))
    bad_spec = copy.deepcopy(spec); bad_spec['high'][0] = bad_spec['low'][0]
    rejected.append(expect_rejection('constant_range',lambda:grid(bad_spec,signature)))
    args.output.mkdir(parents=True,exist_ok=False)
    episodes = [round_trip(args.output/'episodes'/m['episode_id'],a,m) for a,m in episodes]
    results=[]
    payload={}
    for arrays,meta in episodes:
        current = views(arrays,meta)[1]['joint_absolute']
        ids,saturated,clipped = encode(arrays['action'],spec,current)
        decoded = decode(ids,spec,current)
        serialized = ids.reshape(-1)
        require(np.array_equal(serialized.reshape(ids.shape),ids), 'time-major sequence round-trip')
        # Center representatives always re-encode to the same bin in this tested codec.
        require(np.array_equal(encode(decoded,spec,current)[0],ids), 'decode -> encode ID stability')
        for key,x in dict(ids=ids,serialized_ids=serialized,decoded=decoded,saturated=saturated).items():
            payload[meta['episode_id']+'_'+key]=x
        results.append(dict(episode_id=meta['episode_id'],split=meta['split'],
            error=reconstruction_error(arrays['action'],decoded,clipped,saturated,width),
            ids=ids.tolist(),replay_gain_half=replay(arrays,decoded,.5),replay_gain_one=replay(arrays,decoded,1.)))
    np.savez_compressed(args.output/'tokens.npz',**payload)
    with np.load(args.output/'tokens.npz',allow_pickle=False) as saved:
        require(all(np.array_equal(saved[k],x) for k,x in payload.items()), 'token NPZ round-trip')
    (args.output/'tokenizer.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
    saved_spec=json.loads((args.output/'tokenizer.json').read_text())
    require(saved_spec == spec, 'tokenizer JSON round-trip')
    require(np.array_equal(decode(payload['servo_0_ids'],saved_spec,signature),payload['servo_0_decoded']), 'saved codec decode')
    evaluation = {}
    for split in ('train', 'validation'):
        selected = [r for r in results if r['split'] == split]
        evaluation[split] = dict(episodes=len(selected),
            saturated_scalars=sum(r['error']['saturated_scalars'] for r in selected),
            total_scalars=sum(r['error']['total_scalars'] for r in selected),
            replay={key:dict(baseline_successes=sum(r[key]['baseline_success'] for r in selected),
                            quantized_successes=sum(r[key]['quantized_success'] for r in selected),
                            denominator=len(selected)) for key in ('replay_gain_half','replay_gain_one')})
    report=dict(schema_version='action_tokenization_experiment/1.0',
        environment=dict(python=sys.version.split()[0],executable=sys.executable,numpy=np.__version__,numpy_path=np.__file__),
        tokenizer=spec,vocabulary_size=2*args.bins,token_shape=[4,2],serialized_shape=[8],
        artifact_dtypes=dict(ids='int64',serialized_ids='int64',decoded='float32',saturated='bool'),
        decoded_unit='rad: joint_absolute reference, ordered q1/q2, outgoing command t -> t+1',
        overflow='explicit clip with per-scalar saturation flags; reject option tested; never silent clipping',
        boundary_probe=probe,manual_oracle=dict(ids=toy_ids.tolist(),decoded=decode(toy_ids,manual,signature).tolist()),
        range_leakage_probe=dict(train_high=spec['high'],wrong_all_high=wrong_high.tolist(),wrong_high_after_validation_change=changed_high.tolist()),
        uncompressed_action_bytes_per_episode=episodes[0][0]['action'].nbytes,
        uncompressed_id_bytes_per_episode=payload['servo_0_ids'].nbytes,
        evaluation=evaluation,episodes=results,rejected_cases=rejected,
        limitations='Scalar uniform toy codec only; not a learned tokenizer, language vocabulary, FAST or OpenVLA reproduction; no real robot execution or policy training.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(bins=args.bins,vocabulary_size=report['vocabulary_size'],boundary=probe['in_range_half_width_bound_rad'],
        episodes=[dict(episode_id=r['episode_id'],error=r['error'],gain1_success=r['replay_gain_one']['quantized_success']) for r in results],
        rejected_cases=len(rejected)),indent=2))
    print(f'PASS raw/token/codec round-trip / boundaries / half-bin error bound / saturation / semantic rejection / toy replay; output={args.output}')


if __name__ == '__main__':
    main()
