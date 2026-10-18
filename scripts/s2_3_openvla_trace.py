"""CPU OpenVLA-inspired sequence trace and exact NumPy codec algebra; no pretrained model."""
import argparse
import json
from pathlib import Path

import numpy as np


def encode(normalized, vocab=32000):
    edges = np.linspace(-1, 1, 256)
    return (vocab - np.digitize(np.clip(normalized, -1, 1), edges)).astype(np.int64)


def decode(ids, vocab=32000, strict=True):
    if ids.dtype != np.int64 or not np.isfinite(ids).all():
        raise ValueError('int64 IDs required')
    if strict and not ((ids >= vocab-256) & (ids < vocab)).all():
        raise ValueError('outside reserved action ID range')
    edges = np.linspace(-1, 1, 256)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers[np.clip(vocab-ids-1, 0, len(centers)-1)]


def stats():
    # Separate synthetic training-domain fixtures in physical world XY delta[m].
    train = np.stack([np.linspace(-.06, .06, 101), np.linspace(-.04, .04, 101)], axis=1)
    profiles = {key: dict(q01=np.quantile(train*factor, .01, axis=0),
                      q99=np.quantile(train*factor, .99, axis=0),
                      mask=np.ones(2, dtype=bool))
            for key, factor in [('compact', 1), ('wide', 2)]}
    return profiles, dict(compact_train_actions_m=train, wide_train_actions_m=train*2)


def unnormalize(normalized, spec):
    return np.where(spec['mask'], .5*(normalized+1)*(spec['q99']-spec['q01'])+spec['q01'], normalized)


def block(x):
    score = x @ np.swapaxes(x, -1, -2) / np.sqrt(np.float32(x.shape[-1]))
    allowed = np.tril(np.ones(score.shape[-2:], dtype=bool))
    score = np.where(allowed, score, -np.inf)
    p = np.exp(score-score.max(axis=-1, keepdims=True)); p /= p.sum(axis=-1, keepdims=True)
    return x+p@x, p


def trace():
    rng = np.random.default_rng(17)
    rgb = np.arange(16*16*3, dtype=np.uint16).reshape(16,16,3) % 256
    rgb = rgb.astype(np.uint8)
    patch = (rgb.astype(np.float32)/255).reshape(4,4,4,4,3).mean(axis=(1,3)).reshape(1,16,3)
    # Fixed random proxies: not DINOv2, SigLIP or their preprocessing.
    branch_a = patch @ rng.normal(size=(3,3)).astype(np.float32)
    branch_b = patch @ rng.normal(size=(3,5)).astype(np.float32)
    fused = np.concatenate([branch_a,branch_b],axis=-1)
    projected = fused @ rng.normal(size=(8,8)).astype(np.float32)
    embedding = rng.normal(size=(32000,8)).astype(np.float32)
    # Hand-declared toy prompt IDs, not Llama tokenizer output.
    prompt_ids = np.array([[1,10,11,12]],dtype=np.int64)
    prompt = embedding[prompt_ids]
    sequence = np.concatenate([prompt[:,:1],projected,prompt[:,1:]],axis=1)
    # Teacher-forced fixed two-field action labels; no policy learns/generates these.
    action_ids = encode(np.array([-.3,.4]))
    all_ids = np.concatenate([prompt_ids[0],action_ids,[2]]).astype(np.int64)
    text = embedding[all_ids][None]
    multimodal = np.concatenate([text[:,:1],projected,text[:,1:]],axis=1)
    labels = np.full(multimodal.shape[:2],-100,dtype=np.int64)
    labels[0,20:22] = action_ids  # prompt4 + patches16, then action2
    out, attention = block(multimodal)
    changed = multimodal.copy(); changed[:,21] += 3
    other,_ = block(changed)
    np.testing.assert_array_equal(out[:,:21],other[:,:21])
    np.testing.assert_array_equal(multimodal[:,:1],text[:,:1])
    np.testing.assert_array_equal(multimodal[:,1:17],projected)
    assert len(fused[0])==16 and fused.shape[-1]==8
    # Shift inside a causal LM: output at 19 predicts label at 20, output20 predicts label21.
    prediction_positions = np.flatnonzero(labels[0,1:] != -100)
    assert prediction_positions.tolist()==[19,20]
    return dict(rgb=rgb, branch_a=branch_a,branch_b=branch_b,fused=fused,
                projected=projected,prompt_ids=prompt_ids,prefix=sequence,
                teacher_forced=multimodal,labels=labels,embeddings=out,
                attention=attention,prediction_positions=prediction_positions,
                fixed_action_ids=action_ids)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unnorm-key',choices=('compact','wide'),default='compact')
    p.add_argument('--output',type=Path,default=Path('tmp/s2_3'))
    args=p.parse_args()
    arrays=trace(); profiles, training_arrays=stats(); arrays.update(training_arrays)
    normalized=decode(arrays['fixed_action_ids'])
    physical={k:unnormalize(normalized,s) for k,s in profiles.items()}
    np.testing.assert_allclose(physical['wide'],2*physical['compact'],atol=1e-15)
    # Match the selected fixed code's endpoint and ID-offset rules, not S0.6's codec.
    samples=np.array([-2.,-1.,0.,1.,2.]); ids=encode(samples); recovered=decode(ids)
    assert ids.tolist()==[31999,31999,31872,31744,31744]
    assert encode(np.array([1-1e-9,1.])).tolist()==[31745,31744]
    assert decode(np.array([31745,31744],dtype=np.int64))[0]==decode(np.array([31745,31744],dtype=np.int64))[1]
    assert len(np.unique(decode(np.arange(31744,32000,dtype=np.int64))))==255
    assert np.max(np.abs(recovered-np.clip(samples,-1,1))) <= 1/255+1e-15
    rejected=0
    for bad in (np.array([1],dtype=np.int64),np.array([32000],dtype=np.int64),np.array([31872.])):
        try: decode(bad)
        except ValueError: rejected+=1
        else: raise AssertionError('bad ID accepted')
    assert rejected==3
    # Official permissive clipping can turn a non-action ID into a boundary action.
    clipped_bad=decode(np.array([1],dtype=np.int64),strict=False)
    spec=dict(q01=np.array([-.1,-.2]),q99=np.array([.1,.2]),mask=np.array([True,False]))
    assert unnormalize(np.array([.5,.5]),spec)[1]==.5
    arrays.update(normalized_action=normalized,compact_action_m=physical['compact'],
                  wide_action_m=physical['wide'],codec_samples=samples,codec_ids=ids,codec_decoded=recovered)
    report=dict(schema_version='openvla_trace_concept/1.0',learned_policy=False,
                selected_unnorm_key=args.unnorm_key,selected_action_m=physical[args.unnorm_key].tolist(),
                profiles={k:{j:v.tolist() for j,v in s.items()} for k,s in profiles.items()},
                tensor_shapes={k:list(v.shape) for k,v in arrays.items()},
                official_codec=dict(effective_vocab=32000,model_vocab=32064,grid_points=256,
                                    unique_centers=255,valid_id_range=[31744,31999],
                                    bad_text_id_1_permissive_decoded=clipped_bad.tolist()),
                strict_rejections=rejected,units='physical fixture: world XY delta[m], dt=.1s; head/attention unitless',
                evidence_limit='fixed synthetic teacher forcing and codec only; no Llama tokenizer, learned encoders, generation, robot control or latency')
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'trace.npz',**arrays)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with np.load(args.output/'trace.npz',allow_pickle=False) as saved:
        for k,v in arrays.items(): np.testing.assert_array_equal(saved[k],v)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
