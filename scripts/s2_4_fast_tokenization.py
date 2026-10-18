"""FAST-inspired orthonormal DCT + integer-pair BPE, CPU only; not the released FAST tokenizer."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


H, A = 16, 2
PHYSICAL_SCALE = .06  # m per normalized displacement; declared, not fitted
ALPHABET_LIMIT = 128  # fixed arithmetic domain, not inferred from held-out data


def dct_matrix(h):
    k = np.arange(h)[:, None]
    t = np.arange(h)[None, :]
    matrix = np.sqrt(2/h)*np.cos(np.pi*(t+.5)*k/h)
    matrix[0] /= np.sqrt(2)
    return matrix


def fixtures():
    t = (np.arange(H)+.5)/H
    train = []
    for i in range(8):
        x = np.stack([.15+.03*i+.10*np.cos(np.pi*t),
                      -.12+.02*i+.08*np.cos(2*np.pi*t)], axis=-1)
        train.append(x)
    heldout = dict(smooth=np.stack([.375+.11*np.cos(np.pi*t),
                                   -.055+.07*np.cos(2*np.pi*t)], axis=-1),
                   rough=np.random.default_rng(31).uniform(-.7,.7,(H,A)))
    return np.stack(train)*PHYSICAL_SCALE, {k:v*PHYSICAL_SCALE for k,v in heldout.items()}


def quantize(raw_m, scale, matrix):
    if raw_m.shape != (H,A) or not np.isfinite(raw_m).all():
        raise ValueError('finite chunk (16,2) required')
    coeff = matrix @ (raw_m/PHYSICAL_SCALE)
    q = np.rint(scale*coeff).astype(np.int64)
    if np.max(np.abs(q)) > ALPHABET_LIMIT:
        raise ValueError('coefficient exceeds fixed alphabet; no silent clipping')
    return coeff, q


def merge_pair(seq, pair, new_id):
    result=[]; i=0
    while i<len(seq):
        if i+1<len(seq) and (seq[i],seq[i+1])==pair:
            result.append(new_id); i+=2
        else:
            result.append(seq[i]); i+=1
    return result


def fit_bpe(train_sequences, max_merges=24):
    # Every signed integer is one base symbol. This is NOT byte-level Unicode BPE.
    expansions={q+ALPHABET_LIMIT:[q] for q in range(-ALPHABET_LIMIT,ALPHABET_LIMIT+1)}
    sequences=[[int(q)+ALPHABET_LIMIT for q in seq] for seq in train_sequences]
    merges=[]
    for _ in range(max_merges):
        counts=Counter(pair for seq in sequences for pair in zip(seq,seq[1:]))
        if not counts: break
        pair,count=min(counts.items(),key=lambda item:(-item[1],item[0]))
        if count<2: break
        new_id=len(expansions)
        expansions[new_id]=expansions[pair[0]]+expansions[pair[1]]
        merges.append((pair,new_id))
        sequences=[merge_pair(seq,pair,new_id) for seq in sequences]
    return merges,expansions


def bpe_encode(q, merges):
    seq=(q.flatten()+ALPHABET_LIMIT).tolist()
    for pair,new_id in merges: seq=merge_pair(seq,pair,new_id)
    return seq


def bpe_decode(tokens, expansions):
    if not tokens or any(type(i) is not int or i not in expansions for i in tokens):
        raise ValueError('nonempty valid token IDs required')
    q=np.array([q for token in tokens for q in expansions[token]],dtype=np.int64)
    if len(q)!=H*A: raise ValueError('decoded coefficient count mismatch; H/A contract required')
    return q.reshape(H,A)


def metrics(raw,decoded):
    diff=decoded-raw
    return dict(rmse_m=float(np.sqrt(np.mean(diff**2))), max_absolute_error_m=float(np.max(np.abs(diff))),
                integrated_displacement_error_m=float(np.linalg.norm(diff.sum(axis=0))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale',type=int,choices=(10,20),default=10)
    p.add_argument('--output',type=Path,default=Path('tmp/s2_4'))
    args=p.parse_args()
    matrix=dct_matrix(H)
    np.testing.assert_allclose(matrix.T@matrix,np.eye(H),atol=5e-15)
    train,heldout=fixtures()
    train_q=[quantize(raw,args.scale,matrix)[1] for raw in train]
    merges,expansions=fit_bpe([q.flatten() for q in train_q])
    rows={}; arrays=dict(train_raw_m=train,dct_matrix=matrix,train_quantized=np.stack(train_q))
    for name,raw in heldout.items():
        coeff,q=quantize(raw,args.scale,matrix)
        np.testing.assert_allclose(matrix.T@coeff,raw/PHYSICAL_SCALE,atol=5e-15)
        tokens=bpe_encode(q,merges)
        restored=bpe_decode(tokens,expansions)
        np.testing.assert_array_equal(restored,q)
        decoded=(matrix.T@(restored/args.scale))*PHYSICAL_SCALE
        m=metrics(raw,decoded)
        assert m['rmse_m'] <= PHYSICAL_SCALE/(2*args.scale)+1e-14
        normalized=raw/PHYSICAL_SCALE
        scalar_ids=np.minimum(np.floor((np.clip(normalized,-1,1)+1)*128).astype(np.int64),255)
        baseline=((scalar_ids+.5)/128-1)*PHYSICAL_SCALE
        rows[name]=dict(baseline_token_count=H*A,coefficient_count=H*A,bpe_token_count=len(tokens),
                        nonzero_coefficients=int(np.count_nonzero(q)),
                        coefficient_to_bpe_ratio=(H*A)/len(tokens),
                        dct_quantization=m,scalar_bins256=metrics(raw,baseline))
        arrays.update({name+'_'+k:v for k,v in dict(raw_m=raw,coefficients=coeff,quantized=q,
                      bpe_ids=np.array(tokens,dtype=np.int64),decoded_m=decoded,
                      scalar_bin_ids=scalar_ids,scalar_decoded_m=baseline).items()})
    # Held-out changes cannot affect fitted dictionary or training quantization.
    merges_again,expansions_again=fit_bpe([q.flatten() for q in train_q])
    assert merges_again==merges and expansions_again==expansions
    rejected=0
    for tokens in ([],[99999],[0]):
        try: bpe_decode(tokens,expansions)
        except ValueError: rejected+=1
        else: raise AssertionError('invalid stream accepted')
    for raw in (np.zeros((2,H)),np.full((H,A),np.nan),np.full((H,A),100.)):
        try: quantize(raw,args.scale,matrix)
        except ValueError: rejected+=1
        else: raise AssertionError('invalid chunk accepted')
    assert rejected==6
    spec=dict(schema_version='integer_pair_bpe/1.0',scale=args.scale,horizon=H,action_dim=A,
              physical_scale_m=PHYSICAL_SCALE,alphabet_limit=ALPHABET_LIMIT,
              train_ids=[f'train_{i}' for i in range(8)],
              flatten='frequency major then action dimension; C-order (H,A)',
              merges=[dict(pair=list(pair),id=i) for pair,i in merges],
              expansions={str(k):v for k,v in expansions.items()})
    report=dict(schema_version='fast_concept/1.0',learned_policy=False,scale=args.scale,
                history=None,horizon=H,action_dim=A,control_dt_s=.1,
                future_chunk_span_s=H*.1,rmse_bound_m=PHYSICAL_SCALE/(2*args.scale),
                train_chunks=8,heldout_chunks=2,merge_count=len(merges),vocab_size=len(expansions),
                rejection_checks=rejected,cases=rows,
                evidence_limit='synthetic codec evaluation only; no VLA training, robot success or model latency; token counts exclude dictionary/framing')
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'trace.npz',**arrays)
    (args.output/'tokenizer.json').write_text(json.dumps(spec,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    loaded=json.loads((args.output/'tokenizer.json').read_text())
    restored_expansions={int(k):v for k,v in loaded['expansions'].items()}
    with np.load(args.output/'trace.npz',allow_pickle=False) as saved:
        for k,v in arrays.items(): np.testing.assert_array_equal(saved[k],v)
        for name in heldout:
            np.testing.assert_array_equal(bpe_decode(saved[name+'_bpe_ids'].tolist(),restored_expansions),saved[name+'_quantized'])
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
