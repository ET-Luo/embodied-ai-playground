"""Pi0-inspired two-expert attention/cache and analytic flow probes; no learned policy."""
import argparse
import json
from pathlib import Path

import numpy as np


def mask():
    groups=np.array([0,0,0,1,2,2,2])  # prefix3, state1, action3
    return groups[None,:] <= groups[:,None]


def weights():
    rng=np.random.default_rng(23)
    # Each expert has its own width/projections; Q/K/V share attention width3.
    return [[{name:rng.normal(scale=.15,size=shape) for name,shape in
              [('q',(width,3)),('k',(width,3)),('v',(width,3)),('o',(3,width))]}
             for width in (6,4)] for _ in range(2)]


def attend(q,k,v,allowed):
    score=q@k.T/np.sqrt(3)
    score=np.where(allowed,score,-np.inf)
    p=np.exp(score-score.max(axis=-1,keepdims=True)); p/=p.sum(axis=-1,keepdims=True)
    return p@v


def full(prefix,suffix,params):
    for wp,ws in params:
        kp,vp=prefix@wp['k'],prefix@wp['v']
        ks,vs=suffix@ws['k'],suffix@ws['v']
        common=attend(np.concatenate([prefix@wp['q'],suffix@ws['q']]),
                      np.concatenate([kp,ks]),np.concatenate([vp,vs]),mask())
        prefix=prefix+common[:3]@wp['o']
        suffix=suffix+common[3:]@ws['o']
    return prefix,suffix


def cache_prefix(prefix,params):
    cache=[]
    for wp,_ in params:
        k,v=prefix@wp['k'],prefix@wp['v']; cache.append((k,v))
        prefix=prefix+attend(prefix@wp['q'],k,v,np.ones((3,3),bool))@wp['o']
    return prefix,cache


def cached_suffix(suffix,cache,params):
    for (_,ws),(kp,vp) in zip(params,cache):
        k=np.concatenate([kp,suffix@ws['k']]); v=np.concatenate([vp,suffix@ws['v']])
        suffix=suffix+attend(suffix@ws['q'],k,v,mask()[3:])@ws['o']
    return suffix


def integrate(initial,velocity,steps):
    if type(steps) is not int or steps<=0: raise ValueError('positive integer steps')
    x=initial.copy(); trace=[x.copy()]; times=[1.]
    for k in range(steps):
        t=1-k/steps
        v=velocity(x,t)
        if v.shape!=x.shape or not np.isfinite(v).all(): raise ValueError('finite shape-preserving velocity')
        x=x-v/steps
        trace.append(x.copy()); times.append(1-(k+1)/steps)
    return np.stack(trace),np.array(times)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,choices=(10,20),default=10)
    p.add_argument('--output',type=Path,default=Path('tmp/s2_5'))
    args=p.parse_args()
    rng=np.random.default_rng(19)
    prefix=rng.normal(size=(3,6)); suffix=rng.normal(size=(4,4)); params=weights()
    po,so=full(prefix,suffix,params)
    cp,cache=cache_prefix(prefix,params); cs=cached_suffix(suffix,cache,params)
    np.testing.assert_allclose(po,cp,atol=1e-14); np.testing.assert_allclose(so,cs,atol=1e-14)
    altered=suffix.copy(); altered[1:]+=1
    ap,ass=full(prefix,altered,params)
    np.testing.assert_array_equal(po,ap); np.testing.assert_array_equal(so[0],ass[0])
    state=suffix.copy(); state[0]+=.5
    _,ss=full(prefix,state,params); assert not np.allclose(so[1:],ss[1:])
    assert int(mask().sum())==34
    raw=np.array([[.01,-.02],[.02,.01],[-.01,.03]],dtype=np.float64)
    clean=raw/.06; noise=np.random.default_rng(29).normal(size=(3,2))
    t=.7; noisy=t*noise+(1-t)*clean; target_v=noise-clean
    np.testing.assert_allclose(noisy-t*target_v,clean,atol=1e-15)
    oracle,flow_times=integrate(noise,lambda x,t:target_v,args.steps)
    np.testing.assert_allclose(oracle[-1],clean,atol=1e-14)
    biased,_=integrate(noise,lambda x,t:target_v+.1,args.steps)
    np.testing.assert_allclose(biased[-1],clean-.1,atol=1e-14)
    # Analytic test ODE: dx/dt=x-mu. Its t=0 endpoint is not the clean action.
    mu=clean
    curved,_=integrate(noise,lambda x,t:x-mu,args.steps)
    exact=mu+(noise-mu)*np.exp(-1)
    curved_error=float(np.sqrt(np.mean((curved[-1]-exact)**2)))
    rejected=0
    for velocity,steps in [(lambda x,t:x,0),(lambda x,t:x[:1],10),(lambda x,t:x*np.nan,10)]:
        try: integrate(noise,velocity,steps)
        except ValueError: rejected+=1
        else: raise AssertionError('bad integrator input accepted')
    arrays=dict(prefix=prefix,suffix=suffix,attention_mask=mask(),full_prefix=po,full_suffix=so,
                cached_suffix=cs,raw_action_m=raw,normalized_clean=clean,initial_noise=noise,
                noisy_training_sample=noisy,target_velocity=target_v,flow_times=flow_times,
                oracle_trace=oracle,biased_trace=biased,analytic_trace=curved,analytic_endpoint=exact)
    report=dict(schema_version='pi0_flow_trace/1.0',learned_policy=False,steps=args.steps,
                horizon=3,action_dim=2,control_dt_s=.1,flow_dt=-1/args.steps,
                prefix_tokens=3,state_tokens=1,action_tokens=3,allowed_edges=int(mask().sum()),
                prefix_width=6,expert_width=4,shared_attention_width=3,
                cache_max_error=float(np.max(np.abs(so-cs))),
                oracle_rmse_normalized=float(np.sqrt(np.mean((oracle[-1]-clean)**2))),
                biased_rmse_normalized=float(np.sqrt(np.mean((biased[-1]-clean)**2))),
                analytic_euler_rmse_normalized=curved_error,rejection_checks=rejected,
                evidence_limit='untrained attention/cache + privileged oracle and analytic ODE only; no VLM, learned velocity, robot success or model latency')
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'trace.npz',**arrays)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with np.load(args.output/'trace.npz',allow_pickle=False) as saved:
        for k,v in arrays.items(): np.testing.assert_array_equal(saved[k],v)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
