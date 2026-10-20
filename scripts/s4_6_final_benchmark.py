"""Frozen paired VLA-only / VLA+WM benchmark on disjoint grid layout pairings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

import s4_5_receding_horizon as loop
from s0_1_data_contract import require

METHODS=('vla_only','vla_wm')
SEEDS=(461,462)


def fresh_protocol(policy_contract):
    coordinates=[(u,v) for v in (4,12,20,28) for u in (4,12,20,28)]
    layouts=[(coordinates[r],coordinates[b]) for r in range(16) for b in range(16) if r!=b]
    forbidden={tuple(map(tuple,row['centers_uv'])) for split in ('train','validation') for row in policy_contract['split'][split]}
    pool=[i for i,pair in enumerate(layouts) if pair not in forbidden]
    selected=[]
    for seed in SEEDS:
        ids=np.random.default_rng(seed).choice(pool,8,replace=False).tolist()
        pool=[i for i in pool if i not in ids]
        for index in ids:selected.append(dict(seed=seed,scene_id=f'layout_{index}',centers_uv=[list(x) for x in layouts[index]]))
    require(len({tuple(map(tuple,s['centers_uv'])) for s in selected})==16,'distinct new layout groups')
    cases=[]
    for scene in selected:
        for case in loop.protocol([scene]):
            case['seed']=scene['seed'];case['case_id']=f"s{scene['seed']}_{case['case_id']}"
            cases.append(case)
    return cases


def control(rgb,state,text,method,weights,vocabulary,model,buffers,scale,strict):
    """Only current observations and frozen artifacts. No plant/evaluator inputs."""
    require(method in METHODS,'benchmark method')
    start=perf_counter_ns()
    bank=loop.proposal.propose(rgb,state,text,weights,vocabulary,1 if method=='vla_only' else 5)
    proposal_end=perf_counter_ns()
    if method=='vla_only':
        return dict(bank=bank,chosen=0,rejected=None,prediction=None,commands=bank[0].copy(),
            proposal_ms=(proposal_end-start)/1e6,world_model_ms=None,gate_ms=None)
    predictions=loop.gate.scoring.wm.imagine(np.broadcast_to(state,(1,5,2)),bank[None],model)
    model_end=perf_counter_ns()
    goal=loop.gate.scoring.observation_goals(dict(image=rgb[None],text=np.array([text])))
    scores=loop.gate.scoring.margins(predictions,goal)
    chosen,rejected=loop.gate.select(scores,buffers,scale,strict)
    commands=loop.gate.selected_commands(bank[None],chosen)[0]
    return dict(bank=bank,chosen=int(chosen[0]),rejected=rejected[0],prediction=predictions[0],commands=commands,
        proposal_ms=(proposal_end-start)/1e6,world_model_ms=(model_end-proposal_end)/1e6,
        gate_ms=(perf_counter_ns()-model_end)/1e6)


def actual_future(state,bank,case,t):
    """Evaluator-only counterfactual full H3, including future kick and unexecuted tail."""
    position=np.broadcast_to(state,(len(bank),2)).copy();trace=[position.copy()]
    for h in range(3):
        position=(position+np.float32(case['plant_gain'])*bank[:,h]).astype(np.float32)
        if t+h+1==case['kick_after_transition']:
            position=(position+np.array(case['kick_m'],np.float32)).astype(np.float32)
        trace.append(position.copy())
    return np.stack(trace,axis=1)


def run_episode(case,method,weights,vocabulary,model,buffers,scale,strict,execute):
    require(type(execute) is int and 1<=execute<=3,'1<=E<=3')
    state=np.array(case['initial_m'],np.float32);states=[state.copy()];actions=[];plans=[];t=0;failure=None
    while t<8:
        start=perf_counter_ns()
        rgb=loop.proposal.vla.render(np.array(case['centers_uv'],np.int64));render_end=perf_counter_ns()
        try:decision=control(rgb,state.copy(),case['text'],method,weights,vocabulary,model,buffers,scale,strict)
        except ValueError as error:
            failure='policy_input_or_output_rejection'
            plans.append(dict(t=t,executed_prefix=0,error=str(error),rejected_attempt_ms=(perf_counter_ns()-start)/1e6))
            break
        control_end=perf_counter_ns()  # All shadow/oracle diagnostics start AFTER the controller stopwatch.
        count=min(execute,8-t);bank=decision['bank'];commands=decision['commands']
        diagnostic_start=perf_counter_ns()
        predictions=decision['prediction']
        if predictions is None:
            predictions=loop.gate.scoring.wm.imagine(state[None],bank,model)
        chosen=decision['chosen']
        selected_prediction=predictions[chosen] if chosen>=0 else loop.gate.scoring.wm.imagine(state[None],commands[None],model)[0]
        rejection=None;primary_rejection=None
        if method=='vla_wm':
            truth=actual_future(state,bank,case,t)
            # Counterfactual evaluator annotation is deliberately separate from controller's RGB goal decoder.
            labels=loop.gate.scoring.margins(truth[None],np.array([case['evaluation_goal_m']],np.float64))
            rejected=decision['rejected'][None]
            rejection=loop.gate.rejection_counts(rejected,labels['risk']>=0,labels['success']>=0)
            primary_rejection=loop.gate.rejection_counts(rejected[:,:1],labels['risk'][:,:1]>=0,labels['success'][:,:1]>=0)
        diagnostic_ms=(perf_counter_ns()-diagnostic_start)/1e6
        apply_start=perf_counter_ns();before=t;prefix=[]
        for h in range(count):
            state=(state+np.float32(case['plant_gain'])*commands[h]).astype(np.float32);t+=1
            if t==case['kick_after_transition']:state=(state+np.array(case['kick_m'],np.float32)).astype(np.float32)
            states.append(state.copy());actions.append(commands[h].copy());prefix.append(state.copy())
        apply_ms=(perf_counter_ns()-apply_start)/1e6
        error=np.array(prefix,np.float64)-selected_prediction[1:count+1]
        plans.append(dict(t=before,planning_state_m=states[before].tolist(),executed_prefix=count,chosen=chosen,
            candidate_bank_m=bank.tolist(),selected_chunk_m=commands.tolist(),rejected=None if decision['rejected'] is None else decision['rejected'].tolist(),
            selected_prediction_m=selected_prediction.tolist(),prediction_sse_m2=float(np.sum(error**2)),
            prediction_scalars=2*count,rejection=rejection,primary_rejection=primary_rejection,
            render_ms=(render_end-start)/1e6,proposal_ms=decision['proposal_ms'],world_model_ms=decision['world_model_ms'],
            gate_ms=decision['gate_ms'],control_ms=(control_end-start)/1e6,
            diagnostic_ms=diagnostic_ms,apply_ms=apply_ms,nominal_prefix_budget_ms=count*100.))
    actual=np.array(states,np.float32);commands=np.array(actions,np.float32).reshape(-1,2)
    unsafe=bool(len(commands) and (np.linalg.norm(actual[1:].astype(np.float64)-loop.gate.scoring.CENTER,axis=-1)<=.02).any())
    error=float(np.abs(actual[-1].astype(np.float64)-case['evaluation_goal_m']).max())
    goal_success=t==8 and error<=.01;success=goal_success and not unsafe and failure is None
    if failure is None and not success:failure='monitoring_risk' if unsafe else 'final_goal_error_after_budget'
    return dict(case_id=case['case_id'],seed=case['seed'],condition=case['condition'],scene_id=case['scene_id'],method=method,
        completed_steps=t,unsafe=unsafe,goal_success=goal_success,success=success,failure_stage=failure,
        final_goal_linf_m=error,plans=plans,state_m=actual,action_m=commands)


def summarize(records):
    plans=[p for r in records for p in r['plans'] if p['executed_prefix']>0]
    rejected=[p for r in records for p in r['plans'] if p['executed_prefix']==0]
    def timing(key):
        values=[p[key] for p in plans if p[key] is not None]
        return None if not values else dict(samples=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)))
    def totals(key):
        rows=[p[key] for p in plans if p[key] is not None]
        return None if not rows else {k:sum(row[k] for row in rows) for k in rows[0]}
    scalars=sum(p['prediction_scalars'] for p in plans)
    episode_times=[sum(p['control_ms'] for p in r['plans'] if p['executed_prefix']>0) for r in records]
    return dict(attempts=len(records),layout_groups=len({r['scene_id'] for r in records}),
        completed=sum(r['completed_steps']==8 for r in records),successes=sum(r['success'] for r in records),
        goal_successes=sum(r['goal_success'] for r in records),unsafe_episodes=sum(r['unsafe'] for r in records),
        failure_stages=dict(Counter(r['failure_stage'] for r in records if r['failure_stage'])),
        accepted_calls=len(plans),input_rejections=len(rejected),fallback_calls=sum(p['chosen']<0 for p in plans),
        alternate_calls=sum(p['chosen']>0 for p in plans),
        final_goal_linf_median_m=float(np.median([r['final_goal_linf_m'] for r in records])),
        executed_prediction_rmse_m=None if not scalars else float(np.sqrt(sum(p['prediction_sse_m2'] for p in plans)/scalars)),
        executed_prediction_scalar_denominator=scalars,rejection=totals('rejection'),primary_rejection=totals('primary_rejection'),
        latency_ms={key:timing(key) for key in ('render_ms','proposal_ms','world_model_ms','gate_ms','control_ms','diagnostic_ms','apply_ms')},
        per_episode_control_ms=None if not plans else dict(median=float(np.median(episode_times)),p95=float(np.percentile(episode_times,95))),
        compute_overruns=sum(p['control_ms']>p['nominal_prefix_budget_ms'] for p in plans),
        rejection_latency_ms=None if not rejected else dict(samples=len(rejected),median=float(np.median([p['rejected_attempt_ms'] for p in rejected]))))


def paired_counts(records):
    by_case={}
    for r in records:by_case.setdefault(r['case_id'],{})[r['method']]=r
    require(all(set(pair)==set(METHODS) for pair in by_case.values()),'matched pair missing')
    return dict(pairs=len(by_case),
        wm_only_success=sum(p['vla_wm']['success'] and not p['vla_only']['success'] for p in by_case.values()),
        vla_only_success=sum(p['vla_only']['success'] and not p['vla_wm']['success'] for p in by_case.values()),
        unsafe_only_vla=sum(p['vla_only']['unsafe'] and not p['vla_wm']['unsafe'] for p in by_case.values()),
        unsafe_only_wm=sum(p['vla_wm']['unsafe'] and not p['vla_only']['unsafe'] for p in by_case.values()),
        unsafe_both=sum(p['vla_wm']['unsafe'] and p['vla_only']['unsafe'] for p in by_case.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('tmp/s4_5_final'))
    parser.add_argument('--execute',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--output',type=Path,default=Path('tmp/s4_6'))
    args=parser.parse_args();require(not args.output.exists(),'fresh output directory required')
    source_files=('contract.json','report.json','protocol.json','executions.npz')
    require(all((args.source/name).is_file() for name in source_files),'existing S4.5 source required')
    source_contract=json.loads((args.source/'contract.json').read_text());source_report=json.loads((args.source/'report.json').read_text())
    require(source_contract.get('schema_version')=='vla_world_model_feedback/1.0'
            and source_report.get('schema_version')=='vla_world_model_feedback_report/1.0','S4.5 source schema')
    weights,vocabulary,model,buffers,scale,strict,_,hashes=loop.load_system(Path(source_contract['gate']['source']))
    hashes={str(Path(name).resolve()):value for name,value in hashes.items()}
    expected={str(Path(name).resolve()):value for name,value in source_report['source_sha256'].items()}
    require(hashes==expected,'S4.5 frozen chain mismatch')
    hashes.update({str(args.source/name):loop.evaluation.sha256(args.source/name) for name in source_files})
    policy_path=next(Path(p) for p in hashes if Path(p).name=='model_contract.json')
    policy_contract=json.loads(policy_path.read_text())
    training_report=json.loads((policy_path.parent/'report.json').read_text())
    cases=fresh_protocol(policy_contract)
    args.output.mkdir(parents=True)
    # Persist the full fixed protocol before executing any measured case. No outcome-driven threshold changes.
    (args.output/'protocol.json').write_text(json.dumps(cases,indent=2)+'\n')
    protocol_hash=loop.evaluation.sha256(args.output/'protocol.json')
    for method in METHODS:
        for _ in range(5):run_episode(cases[0],method,weights,vocabulary,model,buffers,scale,strict,args.execute)
    records=[];arrays={}
    for i,case in enumerate(cases):
        # Alternate order within matched pairs to reduce fixed-order timing bias.
        for method in (METHODS if i%2==0 else METHODS[::-1]):
            row=run_episode(case,method,weights,vocabulary,model,buffers,scale,strict,args.execute)
            stem=method+'_'+case['case_id'];arrays[stem+'_state_m']=row.pop('state_m');arrays[stem+'_action_m']=row.pop('action_m')
            arrays[stem+'_timestamp_s']=np.arange(row['completed_steps']+1,dtype=np.float64)*.1
            records.append(row)
    require(hashes=={name:loop.evaluation.sha256(Path(name)) for name in hashes},'frozen source changed')
    require(protocol_hash==loop.evaluation.sha256(args.output/'protocol.json'),'benchmark protocol changed')
    summary={m:{c:summarize([r for r in records if r['method']==m and r['condition']==c]) for c in loop.CONDITIONS} for m in METHODS}
    per_seed={str(seed):{m:{c:summarize([r for r in records if r['seed']==seed and r['method']==m and r['condition']==c])
        for c in loop.CONDITIONS} for m in METHODS} for seed in SEEDS}
    paired={c:paired_counts([r for r in records if r['condition']==c]) for c in loop.CONDITIONS}
    np.savez_compressed(args.output/'executions.npz',**arrays)
    with np.load(args.output/'executions.npz',allow_pickle=False) as saved:
        for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
    contract=dict(schema_version='paired_vla_wm_benchmark/1.0',world_model_contract=loop.gate.scoring.wm.MODEL_CONTRACT,
        seeds=list(SEEDS),layout_groups=16,groups_per_seed=8,source_train_validation_excluded=40,
        matching='identical case/RGB/text/current-state interface, frozen VLA, E/H3/dt.1/budget8, plant gain/kick and outcome rules; independent reset per method',
        methods='vla_only executes learned candidate0; vla_wm adds four handwritten biases, frozen affine WM, perfect current RGB goal decoder, inherited gate',
        success='complete8 + final L-infinity goal distance<=.01m + no future sampled disk occupancy; no early stopping',
        risk='circle center [.015,-.09] radius .02m, sampled actual states1..8 only; no continuous collision or physical safety claim',
        prediction='executed-prefix RMSE on each own visited trajectory; baseline uses evaluation-only shadow WM AFTER controller stopwatch',
        rejection='VLA+WM only, full H3 counterfactual bank from actual pre-decision state with evaluator gain/kick, may extend beyond budget; bad=risk, good=risk-free AND rolling terminal success',
        timing='render/proposal/WM/gate + control total, diagnostic shadow/oracle separate, toy apply separate; five warmup episodes/method excluded',
        raw='per-case float32 state[T+1,2]/command[T,2] world XY[m], float64 clock[T+1]s; T0 refusals retained',
        scope='new layout pairings on same finite grid; fixed protocol seeds are not model training seeds or broad OOD evidence')
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    report=dict(schema_version='paired_vla_wm_benchmark_report/1.0',
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,numpy_path=np.__file__),
        execute=args.execute,protocol_sha256=protocol_hash,source_sha256=hashes,
        source_code_sha256={name:loop.evaluation.sha256(Path(__file__).parent/name) for name in
            ('s4_6_final_benchmark.py','s4_5_receding_horizon.py','s4_4_rejection_alternate.py','s4_3_risk_success_scoring.py','s4_2_world_model_verification.py','s4_1_vla_proposal.py')},
        seeds=dict(protocol=list(SEEDS),vla_training=training_report['seed'],vla_steps=training_report['steps'],wm_data=421,wm_split=422,calibration=431),
        frozen_gate=dict(buffers_m=buffers,scale=scale,require_success=strict),warmup_episodes_per_method=5,
        trained=False,summary=summary,per_seed=per_seed,paired=paired,records=records,
        limitations='Single failed tiny VLA checkpoint, only16 new object pairings on seen coordinate grid. Added goal decoder/biases confound attribution to WM alone. Own-visited-state metrics and repeated conditions/candidates are correlated. Empirical buffers not recalibrated for new states. Shadow/oracle diagnostics excluded from controller timing, not deployable control. No robot/generalization/safety/realtime or statistically robust multi-training-seed claim.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['method,condition,attempts,successes,goal_successes,unsafe_episodes,accepted_calls,rmse_m,control_p95_ms']
    for method,groups in summary.items():
        for condition,s in groups.items():
            latency=s['latency_ms']['control_ms']
            lines.append(','.join(map(str,[method,condition,*[s[k] for k in ('attempts','successes','goal_successes','unsafe_episodes','accepted_calls','executed_prediction_rmse_m')],None if latency is None else latency['p95']])))
    (args.output/'summary.csv').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(execute=args.execute,summary=summary,paired=paired),indent=2))


if __name__=='__main__':main()
