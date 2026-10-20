"""Frozen VLA -> model -> gate -> execute prefix -> observe; prediction-only feedback ablation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require
import s1_7_closed_loop_evaluation as evaluation
import s4_1_vla_proposal as proposal
import s4_4_rejection_alternate as gate

BUDGET = 8
DT = .1
CONDITIONS = ('nominal','gain065','kick','unsupported_text')
METHODS = ('observed_state','predicted_state')


def load_system(directory):
    require(all((directory/name).is_file() for name in ('contract.json','report.json','decisions.npz')), 'existing S4.4 source required')
    contract=json.loads((directory/'contract.json').read_text())
    report=json.loads((directory/'report.json').read_text())
    require(contract.get('schema_version')=='chunk_rejection_alternate/1.0'
            and contract.get('world_model_contract')==gate.scoring.wm.MODEL_CONTRACT, 'S4.4 plant/version mismatch')
    require(report.get('schema_version')=='chunk_rejection_alternate_report/1.0', 'S4.4 report version')
    scores_dir=Path(report['scores_source'])
    require(report['source_sha256']=={name:evaluation.sha256(scores_dir/name) for name in gate.FILES}, 'S4.3 source mismatch')
    _,buffers,score_report,verification,proposal_dir=gate.load_scores(scores_dir)
    require(buffers==report['calibration_buffers_m'] and report['margin_scale']==score_report['margin_scale'], 'gate thresholds changed')
    model,_,_,_,_=gate.scoring.load_source(verification)
    proposal_report=json.loads((proposal_dir/'report.json').read_text())
    checkpoint=Path(proposal_report['checkpoint'])
    require(proposal_report['source_sha256']=={name:evaluation.sha256(checkpoint/name) for name in evaluation.SOURCE_FILES}, 'VLA checkpoint mismatch')
    weights,vocabulary,policy_contract=evaluation.load_checkpoint(checkpoint)
    tracked={str(path):evaluation.sha256(path) for folder,names in
        [(directory,('contract.json','report.json','decisions.npz')),(scores_dir,gate.FILES),
         (verification,gate.scoring.FILES),(proposal_dir,gate.scoring.wm.SOURCE_FILES),(checkpoint,evaluation.SOURCE_FILES)]
        for path in (folder/name for name in names)}
    return weights,vocabulary,model,buffers,report['margin_scale'],report['require_success'],policy_contract['split']['validation'],tracked


def protocol(scenes):
    cases=[]
    for condition in CONDITIONS:
        for scene in scenes:
            centers=np.array(scene['centers_uv'],np.int64)
            for color,text in enumerate(proposal.vla.LANGUAGES):
                goal=proposal.vla.world(centers[color]).tolist()
                cases.append(dict(case_id=f'{condition}_{scene["scene_id"]}_{color}',condition=condition,
                    scene_id=scene['scene_id'],centers_uv=scene['centers_uv'],
                    text=f'grab {"red" if color==0 else "blue"} block' if condition=='unsupported_text' else text,
                    initial_m=[0.,-.12],evaluation_goal_m=goal,
                    plant_gain=.65 if condition=='gain065' else 1.,
                    kick_after_transition=2,kick_m=[.12,0.] if condition=='kick' else [0.,0.]))
    return cases


def run_episode(case,weights,vocabulary,model,buffers,scale,execute,feedback,require_success=False):
    require(type(execute) is int and 1<=execute<=3, '1<=E<=H3')
    position=np.array(case['initial_m'],np.float32)
    belief=position.copy();trace=[position.copy()];actions=[];plans=[];t=0;failure=None
    while t<BUDGET:
        planning_state=position.copy() if feedback else belief.copy()
        start=perf_counter_ns()
        rgb=proposal.vla.render(np.array(case['centers_uv'],np.int64))
        render_end=perf_counter_ns()
        try:
            bank=proposal.propose(rgb,planning_state,case['text'],weights,vocabulary,5)
        except ValueError as error:
            failure='policy_input_or_output_rejection'
            plans.append(dict(t=t,planning_state_m=planning_state.tolist(),executed_prefix=0,
                error=str(error),render_ms=(render_end-start)/1e6,
                rejected_attempt_ms=(perf_counter_ns()-start)/1e6))
            break
        proposal_end=perf_counter_ns()
        initial=np.broadcast_to(planning_state,(1,5,2))
        predictions=gate.scoring.wm.imagine(initial,bank[None],model)
        model_end=perf_counter_ns()
        # Goal comes only from current RGB + literal text, separate from evaluator annotation.
        goal=gate.scoring.observation_goals(dict(image=rgb[None],text=np.array([case['text']])))
        scores=gate.scoring.margins(predictions,goal)
        chosen,rejected=gate.select(scores,buffers,scale,require_success)
        chunk=gate.selected_commands(bank[None],chosen)[0]
        count=min(execute,BUDGET-t)
        selected_prediction=predictions[0,chosen[0]] if chosen[0]>=0 else gate.scoring.wm.imagine(
            planning_state[None],chunk[None],model)[0]
        planning_end=perf_counter_ns()
        # Update the ablation's belief only from its own selected predicted prefix endpoint.
        belief=selected_prediction[count].astype(np.float32)
        before=t;prefix_actual=[]
        apply_start=perf_counter_ns()
        for offset in range(count):
            position=(position+np.float32(case['plant_gain'])*chunk[offset]).astype(np.float32)
            t+=1
            if t==case['kick_after_transition']:
                position=(position+np.array(case['kick_m'],np.float32)).astype(np.float32)
            actions.append(chunk[offset].copy());trace.append(position.copy());prefix_actual.append(position.copy())
        apply_ms=(perf_counter_ns()-apply_start)/1e6
        prefix_error=np.array(prefix_actual,np.float64)-selected_prediction[1:count+1]
        plans.append(dict(t=before,planning_state_m=planning_state.tolist(),candidate_bank_m=bank.tolist(),
            predicted_bank_m=predictions[0].tolist(),scores_m={key:value[0].tolist() for key,value in scores.items()},
            rejected=rejected[0].tolist(),chosen=int(chosen[0]),selected_chunk_m=chunk.tolist(),
            selected_prediction_m=selected_prediction.tolist(),executed_prefix=count,
            predicted_executed_m=selected_prediction[1:count+1].tolist(),actual_executed_m=np.array(prefix_actual).tolist(),
            prefix_prediction_rmse_m=float(np.sqrt(np.mean(prefix_error**2))),
            render_ms=(render_end-start)/1e6,proposal_ms=(proposal_end-render_end)/1e6,
            world_model_ms=(model_end-proposal_end)/1e6,score_gate_ms=(planning_end-model_end)/1e6,
            planning_ms=(planning_end-start)/1e6,apply_ms=apply_ms,nominal_prefix_budget_ms=count*DT*1000))
    actual=np.array(trace,np.float32);commands=np.array(actions,np.float32).reshape(-1,2)
    future=actual[1:].astype(np.float64)
    unsafe=False if not len(future) else bool((np.linalg.norm(future-gate.scoring.CENTER,axis=-1)<=gate.scoring.RADIUS).any())
    final_error=float(np.max(np.abs(actual[-1].astype(np.float64)-case['evaluation_goal_m'])))
    goal_success=t==BUDGET and final_error<=gate.scoring.TOLERANCE
    success=goal_success and not unsafe and failure is None
    if failure is None and not success:failure='monitoring_risk' if unsafe else 'final_goal_error_after_budget'
    return dict(case_id=case['case_id'],condition=case['condition'],scene_id=case['scene_id'],feedback=feedback,
        completed_steps=t,success=success,goal_success=goal_success,unsafe=unsafe,failure_stage=failure,
        final_goal_linf_m=final_error,plans=plans,state_m=actual,action_m=commands)


def summarize(records):
    accepted=[p for row in records for p in row['plans'] if p['executed_prefix']>0]
    rejected=[p for row in records for p in row['plans'] if p['executed_prefix']==0]
    def timing(key):
        values=[p[key] for p in accepted]
        return None if not values else dict(samples=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)))
    scalar_count=sum(2*p['executed_prefix'] for p in accepted)
    squared=sum(p['prefix_prediction_rmse_m']**2*2*p['executed_prefix'] for p in accepted)
    return dict(attempts=len(records),completed=sum(r['completed_steps']==BUDGET for r in records),
        successes=sum(r['success'] for r in records),goal_successes=sum(r['goal_success'] for r in records),
        unsafe_episodes=sum(r['unsafe'] for r in records),failure_stages=dict(Counter(r['failure_stage'] for r in records if r['failure_stage'])),
        accepted_plans=len(accepted),rejected_attempts=len(rejected),fallback_plans=sum(p['chosen']<0 for p in accepted),
        alternate_plans=sum(p['chosen']>0 for p in accepted),primary_plans=sum(p['chosen']==0 for p in accepted),
        final_goal_linf_median_m=float(np.median([r['final_goal_linf_m'] for r in records])),
        executed_prediction_scalar_denominator=scalar_count,
        executed_prediction_rmse_m=None if scalar_count==0 else float(np.sqrt(squared/scalar_count)),
        latency_ms={key:timing(key) for key in ('render_ms','proposal_ms','world_model_ms','score_gate_ms','planning_ms','apply_ms')},
        per_episode_planning_ms=None if not accepted else dict(median=float(np.median([sum(p['planning_ms'] for p in r['plans'] if p['executed_prefix']>0) for r in records])),
            p95=float(np.percentile([sum(p['planning_ms'] for p in r['plans'] if p['executed_prefix']>0) for r in records],95))),
        compute_overruns=sum(p['planning_ms']>p['nominal_prefix_budget_ms'] for p in accepted),
        rejection_latency_ms=None if not rejected else dict(samples=len(rejected),median=float(np.median([p['rejected_attempt_ms'] for p in rejected]))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate',type=Path,default=Path('tmp/s4_4_final'))
    parser.add_argument('--execute',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--output',type=Path,default=Path('tmp/s4_5'))
    args=parser.parse_args()
    require(not args.output.exists(),'fresh output directory required')
    weights,vocabulary,model,buffers,scale,strict,scenes,hashes=load_system(args.gate)
    cases=protocol(scenes)
    # Five complete warmup episodes, excluded from all reported records/timings.
    for _ in range(5):run_episode(cases[0],weights,vocabulary,model,buffers,scale,args.execute,True,strict)
    records=[];arrays={}
    for method in METHODS:
        for case in cases:
            row=run_episode(case,weights,vocabulary,model,buffers,scale,args.execute,method=='observed_state',strict)
            row['method']=method
            stem=method+'_'+case['case_id']
            arrays[stem+'_state_m']=row.pop('state_m');arrays[stem+'_action_m']=row.pop('action_m')
            arrays[stem+'_timestamp_s']=np.arange(row['completed_steps']+1,dtype=np.float64)*DT
            records.append(row)
    require(hashes=={path:evaluation.sha256(Path(path)) for path in hashes},'frozen upstream changed')
    summary={method:{condition:summarize([r for r in records if r['method']==method and r['condition']==condition])
        for condition in CONDITIONS} for method in METHODS}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'executions.npz',**arrays)
    with np.load(args.output/'executions.npz',allow_pickle=False) as saved:
        for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
    contract=dict(schema_version='vla_world_model_feedback/1.0',world_model_contract=gate.scoring.wm.MODEL_CONTRACT,
        state='float32 [T+1,2] world XY measured positions[m]; action float32 [T,2] world XY commands[m]',
        time='float64 [T+1] simulated seconds, dt=.1s; fixed 8-transition budget, no early success stop',
        proposal='frozen S1.5 VLA + fixed S4.1 biases, K5/H3 every decision; no warm suffix or retraining',
        prediction='float64 [5,4,2] from planning state; fixed H3 including final unexecuted suffix beyond budget',
        execution='min(E,8-t) commands then new decision, discard old suffix; full-H3 gate, no horizon truncation',
        feedback='observed_state refreshes exact current plant XY; predicted_state uses only model prefix endpoint cast float32',
        gate=dict(buffers_m=buffers,scale=scale,require_success=strict,source=str(args.gate.resolve())),
        success='completed8 AND final L-infinity error<=.01m AND no future sampled monitoring disk occupancy; goal_success reported separately',
        conditions='nominal gain1; hidden gain.65; hidden +.12m world-x kick after transition2; unsupported literal input rejection',
        risk='sampled actual h1..8 disk occupancy, initial excluded; not continuous collision or real safety',
        scope='single-checkpoint feedback/execute-prefix learning experiment, not final matched VLA-only vs VLA+WM benchmark')
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (args.output/'protocol.json').write_text(json.dumps(cases,indent=2)+'\n')
    report=dict(schema_version='vla_world_model_feedback_report/1.0',
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,numpy_path=np.__file__),
        execute=args.execute,source_sha256=hashes,
        source_code_sha256={name:evaluation.sha256(Path(__file__).parent/name) for name in
            ('s4_5_receding_horizon.py','s4_4_rejection_alternate.py','s4_3_risk_success_scoring.py','s4_2_world_model_verification.py','s4_1_vla_proposal.py')},
        warmup_episodes=5,trained=False,summary=summary,records=records,
        limitations='Old eight validation layouts, single failed VLA, ideal static RGB/exact state. Buffers calibrated on different probe states/gains, not newly visited closed-loop states. Full H3 score may gate unexecuted tail. No probability/safety/generalization/realtime guarantee. Prediction-only ablation is not true feedback. No final benchmark.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(execute=args.execute,summary=summary),indent=2))


if __name__=='__main__':main()
