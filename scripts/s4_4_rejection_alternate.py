"""Frozen score gate, primary-preserving alternatives and explicit zero fallback; one H3 replay."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require
from s1_7_closed_loop_evaluation import sha256
import s4_3_risk_success_scoring as scoring

FILES = ('contract.json', 'report.json', 'scores.npz')
SETS = ('heldout_id', 'heldout_gain065', 'vla_nominal', 'vla_gain_shift')


def select(scores, buffers, scale, require_success=False, alternate=True):
    """No actual labels/truth/gain inputs. Candidate0 has priority if it passes."""
    require(set(scores) == {'risk', 'success'}, 'two geometric scores required')
    require(scores['risk'].ndim == 2 and scores['success'].shape == scores['risk'].shape
            and scores['risk'].shape[1] > 0, 'nonempty [context,candidate] scores')
    require(all(np.isfinite(value).all() for value in scores.values()), 'finite scores')
    flags = scoring.decisions(scores, buffers, scale)
    rejected = flags['risk'] | (~flags['success'] if require_success else False)
    chosen = np.full(len(rejected), -1, np.int64)
    for i in range(len(chosen)):
        if not rejected[i, 0]:
            chosen[i] = 0
        elif alternate:
            eligible = np.flatnonzero(~rejected[i])
            if len(eligible):
                chosen[i] = eligible[np.argmax(scores['success'][i, eligible])]
    return chosen, rejected


def selected_commands(bank, chosen):
    require(bank.ndim == 4 and bank.shape[-2:] == (3, 2) and bank.dtype == np.float32
            and np.isfinite(bank).all() and np.all(np.abs(bank) <= np.float32(.06)), 'physical float32 command bank')
    require(chosen.shape == (len(bank),) and chosen.dtype == np.int64
            and np.all((chosen >= -1) & (chosen < bank.shape[1])), 'candidate indices or -1 fallback')
    result = np.zeros((len(bank), 3, 2), np.float32)
    rows = np.flatnonzero(chosen >= 0)
    result[rows] = bank[rows, chosen[rows]]  # -1 must NEVER accidentally select the last candidate.
    return result


def rejection_counts(rejected, actual_risk, actual_success):
    require(rejected.shape == actual_risk.shape == actual_success.shape
            and rejected.dtype == actual_risk.dtype == actual_success.dtype == np.bool_, 'matching boolean rejection/labels')
    good = ~actual_risk & actual_success
    return dict(candidates=int(rejected.size), rejected=int(rejected.sum()),
        actual_bad=int(actual_risk.sum()), rejected_bad=int((rejected & actual_risk).sum()),
        allowed_bad=int((~rejected & actual_risk).sum()),
        actual_risk_free=int((~actual_risk).sum()), rejected_risk_free=int((rejected & ~actual_risk).sum()),
        actual_good=int(good.sum()), false_rejected_good=int((rejected & good).sum()),
        allowed_good=int((~rejected & good).sum()))


def replay(initial, commands, gains):
    return np.stack([scoring.wm.plant_rollout(initial[i], commands[i], float(gain))
                     for i, gain in enumerate(gains)])


def load_scores(directory):
    require(all((directory/name).is_file() for name in FILES), 'need existing S4.3 score artifacts')
    contract = json.loads((directory/'contract.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    require(contract.get('schema_version') == 'risk_success_margins/1.0'
            and contract.get('world_model_contract') == scoring.wm.MODEL_CONTRACT
            and contract.get('coordinate_frame') == 'world_xy' and contract.get('unit') == 'm'
            and contract.get('dt_s') == .1 and contract.get('horizon') == 3, 'score contract semantics')
    require(contract['risk']['center_m'] == scoring.CENTER.tolist()
            and contract['risk']['radius_m'] == scoring.RADIUS
            and contract['success']['tolerance_m'] == scoring.TOLERANCE, 'score geometry mismatch')
    require(report.get('schema_version') == 'risk_success_scoring_report/1.0', 'score report version')
    verification = Path(report['verification'])
    require(report['source_sha256'] == {name:sha256(verification/name) for name in scoring.FILES}, 'S4.2 provenance mismatch')
    model, source, proposal, proposal_report, proposal_dir = scoring.load_source(verification)
    require(report['proposal_sha256'] == {name:sha256(proposal_dir/name) for name in scoring.wm.SOURCE_FILES}, 'proposal provenance mismatch')
    with np.load(directory/'scores.npz', allow_pickle=False) as saved:
        arrays = {key:saved[key] for key in saved.files}
    buffers = report['calibration_buffers_m']
    require(set(buffers) == {'risk','success'} and all(np.isfinite(v) and v >= 0 for v in buffers.values()), 'valid frozen buffers')
    cal_goal = arrays['calibration_goal']
    cal_pred = scoring.margins(arrays['calibration_prediction_m'], cal_goal)
    cal_actual = scoring.margins(arrays['calibration_truth_m'], cal_goal)
    require(buffers == scoring.calibrate(cal_pred, cal_actual), 'saved buffers do not match calibration')
    require(set(arrays['calibration_group_id']).isdisjoint(arrays['heldout_group_id']), 'calibration/heldout group overlap')
    contexts = {}
    for name in SETS:
        if name.startswith('heldout'):
            bank = arrays['heldout_action']; initial = arrays['heldout_initial']
            gains = arrays['heldout_gain'] if name == 'heldout_id' else np.full(len(bank), .65)
        else:
            bank = source['action_m']; initial = source['initial_state_m']
            gain = 1. if name == 'vla_nominal' else json.loads((verification/'contract.json').read_text())['conditions']['shifted_gain']
            gains = np.full(len(bank), gain)
        goal = arrays[name+'_goal_m']
        predicted = arrays[name+'_prediction_m']; truth = arrays[name+'_truth_m']
        require(predicted.shape == (*bank.shape[:2],4,2) and predicted.dtype == np.float64
                and truth.shape == predicted.shape and truth.dtype == np.float32, 'future shape/dtype')
        bank_initial = np.broadcast_to(initial[:,None], (*bank.shape[:2],2))
        np.testing.assert_allclose(predicted,scoring.wm.imagine(bank_initial,bank,model),rtol=0,atol=1e-14)
        actual = np.stack([scoring.wm.plant_rollout(bank_initial[i],bank[i],float(g)) for i,g in enumerate(gains)])
        np.testing.assert_array_equal(truth,actual)
        if name.startswith('vla'):
            np.testing.assert_array_equal(goal,scoring.observation_goals(proposal))
        else:
            np.testing.assert_array_equal(goal,arrays['heldout_goal'])
        scores = scoring.margins(predicted,goal); actual_scores = scoring.margins(truth,goal)
        for key in scores:
            np.testing.assert_array_equal(arrays[name+'_'+key+'_score_m'],scores[key])
            np.testing.assert_array_equal(arrays[name+'_'+key+'_label'],actual_scores[key]>=0)
        contexts[name] = dict(bank=bank,initial=initial,goal=goal,gains=gains,scores=scores,
            actual_risk=actual_scores['risk']>=0,actual_success=actual_scores['success']>=0)
    for value in arrays.values(): value.flags.writeable=False
    return contexts,buffers,report,verification,proposal_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',type=Path,default=Path('tmp/s4_3_final_checked'))
    parser.add_argument('--require-success',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('tmp/s4_4'))
    args=parser.parse_args()
    require(not args.output.exists(),'fresh output directory required')
    contexts,buffers,source_report,verification,proposal_dir=load_scores(args.scores)
    hashes={name:sha256(args.scores/name) for name in FILES}
    upstream={str(directory/name):sha256(directory/name) for directory,names in
              [(verification,scoring.FILES),(proposal_dir,scoring.wm.SOURCE_FILES)] for name in names}
    scale=source_report['margin_scale']
    output={'timestamp_s':np.arange(4,dtype=np.float64)*.1};summary={}
    for name,case in contexts.items():
        scores=case['scores']; bank=case['bank']; goal=case['goal']
        # All choices are made from predicted margins BEFORE replay evaluation; selector takes no truth.
        chosen,rejected=select(scores,buffers,scale,args.require_success,True)
        reject_only,_=select(scores,buffers,scale,args.require_success,False)
        choices=dict(candidate0=np.zeros(len(bank),np.int64),reject_only=reject_only,alternate=chosen)
        count=rejection_counts(rejected,case['actual_risk'],case['actual_success'])
        primary=rejection_counts(rejected[:,:1],case['actual_risk'][:,:1],case['actual_success'][:,:1])
        output[name+'_bank_m']=bank; output[name+'_initial_m']=case['initial'];output[name+'_goal_m']=goal
        output[name+'_plant_gain']=case['gains'];output[name+'_rejected']=rejected
        for key,value in scores.items(): output[name+'_'+key+'_score_m']=value
        output[name+'_actual_risk']=case['actual_risk'];output[name+'_actual_success']=case['actual_success']
        methods={}
        for method,indices in choices.items():
            commands=selected_commands(bank,indices)
            trajectory=replay(case['initial'],commands,case['gains'])
            actual=scoring.margins(trajectory[:,None],goal)
            risk=actual['risk'][:,0]>=0;success=actual['success'][:,0]>=0;fallback=indices<0
            output[name+'_'+method+'_chosen']=indices;output[name+'_'+method+'_command_m']=commands
            output[name+'_'+method+'_truth_m']=trajectory
            methods[method]=dict(attempts=len(bank),primary_kept=int((indices==0).sum()),
                alternatives=int((indices>0).sum()),fallbacks=int(fallback.sum()),
                unsafe_chunks=int(risk.sum()),goal_successes=int(success.sum()),
                risk_free_goal_successes=int((~risk & success).sum()),
                fallback_unsafe=int((fallback & risk).sum()),fallback_goal_successes=int((fallback & success).sum()),
                failure_stages=dict(unsafe=int(risk.sum()),risk_free_goal_miss=int((~risk & ~success).sum())),
                final_goal_linf_median_m=float(np.median(np.abs(trajectory[:,-1].astype(np.float64)-goal).max(-1))))
        for _ in range(5): select(scores,buffers,scale,args.require_success,True)
        times=[]
        for _ in range(30):
            start=perf_counter_ns();select(scores,buffers,scale,args.require_success,True)
            times.append((perf_counter_ns()-start)/1e6)
        summary[name]=dict(rejection=count,primary_rejection=primary,methods=methods,
            gate_latency_ms=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),samples=30,warmup=5,
                scope='all-context flags + candidate selection only; excludes WM, geometric scores, commands, plant, IO'))
    require(hashes=={name:sha256(args.scores/name) for name in FILES},'scores source changed')
    require(upstream=={name:sha256(Path(name)) for name in upstream},'upstream source changed')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'decisions.npz',**output)
    with np.load(args.output/'decisions.npz',allow_pickle=False) as saved:
        for key,value in output.items(): np.testing.assert_array_equal(saved[key],value)
    contract=dict(schema_version='chunk_rejection_alternate/1.0',world_model_contract=scoring.wm.MODEL_CONTRACT,
        action='float32 [G,3,2] commanded world XY displacement[m], dt=.1s, per-component limit .06m',
        trajectory='float32 [G,4,2] including initial and three actual next states; every method independently resets',
        gate='reject risk flag; optionally also reject missing success flag; frozen S4.3 buffers/scale',
        selection='keep candidate0 if eligible; otherwise eligible argmax predicted success margin, tie lowest index; else -1',
        fallback='-1 means explicit three zero commands, not last-bank index; no safety or success guarantee',
        good_action='actual risk-free AND final goal L-infinity distance <= .01m; bad_action=actual risk event',
        scope='single decision then H3 open-loop execution, no observe/replan or closed-loop benchmark',
        limitations='discrete future disk monitoring only, initial excluded, no continuous collision or robot guarantees')
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    report=dict(schema_version='chunk_rejection_alternate_report/1.0',
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,numpy_path=np.__file__),
        scores_source=str(args.scores.resolve()),source_sha256=hashes,upstream_sha256=upstream,
        source_code_sha256={name:sha256(Path(__file__).parent/name) for name in
            ('s4_4_rejection_alternate.py','s4_3_risk_success_scoring.py','s4_2_world_model_verification.py')},
        require_success=args.require_success,margin_scale=scale,calibration_buffers_m=buffers,summary=summary,
        proposal_attempts=source_report['proposal_attempts'],supported_inputs=source_report['supported_inputs'],
        input_rejections=source_report['input_rejections'],trained_during_evaluation=False,closed_loop_success=None,
        limitations='Reused heldout diagnostic contexts and old VLA layouts, not a new blind test. Synthetic probe successes are not VLA successes. Risk-free rejected actions may still miss the goal; zero fallback may remain in disk. Whole H3 risk versus single H3 execution only; no receding-horizon or real-robot result.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['set,method,attempts,primary_kept,alternatives,fallbacks,unsafe_chunks,goal_successes,risk_free_goal_successes,fallback_unsafe']
    for name,group in summary.items():
        for method,m in group['methods'].items():
            lines.append(','.join(map(str,[name,method,*[m[k] for k in lines[0].split(',')[2:]]])))
    (args.output/'summary.csv').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(require_success=args.require_success,summary=summary),indent=2))


if __name__=='__main__':main()
