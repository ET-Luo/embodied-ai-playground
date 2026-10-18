"""Frozen S1.5 policy: explicit closed-loop protocol, visited-state error and latency scopes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require, toy_episode, validate
from s0_2_sequence_dataset import masked_mse
from s0_5_action_representation import expect_rejection, round_trip
import s1_5_tiny_transformer_vla as vla
from s1_3_visual_encoder import png

DT = .1
BUDGET = 8
TOLERANCE_M = .01
POLICIES = ('learned', 'oracle', 'zero')
SOURCE_FILES = ('weights.npz', 'tokenizer.json', 'model_contract.json', 'validation_chunks.npz', 'report.json')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(directory):
    require(directory.is_dir() and all((directory/name).is_file() for name in SOURCE_FILES),
            'need frozen S1.5 checkpoint; run S1.5 into a new directory first, or select --checkpoint')
    tokenizer = json.loads((directory/'tokenizer.json').read_text())
    contract = json.loads((directory/'model_contract.json').read_text())
    require(contract.get('schema_version') == 'tiny_patch_vla_contract/1.0', 'S1.5 model contract version')
    require(tokenizer.get('vocabulary') == vla.fit_vocabulary(vla.LANGUAGES)
            and tokenizer.get('known_instructions') == list(vla.LANGUAGES), 'incompatible language encoding')
    vocabulary = tokenizer['vocabulary']
    with np.load(directory/'weights.npz', allow_pickle=False) as saved:
        weights = {key:saved[key] for key in saved.files}
    expected = vla.initialize(len(vocabulary), 0)
    require(weights.keys() == expected.keys(), 'checkpoint parameter keys')
    for name, value in weights.items():
        require(value.shape == expected[name].shape and value.dtype == np.float32 and np.isfinite(value).all(),
                f'checkpoint shape/dtype/finite: {name}')
        value.flags.writeable = False
    train_ids = {scene['scene_id'] for scene in contract['split']['train']}
    scenes = contract['split']['validation']
    require(len(scenes) == 8 and len({s['scene_id'] for s in scenes}) == 8 and not train_ids & {s['scene_id'] for s in scenes},
            'expected eight disjoint validation layouts')
    for scene in scenes:
        vla.render(np.array(scene['centers_uv'], dtype=np.int64))
    return weights, vocabulary, contract


def protocol(scenes, execute):
    cases = []
    for condition in ('nominal', 'disturbed', 'shifted_start', 'unsupported_text'):
        for scene in scenes:
            for color, language in enumerate(vla.LANGUAGES):
                text = f'grab {"red" if color == 0 else "blue"} block' if condition == 'unsupported_text' else language
                cases.append(dict(case_id=f'{condition}_{scene["scene_id"]}_{color}', condition=condition,
                    scene_id=scene['scene_id'], centers_uv=scene['centers_uv'], evaluator_color=color, text=text,
                    start_m=[.18, .12] if condition == 'shifted_start' else [0, -.12],
                    kick_x_m=.12 if condition == 'disturbed' else 0., kick_after_transition=2))
    return dict(schema_version='closed_loop_protocol/1.0', horizon=vla.HORIZON, execute=execute, dt_s=DT,
        budget_steps=BUDGET, success='final max XY error <= .01m; final state only, no early stopping',
        tolerance_m=TOLERANCE_M, policies=POLICIES, cases=cases,
        action_contract='executed world XY displacement[m], per-component clip +/-.06m; ideal point plus declared kick',
        groups='same eight layout groups across conditions; 16 attempts per policy/condition, not independent across groups',
        unsupported='literal S1.5 tokenizer rejection; no S1.6 aliasing or replacement policy',
        comparison='oracle gets evaluator goal; zero ignores language; these are controls, not learned VLA alternatives')


def validate_chunk(chunk):
    require(chunk.shape == (vla.HORIZON, 2) and chunk.dtype == np.float32 and np.isfinite(chunk).all(),
            'finite H3x2 float32 chunk')
    require(np.all(np.abs(chunk) <= np.float32(.06)+1e-8), 'executed chunk exceeds physical displacement limit')


def timing_summary(samples):
    return None if not samples else dict(samples=len(samples), median=float(np.median(samples)),
        p95=float(np.percentile(samples, 95)), maximum=float(np.max(samples)))


def run_case(case, policy, weights, vocabulary, execute):
    centers = np.array(case['centers_uv'], dtype=np.int64)
    goal = vla.world(centers[case['evaluator_color']])
    position = np.array(case['start_m'], dtype=np.float32)
    trace = [position.copy()]; actions = []; expert_actions = []; decisions = []; timings = []
    distances = [float(np.max(np.abs(position-goal)))]
    stage = None; message = None; tick = 0
    while tick < BUDGET:
        loop_start = perf_counter_ns()
        rgb = vla.render(centers)
        render_end = perf_counter_ns()
        policy_start = perf_counter_ns()
        try:
            if policy == 'learned':
                chunk = vla.predict(rgb, position.copy(), case['text'], weights, vocabulary)
            elif policy == 'oracle':
                simulated = position.copy(); proposed = []
                for _ in range(vla.HORIZON):
                    action = np.clip(goal-simulated, -vla.STEP, vla.STEP)
                    proposed.append(action); simulated += action
                chunk = np.stack(proposed)
            elif policy == 'zero':
                chunk = np.zeros((vla.HORIZON, 2), dtype=np.float32)
            else:
                raise RuntimeError('unknown evaluator policy')
        except ValueError as error:
            stage = 'policy_input_rejection'; message = str(error); chunk = None
        policy_end = perf_counter_ns()
        if stage is None:
            try:
                validate_chunk(chunk)
            except ValueError as error:
                stage = 'policy_output_contract'; message = str(error)
        count = min(execute, BUDGET-tick) if stage is None else 0
        decisions.append(dict(observation_tick=tick, state_m=position.tolist(), executed_prefix=count,
                              chunk_m=None if stage else chunk.tolist()))
        apply_ms = []
        for offset in range(count):
            # Counterfactual expert is evaluated at the ACTUAL pre-action state, even for stale prefixes.
            expert_actions.append(np.clip(goal-position, -vla.STEP, vla.STEP))
            action = chunk[offset].copy()
            apply_start = perf_counter_ns()
            position += action; tick += 1
            if tick == case['kick_after_transition']:
                position[0] += case['kick_x_m']
            apply_ms.append((perf_counter_ns()-apply_start)/1e6)
            actions.append(action); trace.append(position.copy())
            distances.append(float(np.max(np.abs(position-goal))))
        loop_ms = (perf_counter_ns()-loop_start)/1e6
        timings.append(dict(render_ms=(render_end-loop_start)/1e6, policy_ms=(policy_end-policy_start)/1e6,
            apply_ms=sum(apply_ms), local_loop_ms=loop_ms, accepted=stage is None,
            nominal_feedback_interval_ms=1000*DT*count if count else None))
        if stage is not None: break
    error = distances[-1]
    success = stage is None and tick == BUDGET and error <= TOLERANCE_M
    if not success and stage is None: stage = 'final_goal_error_after_budget'
    visited_mse = None if not actions else float(np.mean((np.stack(actions).astype(np.float64)-np.stack(expert_actions))**2))
    return dict(case_id=case['case_id'], condition=case['condition'], policy=policy, text=case['text'],
        success=success, failure_stage=stage, failure_message=message, completed_steps=tick,
        truncated=tick == BUDGET, final_error_m=error, minimum_error_m=min(distances),
        ever_in_goal=any(d <= TOLERANCE_M for d in distances),
        first_goal_tick=next((i for i,d in enumerate(distances) if d <= TOLERANCE_M), None),
        goal_world_m=goal.tolist(), trace=np.stack(trace).tolist(), commands=np.asarray(actions).reshape(-1,2).tolist(),
        evaluator_expert_commands=np.asarray(expert_actions).reshape(-1,2).tolist(),
        visited_state_action_mse_m2=visited_mse, decisions=decisions, timings=timings, policy_attempts=len(decisions))


def export_episode(directory, record, case):
    if record['completed_steps'] == 0: return None
    length = record['completed_steps']
    arrays, meta = toy_episode(1)
    arrays.update(proprioception=np.array(record['trace'], dtype=np.float32), action=np.array(record['commands'], dtype=np.float32),
        image=np.repeat(vla.render(np.array(case['centers_uv'], dtype=np.int64))[None], length+1, axis=0),
        timestamp=np.arange(length+1, dtype=np.float64)*DT,
        terminated=np.array([False]*(length-1)+[not record['truncated']]),
        truncated=np.array([False]*(length-1)+[record['truncated']]))
    meta.update(episode_id=f'{record["policy"]}_{case["case_id"]}', language_instruction=case['text'],
        task_name='frozen_patch_vla_evaluation', embodiment='ideal_world_xy_patch_grid_v1', success=record['success'],
        success_definition='final max world XY error to original selected goal <= .01m after 8 commands')
    meta['provenance'].update(source='S1.7 frozen checkpoint evaluator', group_id=case['scene_id'],
        controller='ideal displacement plus explicit external perturbation', condition=case['condition'],
        external_kick=dict(after_transition=case['kick_after_transition'], world_x_m=case['kick_x_m']),
        failure_stage=record['failure_stage'])
    for key, value in arrays.items():
        if key == 'image':
            meta['modalities'][key] = dict(shape=list(value.shape), dtype='uint8', color_space='RGB',
                semantic='static scene RGB, no agent marker', temporal='observation at t including final T',
                frame='world XY orthographic synthetic camera', calibration='u=16+x/.02;v=16-y/.02; patch-center squares',
                source='S1.5 renderer, no noise/delay')
        else:
            meta['modalities'][key]['shape'] = list(value.shape)
    validate(arrays, meta)
    target = directory/meta['episode_id']
    round_trip(target, arrays, meta)
    return str(target)


def summarize(records):
    errors = [record['final_error_m'] for record in records]
    actions = [np.asarray(row['commands'], dtype=np.float64).reshape(-1,2) for row in records if row['completed_steps']]
    expert = [np.asarray(row['evaluator_expert_commands'], dtype=np.float64).reshape(-1,2) for row in records if row['completed_steps']]
    samples = [sample for row in records for sample in row['timings']]
    policy_all = [t['policy_ms'] for t in samples]
    policy_accepted = [t['policy_ms'] for t in samples if t['accepted']]
    return dict(attempts=len(records), successes=sum(row['success'] for row in records),
        completed=sum(row['completed_steps'] == BUDGET for row in records),
        failure_stage_counts=dict(Counter(row['failure_stage'] for row in records if row['failure_stage'])),
        final_error_m=dict(median=float(np.median(errors)), maximum=max(errors)),
        ever_in_goal=sum(row['ever_in_goal'] for row in records),
        left_goal_after_visiting=sum(row['ever_in_goal'] and not row['success'] for row in records),
        visited_action_scalar_denominator=2*sum(row['completed_steps'] for row in records),
        visited_state_action_mse_m2=None if not actions else float(np.mean((np.concatenate(actions)-np.concatenate(expert))**2)),
        latency_ms=dict(policy_all_attempts=timing_summary(policy_all), policy_accepted=timing_summary(policy_accepted),
            render=timing_summary([t['render_ms'] for t in samples]),
            apply_prefix=timing_summary([t['apply_ms'] for t in samples if t['accepted']]),
            local_evaluator_loop=timing_summary([t['local_loop_ms'] for t in samples])),
        policy_calls_over_nominal_feedback_interval=sum(t['accepted'] and t['policy_ms'] > t['nominal_feedback_interval_ms'] for t in samples),
        timing_scope='wall compute only; loop includes evaluator diagnostics/logging, not sensor/plant/transport/wait; no real-time guarantee')


def offline_metrics(directory, weights, vocabulary):
    with np.load(directory/'validation_chunks.npz', allow_pickle=False) as saved:
        batch = {key:saved[key] for key in saved.files}
    require(np.array_equal(batch['word_ids'], vla.encode_text(batch['language_instruction'], vocabulary)), 'offline IDs mismatch')
    require(np.array_equal(batch['patches'], vla.image_tokens(batch['image'])), 'offline patch mismatch')
    require(np.array_equal(batch['action_mask'], batch['action_index'] >= 0), 'offline mask mismatch')
    prediction, _ = vla.forward(batch['patches'], batch['word_ids'], batch['observation'][:, -1]/np.float32(.3), weights)
    return dict(anchors=len(batch['anchor']), valid_slots=int(batch['action_mask'].sum()),
        scalar_denominator=2*int(batch['action_mask'].sum()),
        teacher_forced_raw_chunk_mse_m2=masked_mse(prediction*vla.STEP, batch['action'], batch['action_mask']),
        teacher_forced_executed_chunk_mse_m2=masked_mse(np.clip(prediction*vla.STEP,-vla.STEP,vla.STEP),batch['action'],batch['action_mask']),
        zero_action_chunk_mse_m2=masked_mse(np.zeros_like(prediction), batch['action'], batch['action_mask']),
        scope='expert-state anchors, future labels/masks; not visited-state rollout action MSE')


def checks(weights, vocabulary, scene):
    # Unsupported text is a real counted policy attempt with no fabricated zero-action episode.
    case = dict(case_id='input_check', condition='unsupported_text', scene_id=scene['scene_id'], centers_uv=scene['centers_uv'],
        evaluator_color=0, text='grab red block', start_m=[0,-.12], kick_x_m=0., kick_after_transition=2)
    rejected = run_case(case,'learned',weights,vocabulary,1)
    require(rejected['failure_stage'] == 'policy_input_rejection' and rejected['completed_steps'] == 0
        and not rejected['truncated'] and len(rejected['trace']) == 1, 'rejection denominator/boundary')
    names=[]
    for name, chunk in [('wrong_shape',np.zeros((1,2),np.float32)),('nonfinite',np.full((3,2),np.nan,np.float32)),
                        ('wrong_dtype',np.zeros((3,2),np.float64)),('over_limit',np.full((3,2),.07,np.float32))]:
        names.append(expect_rejection(name,lambda:validate_chunk(chunk)))
    # Goal annotation changes only evaluator diagnostics, never learned commands.
    case.update(text=vla.LANGUAGES[0],condition='goal_isolation')
    first = run_case(case,'learned',weights,vocabulary,1)
    changed = dict(case,evaluator_color=1)
    second = run_case(changed,'learned',weights,vocabulary,1)
    np.testing.assert_array_equal(first['commands'],second['commands'])
    np.testing.assert_array_equal(first['trace'],second['trace'])
    return dict(input_rejection_preserved=True, evaluator_goal_command_isolation=True, rejected_chunk_cases=names)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('tmp/s1_5_final'))
    parser.add_argument('--execute',type=int,choices=(1,3),default=1)
    parser.add_argument('--output',type=Path,default=Path('tmp/s1_7'))
    args=parser.parse_args()
    require(not args.output.exists(),'fresh output directory required')
    weights,vocabulary,contract=load_checkpoint(args.checkpoint)
    before={name:sha256(args.checkpoint/name) for name in SOURCE_FILES}
    plan=protocol(contract['split']['validation'],args.execute)
    verification=checks(weights,vocabulary,contract['split']['validation'][0])
    # Warm-up is explicitly excluded from all measured cases; no learning/update occurs.
    first=plan['cases'][0]
    for _ in range(5):
        vla.predict(vla.render(np.array(first['centers_uv'],dtype=np.int64)),np.array(first['start_m'],np.float32),vla.LANGUAGES[0],weights,vocabulary)
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    records=[]
    for policy in POLICIES:
        for case in plan['cases']:
            record=run_case(case,policy,weights,vocabulary,args.execute)
            record['raw_episode_artifact']=export_episode(args.output/'episodes',record,case)
            records.append(record)
    summary={policy:{condition:summarize([r for r in records if r['policy']==policy and r['condition']==condition])
                     for condition in ('nominal','disturbed','shifted_start','unsupported_text')} for policy in POLICIES}
    after={name:sha256(args.checkpoint/name) for name in SOURCE_FILES}
    require(before==after,'source checkpoint mutated during evaluation')
    training_report=json.loads((args.checkpoint/'report.json').read_text())
    evidence=dict(checkpoint=str(args.checkpoint.resolve()),files_sha256=before,
        training_metadata=dict(seed=training_report.get('seed'),steps=training_report.get('steps'),
                               source_schema=training_report.get('schema_version')),
        source_code_sha256={name:sha256(Path(__file__).parent/name) for name in
            ('s0_1_data_contract.py','s0_2_sequence_dataset.py','s0_5_action_representation.py',
             's1_3_visual_encoder.py','s1_4_multimodal_fusion.py','s1_5_tiny_transformer_vla.py','s1_7_closed_loop_evaluation.py')},
        parameters=sum(w.size for w in weights.values()),trained_during_evaluation=False,warmup_calls=5)
    (args.output/'source_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    report=dict(schema_version='frozen_closed_loop_evaluation/1.0',environment=dict(python=sys.version,executable=sys.executable,
        numpy=np.__version__,numpy_path=np.__file__),execute=args.execute,horizon=vla.HORIZON,checks=verification,
        source=evidence,offline=offline_metrics(args.checkpoint,weights,vocabulary),summary=summary,records=records,
        limitations='Fixed failed S1.5 model, same eight validation layouts already used before; no independent test/generalization or S1.6 bridge substitution. Visited states differ by policy, so their action MSEs have different input distributions. Stage labels are observed pipeline symptoms, not attribution to vision/language. Synthetic timing excludes real sensor/controller delays.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['policy,condition,successes,attempts,completed,visited_action_mse_m2,terminal_error_median_m,policy_accepted_p95_ms']
    for policy,groups in summary.items():
        for condition,s in groups.items():
            timing=s['latency_ms']['policy_accepted']
            lines.append(','.join(map(str,[policy,condition,s['successes'],s['attempts'],s['completed'],
                s['visited_state_action_mse_m2'],s['final_error_m']['median'],None if timing is None else timing['p95']])))
    (args.output/'summary.csv').write_text('\n'.join(lines)+'\n')
    png(args.output/'scenes.png',np.concatenate([np.repeat(np.repeat(vla.render(np.array(s['centers_uv'],np.int64)),4,0),4,1)
        for s in contract['split']['validation'][:4]],axis=1))
    print(json.dumps(dict(execute=args.execute,offline=report['offline'],
        successes={p:{c:f"{s['successes']}/{s['attempts']}" for c,s in groups.items()} for p,groups in summary.items()},checks=verification),indent=2))
    print(f'PASS frozen source hashes / rejected attempts / goal isolation / raw episode round-trip; {args.output}')


if __name__=='__main__':
    main()
