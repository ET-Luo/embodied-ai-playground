"""S0.7: causal RGB/language/state -> rule-generated chunks, masks and execution prefix."""
import argparse
import copy
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
import PIL
from PIL import Image, ImageDraw

from s0_1_data_contract import require, toy_episode, validate
from s0_2_sequence_dataset import audit_windows, make_windows, masked_mse
from s0_5_action_representation import expect_rejection, round_trip

PIXEL_M = .02
GOALS = {'red': np.array([.24, 0], dtype=np.float32), 'blue': np.array([-.24, 0], dtype=np.float32)}
COLORS = {'red': [255, 0, 0], 'blue': [0, 0, 255]}
STEP_M = .06


def render(position):
    """Explicit orthographic raster; goals remain visible even when agent overlaps."""
    image = np.full((32,32,3), 10, dtype=np.uint8)
    u, v = int(np.rint(position[0]/PIXEL_M+16)), int(np.rint(16-position[1]/PIXEL_M))
    require(0 <= u < 32 and 0 <= v < 32, 'agent outside synthetic camera')
    image[v,u] = [255,255,0]
    for name, goal in GOALS.items():
        u, v = int(np.rint(goal[0]/PIXEL_M+16)), int(np.rint(16-goal[1]/PIXEL_M))
        image[v,u] = COLORS[name]
    return image


def target_from_context(image, language):
    require(image.shape == (32,32,3) and image.dtype == np.uint8, 'RGB (32,32,3) uint8')
    require(language in ('move to red','move to blue'), 'known literal toy instruction')
    name = language.split()[-1]
    pixels = np.argwhere(np.all(image == np.asarray(COLORS[name],dtype=np.uint8),axis=-1))
    require(pixels.shape == (1,2), 'exactly one selected target marker')
    v, u = pixels[0]
    return np.array([(u-16)*PIXEL_M,(16-v)*PIXEL_M],dtype=np.float32)


def plan(history, history_mask, image, language, horizon):
    """A handwritten planner, not a learned VLA or world model; no future source data."""
    require(type(horizon) is int and horizon > 0, 'positive horizon')
    require(history.ndim == 2 and history.shape[1] == 2 and history.dtype == np.float32, 'history (L,2) float32')
    require(history_mask.shape == (len(history),) and history_mask.dtype == np.bool_
            and history_mask[-1], 'current observation must be real')
    require(np.isfinite(history[history_mask]).all(), 'finite real state history')
    goal = target_from_context(image,language)
    simulated = history[-1].copy()
    actions = np.empty((horizon,2),dtype=np.float32)
    for k in range(horizon):
        actions[k] = np.clip(goal-simulated,-STEP_M,STEP_M)
        simulated += actions[k]
    return actions


def demonstration(group, start, length, name):
    """Measured XY and executed per-interval displacement; keep final image/state."""
    language = 'move to '+name
    positions = np.empty((length+1,2),dtype=np.float32); positions[0] = start
    actions = np.empty((length,2),dtype=np.float32)
    for t in range(length):
        actions[t] = np.clip(GOALS[name]-positions[t],-STEP_M,STEP_M)
        positions[t+1] = positions[t]+actions[t]
    arrays, meta = toy_episode(1)
    arrays.update(timestamp=np.arange(length+1,dtype=np.float64)*.1,proprioception=positions,action=actions,
                  image=np.stack([render(p) for p in positions]),terminated=np.zeros(length,dtype=bool),
                  truncated=np.array([False]*(length-1)+[True],dtype=bool))
    meta.update(episode_id=f'chunk_g{group}_{name}',task_name='colored_point_goal',
                embodiment='ideal_world_xy_point_rgb_v1',language_instruction=language,
                success=bool(np.max(np.abs(positions[-1]-GOALS[name])) <= 1e-6),
                success_definition='max final XY component error to selected static goal <= 1e-6 m',
                split='validation' if group == 2 else 'train')
    meta['provenance'].update(source='synthetic S0.7',group_id=f'independent_scene_reset_{group}',
                              controller='ideal per-interval world XY displacement',sampling='synthetic synchronous 10Hz')
    for key, value in arrays.items():
        if key == 'image':
            meta['modalities'][key]=dict(shape=list(value.shape),dtype='uint8',semantic='synthetic orthographic RGB markers',
                temporal='observation at t including final T; no sensor delay in this generator',
                frame='synthetic fixed world XY camera',color_space='RGB',source='S0.7 deterministic raster',
                calibration='u=16+x/0.02, v=16-y/0.02; nearest pixel; positions m; no real camera optics')
        else:
            meta['modalities'][key]['shape']=list(value.shape)
            meta['modalities'][key]['temporal']='observation at t including final T' if key in ('timestamp','proprioception') else 'transition t -> t+1'
    meta['modalities']['action']['semantic']='executed world XY displacement, component limit 0.06 m per interval'
    meta['modalities']['truncated']['semantic']='explicit demonstration command budget exhausted'
    validate(arrays,meta)
    return arrays,meta


def context_windows(episodes, history, horizon):
    windows = make_windows(episodes,history,horizon)
    source={m['episode_id']:a for a,m in episodes}
    # Only anchor RGB is policy input, not the future/terminal image array.
    windows['image'] = np.stack([source[e]['image'][t] for e,t in zip(windows['episode_id'],windows['anchor'])])
    windows['image_index'] = windows['anchor'].copy()
    audit_context(windows,episodes,history,horizon)
    return windows


def audit_context(windows, episodes, history, horizon):
    audit_windows(windows,episodes,history,horizon)
    source={m['episode_id']:(a,m) for a,m in episodes}
    camera=episodes[0][1]['modalities']['image']
    for _, meta in episodes:
        for key in ('frame','calibration','color_space','source'):
            require(meta['modalities']['image'][key] == camera[key], 'camera semantics mismatch')
    for i,(ep,t) in enumerate(zip(windows['episode_id'],windows['anchor'])):
        arrays,meta=source[ep]
        require(windows['image_index'][i] == t, 'future image index in context')
        require(np.array_equal(windows['image'][i],arrays['image'][t]), 'wrong anchor RGB')
        require(str(windows['language_instruction'][i]) == meta['language_instruction'], 'wrong episode language')


def predictions(windows, horizon):
    return np.stack([plan(h,m,image,str(language),horizon) for h,m,image,language in
                     zip(windows['observation'],windows['observation_mask'],windows['image'],windows['language_instruction'])])


def rollout(language, horizon, execute, kick_x, budget=4):
    require(type(execute) is int and 1 <= execute <= horizon, 'execute in [1,H]')
    position=np.array([0,-.12],dtype=np.float32)
    initial=position.copy(); trace=[position.copy()]; commands=[]; decisions=[]; latency=[]
    tick=0
    while tick < budget:
        image=render(position)
        start=perf_counter_ns()
        chunk=plan(position[None,:],np.array([True]),image,language,horizon)
        latency.append((perf_counter_ns()-start)/1e6)
        count=min(execute,horizon,budget-tick)
        decisions.append(dict(observation_tick=tick,executed_prefix=count,position=position.tolist(),
                              planned_chunk=chunk.tolist()))
        for k in range(count):
            commands.append(chunk[k].copy())
            position=position+chunk[k]; tick+=1
            if tick == 2:
                position=position+np.array([kick_x,0],dtype=np.float32)
            trace.append(position.copy())
    goal=target_from_context(render(initial),language)
    error=float(np.max(np.abs(position-goal))); success=error <= 1e-6
    return dict(language=language,horizon=horizon,execute=execute,budget_steps=budget,dt_s=.1,
        kick_after_transition=2,kick_x_m=kick_x,success=success,final_error_m=error,
        failure_stage=None if success else 'final goal error after command budget',policy_calls=len(decisions),
        rule_planner_latency_ms=dict(median=float(np.median(latency)),max=float(np.max(latency)),samples=len(latency),
            scope='single-run rule plan only; excludes rendering/controller; not neural inference or a deployment benchmark'),
        truncated=True,commands=np.asarray(commands).tolist(),trace=np.asarray(trace).tolist(),decisions=decisions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizon',type=int,default=4)
    parser.add_argument('--execute',type=int,default=1)
    parser.add_argument('--history',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path('tmp/s0_7'))
    args=parser.parse_args()
    require(args.horizon > 0 and args.history > 0, 'positive H/L')
    require(1 <= args.execute <= args.horizon, 'execute in [1,H]')
    require(not args.output.exists(), 'choose a new output directory')
    episodes=[demonstration(group,np.array(start,dtype=np.float32),length,name)
              for group,start,length in [(0,[0,-.12],2),(1,[0,.12],6),(2,[.06,-.12],4)] for name in ('red','blue')]
    args.output.mkdir(parents=True,exist_ok=False)
    episodes=[round_trip(args.output/'episodes'/m['episode_id'],a,m) for a,m in episodes]
    train=[ep for ep in episodes if ep[1]['split'] == 'train']; validation=[ep for ep in episodes if ep[1]['split'] == 'validation']
    require(not {m['provenance']['group_id'] for _,m in train} & {m['provenance']['group_id'] for _,m in validation}, 'split group overlap')
    batches={name:context_windows(eps,args.history,args.horizon) for name,eps in [('train',train),('validation',validation)]}
    metrics={}
    for name,windows in batches.items():
        predicted=predictions(windows,args.horizon)
        loss=masked_mse(predicted,windows['action'],windows['action_mask'])
        require(loss < 1e-12, 'handwritten rule/source label alignment')
        adversarial=predicted.copy(); adversarial[~windows['action_mask']]=np.nan
        require(masked_mse(adversarial,windows['action'],windows['action_mask']) == loss, 'padding affected masked loss')
        np.savez_compressed(args.output/(name+'_chunks.npz'),**windows,predicted=predicted)
        with np.load(args.output/(name+'_chunks.npz'),allow_pickle=False) as saved:
            require(all(np.array_equal(saved[k],x) for k,x in windows.items()) and np.array_equal(saved['predicted'],predicted), 'chunk round-trip')
        metrics[name]=dict(episodes=len(set(windows['episode_id'])),anchors=len(windows['anchor']),
            observation_shape=list(windows['observation'].shape),image_shape=list(windows['image'].shape),action_shape=list(predicted.shape),
            valid_action_tokens=int(windows['action_mask'].sum()),masked_mse_m2=loss,
            masked_loss_scalar_denominator=2*int(windows['action_mask'].sum()),
            last_action_index=windows['action_index'][-1].tolist(),last_mask=windows['action_mask'][-1].tolist())
    first_red=batches['train']['image'][0]; first_blue=batches['train']['image'][2]
    original_predictions=predictions(batches['train'],args.horizon)
    require(np.array_equal(first_red,first_blue), 'paired RGB differs')
    require(np.array_equal(batches['train']['observation'][0],batches['train']['observation'][2]), 'paired state differs')
    require(not np.array_equal(original_predictions[0],original_predictions[2]), 'language failed to select different target')
    # Future/result interventions must not change anchor-0 inputs or prediction.
    changed=copy.deepcopy(train)
    for arrays,meta in changed:
        arrays['proprioception'][1:]+=.01; arrays['image'][1:]=0; arrays['action'][:]*=-1; meta['success']=not meta['success']
    changed_windows=context_windows(changed,args.history,args.horizon)
    for i,t in enumerate(batches['train']['anchor']):
        if t == 0:
            for key in ('observation','observation_mask','image','language_instruction'):
                require(np.array_equal(changed_windows[key][i],batches['train'][key][i]), 'future/result leaked to anchor input')
            np.testing.assert_array_equal(plan(changed_windows['observation'][i],changed_windows['observation_mask'][i],changed_windows['image'][i],str(changed_windows['language_instruction'][i]),args.horizon),
                                          original_predictions[i])
    rejected=[]
    raw,meta=train[0]; image=raw['image'][0]; state=raw['proprioception'][0:1]; mask=np.array([True])
    for name,h,m,img,lang,H in [
        ('unknown_language',state,mask,image,'pick green',args.horizon),
        ('absent_target',state,mask,np.zeros_like(image),'move to red',args.horizon),
        ('invalid_rgb_dtype',state,mask,image.astype(np.float32),'move to red',args.horizon),
        ('missing_current_state',state,np.array([False]),image,'move to red',args.horizon),
        ('nonfinite_state',np.full_like(state,np.nan),mask,image,'move to red',args.horizon),
        ('zero_horizon',state,mask,image,'move to red',0)]:
        rejected.append(expect_rejection(name,lambda:plan(h,m,img,lang,H)))
    rejected.append(expect_rejection('execute_greater_than_horizon',lambda:rollout('move to red',args.horizon,args.horizon+1,0)))
    rejected.append(expect_rejection('empty_loss_mask',lambda:masked_mse(np.zeros_like(batches['train']['action']),batches['train']['action'],np.zeros_like(batches['train']['action_mask']))))
    bad=copy.deepcopy(train); bad[1][1]['modalities']['image']['calibration']='unknown'
    rejected.append(expect_rejection('camera_mismatch',lambda:context_windows(bad,args.history,args.horizon)))
    for name in ('future_image_index','corrupt_anchor_rgb','incorrect_tail_mask'):
        broken=copy.deepcopy(batches['train'])
        if name == 'future_image_index': broken['image_index'][0]+=1
        if name == 'corrupt_anchor_rgb': broken['image'][0]=0
        if name == 'incorrect_tail_mask': broken['action_mask'][-1,-1]=not broken['action_mask'][-1,-1]
        rejected.append(expect_rejection(name,lambda:audit_context(broken,train,args.history,args.horizon)))
    nominal=[rollout('move to '+name,args.horizon,args.execute,0) for name in ('red','blue')]
    disturbed=[rollout('move to '+name,args.horizon,args.execute,.12) for name in ('red','blue')]
    report=dict(schema_version='action_chunk_context_experiment/1.0',
        environment=dict(python=sys.version.split()[0],executable=sys.executable,numpy=np.__version__,numpy_path=np.__file__,
                         pillow=PIL.__version__,pillow_path=PIL.__file__),
        history=args.history,horizon=args.horizon,execute=args.execute,
        contract=dict(observation='causal measured world XY history (L,2) float32 m',image='anchor RGB (32,32,3) uint8; u=16+x/.02,v=16-y/.02',
            language='literal constant string move to red/blue',action='world XY measured-anchor displacement (H,2) float32 m per .1s',
            mask='offline target validity only; never given to planner',frame='fixed world XY',rule='read selected color marker; clip each XY error to +-0.06m, unroll ideal point',
            normalized=False,learned=False,limits='Only this explicit synthetic camera/controller; no learned encoder/fusion/policy'),
        split={name:[m['episode_id'] for _,m in eps] for name,eps in [('train',train),('validation',validation)]},
        batches=metrics,raw_valid_zero_action_rows=sum(int(np.all(a['action']==0,axis=1).sum()) for a,_ in episodes),
        raw_episodes=[dict(episode_id=m['episode_id'],actions=len(a['action']),observations=len(a['proprioception']),
                          success=m['success'],truncated=bool(a['truncated'][-1])) for a,m in episodes],
        paired_language_same_rgb_state=True,future_and_result_intervention_passed=True,rejected_cases=rejected,
        rollout=dict(nominal=nominal,disturbed=disturbed,nominal_successes=sum(r['success'] for r in nominal),disturbed_successes=sum(r['success'] for r in disturbed),denominator_per_condition=2),
        limitations='Handwritten deterministic planner and point dynamics; offline label equality is not a trained policy; no real robot success, pretrained vision/language model or world model.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    preview=Image.new('RGB',(512,3*288),'white'); draw=ImageDraw.Draw(preview)
    for i,(arrays,meta) in enumerate(episodes):
        x,y=(i%2)*256,(i//2)*288
        preview.paste(Image.fromarray(arrays['image'][0]).resize((256,256),Image.Resampling.NEAREST),(x,y+32))
        draw.text((x+8,y+8),f"{meta['episode_id']} | {meta['language_instruction']}",fill='black')
    preview.save(args.output/'context_preview.png')
    print(json.dumps({k:report[k] for k in ('history','horizon','execute','batches','raw_valid_zero_action_rows','rejected_cases')},indent=2))
    print(f"Toy nominal={report['rollout']['nominal_successes']}/2; disturbed={report['rollout']['disturbed_successes']}/2; calls={[r['policy_calls'] for r in nominal]}")
    print(f'PASS raw/chunk round-trip / source indices / RGB-language pairing / causal input / masked loss / prefix execution; output={args.output}')


if __name__ == '__main__':
    main()
