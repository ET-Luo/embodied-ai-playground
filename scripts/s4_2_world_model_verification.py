"""Predict frozen S4.1 chunks with a matched affine dynamics model; replay independently."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require
from s1_7_closed_loop_evaluation import sha256
from s4_1_vla_proposal import CONTRACT, validate_proposals

MODEL_CONTRACT = dict(schema_version='ideal_xy_affine_dynamics/1.0',
    embodiment=CONTRACT['embodiment'], representation=CONTRACT['representation'],
    frame='world_xy', unit='m', control_dt_s=.1, component_limit_m=.06,
    nominal_plant='next_state=state+command; no walls, contact, delay or external gain',
    model='delta = [normalized current XY, normalized command XY, 1] @ coefficients',
    normalization='input mean/std fit only train transitions; target remains raw displacement[m]')
SOURCE_FILES = ('contract.json', 'proposals.npz', 'report.json')


def load_proposals(directory):
    require(all((directory/name).is_file() for name in SOURCE_FILES), 'need existing S4.1 artifacts; no implicit generation')
    contract = json.loads((directory/'contract.json').read_text())
    require(contract == CONTRACT, 'S4.1 action semantics/embodiment mismatch')
    report = json.loads((directory/'report.json').read_text())
    require(report.get('schema_version') == 'vla_proposal_report/1.0', 'proposal report version')
    with np.load(directory/'proposals.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'action', 'image', 'state', 'text', 'observation_tick'}, 'proposal payload keys')
        data = {key: saved[key] for key in saved.files}
    actions, state = data['action'], data['state']
    require(actions.ndim == 4 and len(actions) == 16, '16 supported layout/language cases expected')
    for bank in actions: validate_proposals(bank, contract)
    require(state.shape == (16, 2) and state.dtype == np.float32 and np.isfinite(state).all(), 'current physical state')
    require(data['image'].shape == (16, 32, 32, 3) and data['image'].dtype == np.uint8, 'current RGB payload')
    require(data['text'].shape == (16,) and data['text'].dtype.kind == 'U', 'instruction strings')
    require(data['observation_tick'].shape == (16,) and data['observation_tick'].dtype == np.int64
            and np.all(data['observation_tick'] == 0), 'this task supports tick0 proposal anchors')
    require(report['accepted'] == 16 and report['proposal_attempts'] == 18 and len(report['rejected']) == 2
            and all(row['failure_stage'] == 'policy_input_rejection' for row in report['rejected']), 'preserve proposal denominators')
    require(report['candidates'] == actions.shape[1], 'reported K mismatch')
    for value in data.values(): value.flags.writeable = False
    return data, report


def plant_rollout(initial, actions, gain):
    """Independent evaluation plant. Each candidate resets to its own initial state."""
    require(np.isfinite(gain) and 0 < gain <= 1, 'plant gain in (0,1]')
    position = np.array(initial, dtype=np.float32, copy=True)
    trace = [position.copy()]
    for h in range(actions.shape[-2]):
        position = (position + np.float32(gain)*actions[..., h, :]).astype(np.float32)
        trace.append(position.copy())
    return np.stack(trace, axis=-2)


def training_episodes():
    rng = np.random.default_rng(421)
    initial = rng.uniform(-.4, .4, (40, 2)).astype(np.float32)
    actions = rng.uniform(-.06, .06, (40, 20, 2)).astype(np.float32)
    states = plant_rollout(initial, actions, 1.)
    ids = np.random.default_rng(422).permutation(40)
    return states, actions, ids[:30], ids[30:]


def fit_model(states, actions, train_ids):
    current = states[train_ids, :-1].reshape(-1, 2).astype(np.float64)
    commands = actions[train_ids].reshape(-1, 2).astype(np.float64)
    inputs = np.concatenate([current, commands], axis=-1)
    target = states[train_ids, 1:].reshape(-1, 2).astype(np.float64) - current
    mean, std = inputs.mean(0), inputs.std(0)
    design = np.column_stack([(inputs-mean)/std, np.ones(len(inputs))])
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    require(rank == 5, 'full rank train design')
    return dict(coefficients=coefficients, input_mean=mean, input_std=std)


def validate_model(model, contract):
    require(contract == MODEL_CONTRACT, 'world-model semantics/embodiment mismatch')
    require(set(model) == {'coefficients', 'input_mean', 'input_std'}, 'model keys')
    for key, shape in [('coefficients', (5, 2)), ('input_mean', (4,)), ('input_std', (4,))]:
        value = model[key]
        require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(), 'model finite float64 '+key)
    require(np.all(model['input_std'] > 0), 'positive train-only scales')


def predict_next(current, action, model):
    # No gain, true next state, RGB, language, target or success accepted here.
    inputs = np.concatenate([current, action], axis=-1).astype(np.float64)
    normalized = (inputs-model['input_mean'])/model['input_std']
    design = np.concatenate([normalized, np.ones((*normalized.shape[:-1], 1))], axis=-1)
    return np.asarray(current, np.float64) + design @ model['coefficients']


def imagine(initial, actions, model):
    position = np.array(initial, dtype=np.float64, copy=True)
    trace = [position.copy()]
    for h in range(actions.shape[-2]):
        position = predict_next(position, actions[..., h, :], model)
        trace.append(position.copy())
    return np.stack(trace, axis=-2)


def errors(prediction, truth):
    # Exclude the shared initial observation from every metric.
    delta = np.asarray(prediction, np.float64) - np.asarray(truth, np.float64)
    return dict(scalar_denominator=int(delta.size), rmse_m=float(np.sqrt(np.mean(delta**2))),
        per_horizon_rmse_m=np.sqrt(np.mean(delta**2, axis=tuple(range(delta.ndim-2)) + (delta.ndim-1,))).tolist(),
        final_l2_p95_m=float(np.percentile(np.linalg.norm(delta[..., -1, :], axis=-1), 95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals', type=Path, default=Path('tmp/s4_1_verified'))
    parser.add_argument('--plant-gain', type=float, default=.65)
    parser.add_argument('--output', type=Path, default=Path('tmp/s4_2'))
    args = parser.parse_args()
    require(not args.output.exists(), 'fresh output directory required')
    require(np.isfinite(args.plant_gain) and 0 < args.plant_gain < 1, 'shifted gain must be in (0,1)')
    source_hashes = {name: sha256(args.proposals/name) for name in SOURCE_FILES} if args.proposals.is_dir() else {}
    data, source_report = load_proposals(args.proposals)
    states, train_actions, train_ids, val_ids = training_episodes()
    model = fit_model(states, train_actions, train_ids)
    validate_model(model, MODEL_CONTRACT)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'model.npz', **model)
    with np.load(args.output/'model.npz', allow_pickle=False) as saved:
        loaded = {key: saved[key] for key in saved.files}
    for key in model: np.testing.assert_array_equal(model[key], loaded[key])
    model = loaded
    for value in model.values(): value.flags.writeable = False
    model_hash = sha256(args.output/'model.npz')
    np.savez_compressed(args.output/'training_episodes.npz', state_m=states, action_m=train_actions,
        train_episode_ids=train_ids, validation_episode_ids=val_ids, timestamp_s=np.arange(21, dtype=np.float64)*.1)
    action = data['action']
    initial = np.broadcast_to(data['state'][:, None], (*action.shape[:2], 2))
    # AR prediction is complete BEFORE any candidate execution truth is generated.
    for _ in range(5): imagine(initial, action, model)
    times = []; prediction = None
    for _ in range(30):
        start = perf_counter_ns()
        prediction = imagine(initial, action, model)
        times.append((perf_counter_ns()-start)/1e6)
    zero_prediction = imagine(initial, np.zeros_like(action), model)
    nominal = plant_rollout(initial, action, 1.)
    shifted = plant_rollout(initial, action, args.plant_gain)
    result = {}
    output = dict(action_m=action, initial_state_m=data['state'], predicted_ar_m=prediction,
        predicted_zero_action_m=zero_prediction, truth_nominal_m=nominal, truth_shifted_m=shifted,
        timestamp_s=np.arange(4, dtype=np.float64)*.1)
    for name, truth in [('nominal', nominal), ('gain_shift', shifted)]:
        # TF intentionally reads actual pre-action states; used ONLY as a diagnostic.
        tf = predict_next(truth[..., :-1, :], action, model)
        output['teacher_forced_'+name+'_m'] = tf
        result[name] = dict(autoregressive=errors(prediction[..., 1:, :], truth[..., 1:, :]),
            teacher_forced=errors(tf, truth[..., 1:, :]),
            zero_action_ablation=errors(zero_prediction[..., 1:, :], truth[..., 1:, :]))
    val_pred = predict_next(states[val_ids, :-1], train_actions[val_ids], model)
    validation = errors(val_pred, states[val_ids, 1:])
    train_pred = predict_next(states[train_ids, :-1], train_actions[train_ids], model)
    train_rmse = float(np.sqrt(np.mean((train_pred-states[train_ids, 1:])**2)))
    require(model_hash == sha256(args.output/'model.npz'), 'frozen model mutated')
    require(source_hashes == {name: sha256(args.proposals/name) for name in SOURCE_FILES}, 'proposal source mutated')
    np.savez_compressed(args.output/'verification.npz', **output)
    with np.load(args.output/'verification.npz', allow_pickle=False) as saved:
        for key, value in output.items(): np.testing.assert_array_equal(saved[key], value)
    contract = dict(schema_version='candidate_future_verification/1.0', proposal_contract=CONTRACT,
        world_model_contract=MODEL_CONTRACT, states='world XY[m]; raw truth float32, model prediction float64',
        action_shape=list(action.shape), trajectory_shape=list(prediction.shape),
        time='4 states including initial, 3 commands at dt=.1s; each candidate independently resets',
        conditions=dict(nominal_gain=1., shifted_gain=args.plant_gain,
            shift='evaluation-only plant mechanism change; frozen nominal model not informed of gain'),
        data='40 synthetic independent reset episodes, seeds421/422, 30train/10validation, 20transitions each',
        scope='counterfactual open-loop candidate replay, no selection, risk threshold, acceptance or feedback replan')
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    report = dict(schema_version='candidate_future_verification_report/1.0',
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, numpy_path=np.__file__),
        source=dict(directory=str(args.proposals.resolve()), sha256=source_hashes,
            proposal_attempts=18, supported=16, rejected=source_report['rejected']),
        model=dict(parameters=10, fit='least squares, train only; no policy data used for fitting',
            train_rmse_m=train_rmse, validation=validation, sha256=model_hash, frozen_during_verification=True),
        candidate_rollouts_per_condition=int(np.prod(action.shape[:2])), conditions=result,
        prediction_latency_ms=dict(median=float(np.median(times)), p95=float(np.percentile(times, 95)),
            samples=30, warmup=5, batch_candidates=int(np.prod(action.shape[:2])),
            scope='batched AR3 normalize/matmul/state update; excludes proposal, plant, metrics, IO'),
        closed_loop_success=None, failure_stage='gain_shift_prediction_mismatch is a diagnostic, not task failure attribution',
        limitations='Exact linear noiseless ideal plant, train model class contains truth. Old eight VLA validation layouts, paired biased candidates not independent trials. No task scoring, selection, closed-loop controller, calibrated risk or real robot claim.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['condition,horizon,ar_rmse_m,tf_rmse_m']
    for name, group in result.items():
        for h, (ar, tf) in enumerate(zip(group['autoregressive']['per_horizon_rmse_m'], group['teacher_forced']['per_horizon_rmse_m']), 1):
            lines.append(f'{name},{h},{ar},{tf}')
    (args.output/'curves.csv').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(validation_rmse_m=validation['rmse_m'], conditions=result,
        prediction_latency_ms=report['prediction_latency_ms']), indent=2))


if __name__ == '__main__':
    main()
