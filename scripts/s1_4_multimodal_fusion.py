"""CPU concatenate vs one-block Transformer fusion with frozen visual localization."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require
from s0_5_action_representation import expect_rejection
import s1_3_visual_encoder as vision

SCALE_M = np.float32(.3)
LANGUAGES = ('move to red', 'move to blue')
CONDITIONS = ('full', 'no_vision', 'no_language', 'no_state')


def visual_positions(rgb, weights):
    # Fixed R/B symmetry prior: apply the same learned red detector to channel-swapped RGB for blue.
    red = vision.predict(rgb, weights)[0]
    blue = vision.predict(rgb[..., ::-1].copy(), weights)[0]
    return np.stack((red, blue), axis=1)


def tokens(positions, state, languages, condition='full'):
    require(condition in CONDITIONS, 'known modality intervention')
    require(positions.shape == (len(state), 2, 2) and positions.dtype == np.float32 and np.isfinite(positions).all(),
            'visual position features (N,2,2) float32 m')
    require(state.shape == (len(languages), 2) and state.dtype == np.float32 and np.isfinite(state).all(),
            'current world XY (N,2) float32 m')
    require(all(text in LANGUAGES for text in languages), 'known literal color instruction')
    result = np.zeros((len(state), 4, 4), dtype=np.float32)
    result[:, :2, :2] = positions/SCALE_M
    result[:, 0, 2] = 1; result[:, 1, 3] = 1
    result[:, 2, 2:] = np.eye(2, dtype=np.float32)[[LANGUAGES.index(text) for text in languages]]
    result[:, 3, :2] = state/SCALE_M
    # Intervention zeros content after encoding; slots/type markers remain, no retraining.
    if condition == 'no_vision': result[:, :2, :2] = 0
    if condition == 'no_language': result[:, 2, 2:] = 0
    if condition == 'no_state': result[:, 3, :2] = 0
    return result


def layer_norm(value):
    centered = value-value.mean(-1, keepdims=True)
    inverse = 1/np.sqrt(np.mean(centered**2, axis=-1, keepdims=True)+1e-5)
    return centered*inverse, inverse


def norm_backward(gradient, normalized, inverse):
    return inverse*(gradient-gradient.mean(-1, keepdims=True)-normalized*(gradient*normalized).mean(-1, keepdims=True))


def initialize(kind, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    def matrix(a, b): return rng.normal(0, .2, (a, b)).astype(dtype)
    if kind == 'concat':
        return dict(w1=matrix(16, 32), b1=np.zeros(32, dtype), out=matrix(32, 2), bout=np.zeros(2, dtype))
    require(kind == 'transformer', 'known fusion model')
    return dict(proj=matrix(4, 8), types=rng.normal(0, .1, (4, 8)).astype(dtype),
                q=matrix(8, 8), k=matrix(8, 8), v=matrix(8, 8), wo=matrix(8, 8),
                f1=matrix(8, 16), b1=np.zeros(16, dtype), f2=matrix(16, 8), b2=np.zeros(8, dtype),
                out=matrix(8, 2), bout=np.zeros(2, dtype))


def forward(x, weights, kind):
    require(x.ndim == 3 and x.shape[1:] == (4, 4) and np.isfinite(x).all(), 'finite (N,4,4) fusion tokens')
    if kind == 'concat':
        flat = x.reshape(len(x), 16)
        hidden = np.tanh(flat @ weights['w1']+weights['b1'])
        return hidden @ weights['out']+weights['bout'], (flat, hidden)
    projected = x @ weights['proj']+weights['types']
    normalized, inv1 = layer_norm(projected)
    query = normalized @ weights['q']; key = normalized @ weights['k']; value = normalized @ weights['v']
    score = (query @ key.transpose(0, 2, 1))/(8**.5)
    shifted = score-score.max(-1, keepdims=True)
    attention = np.exp(shifted); attention /= attention.sum(-1, keepdims=True)
    context = attention @ value
    residual = projected+context @ weights['wo']
    norm2, inv2 = layer_norm(residual)
    hidden = np.tanh(norm2 @ weights['f1']+weights['b1'])
    final = residual+hidden @ weights['f2']+weights['b2']
    output = final[:, 3] @ weights['out']+weights['bout']
    cache = (projected, normalized, inv1, query, key, value, attention, context, residual, norm2, inv2, hidden, final)
    return output, cache


def gradients(x, target, weights, kind):
    predicted, cache = forward(x, weights, kind)
    loss = float(np.mean((predicted-target)**2))
    delta = 2*(predicted-target)/target.size
    if kind == 'concat':
        flat, hidden = cache
        back = (delta @ weights['out'].T)*(1-hidden**2)
        return loss, dict(w1=flat.T @ back, b1=back.sum(0), out=hidden.T @ delta, bout=delta.sum(0))
    projected, norm1, inv1, query, key, value, attention, context, residual, norm2, inv2, hidden, final = cache
    grad = dict(out=final[:, 3].T @ delta, bout=delta.sum(0))
    dfinal = np.zeros_like(final); dfinal[:, 3] = delta @ weights['out'].T
    grad['f2'] = hidden.reshape(-1, 16).T @ dfinal.reshape(-1, 8)
    grad['b2'] = dfinal.sum((0, 1))
    dhidden = (dfinal @ weights['f2'].T)*(1-hidden**2)
    grad['f1'] = norm2.reshape(-1, 8).T @ dhidden.reshape(-1, 16)
    grad['b1'] = dhidden.sum((0, 1))
    dresidual = dfinal+norm_backward(dhidden @ weights['f1'].T, norm2, inv2)
    grad['wo'] = context.reshape(-1, 8).T @ dresidual.reshape(-1, 8)
    dcontext = dresidual @ weights['wo'].T
    dattention = dcontext @ value.transpose(0, 2, 1)
    dvalue = attention.transpose(0, 2, 1) @ dcontext
    dscore = attention*(dattention-(dattention*attention).sum(-1, keepdims=True))
    dquery = (dscore @ key)/(8**.5)
    dkey = (dscore.transpose(0, 2, 1) @ query)/(8**.5)
    dnorm1 = np.zeros_like(norm1)
    for name, intermediate in [('q', dquery), ('k', dkey), ('v', dvalue)]:
        grad[name] = norm1.reshape(-1, 8).T @ intermediate.reshape(-1, 8)
        dnorm1 += intermediate @ weights[name].T
    dprojected = dresidual+norm_backward(dnorm1, norm1, inv1)
    grad['proj'] = x.reshape(-1, 4).T @ dprojected.reshape(-1, 8)
    grad['types'] = dprojected.sum(0)
    return loss, grad


def fit(x, target_m, kind, epochs, seed):
    weights = initialize(kind, seed)
    first = {key:np.zeros_like(value) for key,value in weights.items()}
    second = {key:value.copy() for key,value in first.items()}
    history = []
    for step in range(1, epochs+1):
        _, grad = gradients(x, target_m/SCALE_M, weights, kind)
        for key in weights:
            first[key] = .9*first[key]+.1*grad[key]
            second[key] = .999*second[key]+.001*grad[key]**2
            weights[key] -= .003*(first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+1e-8)
        if step == 1 or step % 200 == 0 or step == epochs:
            loss = np.mean((forward(x, weights, kind)[0]*SCALE_M-target_m)**2)
            history.append(dict(epoch=step, train_proposal_mse_m2=float(loss)))
    return weights, history


def make_batches(data, visual, output):
    rng = np.random.default_rng(23)
    batches = {}
    for split, batch in data.items():
        rows = []
        for index in range(len(batch['rgb'])):
            states = rng.uniform(-.26, .26, (4, 2)).astype(np.float32)
            for state in states:
                for color, language in enumerate(LANGUAGES):
                    goal = vision.pixel_to_world(batch['target_uv'][index] if color == 0 else batch['distractor_uv'][index])
                    rows.append((visual[split][index], state, language, goal-state, index))
        positions = np.stack([row[0] for row in rows]); state = np.stack([row[1] for row in rows])
        texts = np.array([row[2] for row in rows]); proposal = np.stack([row[3] for row in rows])
        scene_index = np.array([row[4] for row in rows])
        x = tokens(positions, state, texts)
        arrays = dict(positions_m=positions, state_m=state, language=texts, proposal_label_m=proposal,
                      executed_label_m=np.clip(proposal, -vision.STEP_M, vision.STEP_M),
                      scene_index=scene_index, scene_id=batch['scene_id'][scene_index], tokens=x)
        np.savez_compressed(output/(split+'_fusion.npz'), **arrays)
        with np.load(output/(split+'_fusion.npz'), allow_pickle=False) as saved:
            for key, value in arrays.items(): np.testing.assert_array_equal(saved[key], value)
        batches[split] = arrays
    require(not set(batches['train']['scene_id']) & set(batches['validation']['scene_id']), 'fusion scene overlap')
    return batches


def metrics(batch, weights, kind, condition):
    x = tokens(batch['positions_m'], batch['state_m'], batch['language'], condition)
    predicted = forward(x, weights, kind)[0]*SCALE_M
    return dict(samples=len(x), proposal_mse_m2=float(np.mean((predicted-batch['proposal_label_m'])**2)),
                executed_action_mse_m2=float(np.mean((np.clip(predicted, -vision.STEP_M, vision.STEP_M)-batch['executed_label_m'])**2)))


def rollout(batch, weights, cnn, kind, condition, kick_x):
    records = []; latencies = []
    for i, scene in enumerate(batch['scene_id']):
        rgb = batch['rgb'][i:i+1]
        for color, language in enumerate(LANGUAGES):
            state = np.array([0, -.12], np.float32)
            trace = [state.tolist()]; proposals = []; commands = []
            for tick in range(10):
                before = perf_counter_ns()
                positions = visual_positions(rgb, cnn)
                x = tokens(positions, state[None], [language], condition)
                proposal = forward(x, weights, kind)[0][0]*SCALE_M
                action = np.clip(proposal, -vision.STEP_M, vision.STEP_M)
                latencies.append((perf_counter_ns()-before)/1e6)
                state += action
                if tick == 3: state[0] += kick_x
                proposals.append(proposal.tolist()); commands.append(action.tolist()); trace.append(state.tolist())
            goal = vision.pixel_to_world(batch['target_uv'][i] if color == 0 else batch['distractor_uv'][i])
            error = float(np.max(np.abs(state-goal))); success = error <= .01
            records.append(dict(scene_id=str(scene), language=language, success=success, final_error_m=error,
                failure_stage=None if success else 'final goal error after 10-step budget', truncated=True,
                goal_world_m=goal.tolist(), trace=trace, proposals_m=proposals, commands=commands))
    return dict(successes=sum(row['success'] for row in records), denominator=len(records), episodes=records,
                kick_x_m=kick_x, kick_after_transition=4, latency_ms=dict(median=float(np.median(latencies)),
                p95=float(np.percentile(latencies, 95)), samples=len(latencies),
                scope='two CNN passes + encoding + fusion + clipping; excludes rendering/environment; frozen CNN still run in ablations'))


def reference_rollouts(batch, visual, kick_x, use_true_goal):
    records = []
    for i, scene in enumerate(batch['scene_id']):
        for color, language in enumerate(LANGUAGES):
            goal = vision.pixel_to_world(batch['target_uv'][i] if color == 0 else batch['distractor_uv'][i])
            estimate = goal if use_true_goal else visual[i, color]
            state = np.array([0, -.12], np.float32)
            trace = [state.tolist()]; commands = []
            for tick in range(10):
                action = np.clip(estimate-state, -vision.STEP_M, vision.STEP_M)
                state += action
                if tick == 3: state[0] += kick_x
                commands.append(action.tolist()); trace.append(state.tolist())
            error = float(np.max(np.abs(state-goal))); success = error <= .01
            records.append(dict(scene_id=str(scene), language=language, goal_world_m=goal.tolist(),
                success=success, final_error_m=error, trace=trace, commands=commands, truncated=True,
                failure_stage=None if success else 'final goal error after 10-step budget'))
    return dict(successes=sum(row['success'] for row in records), denominator=len(records), episodes=records,
                scope='fixed goal-error controller; oracle uses true goal, visual reference routes cached CNN positions by literal color; no latency comparison')


def checks():
    rng = np.random.default_rng(31)
    x = rng.normal(0, .2, (2, 4, 4))
    target = rng.normal(0, .2, (2, 2))
    errors = {}
    for kind in ('concat', 'transformer'):
        weights = initialize(kind, 3, np.float64)
        _, analytic = gradients(x, target, weights, kind)
        worst = 0.
        for name, parameter in weights.items():
            for index in np.ndindex(parameter.shape):
                old = parameter[index]; eps = 1e-5
                parameter[index] = old+eps; plus = gradients(x, target, weights, kind)[0]
                parameter[index] = old-eps; minus = gradients(x, target, weights, kind)[0]
                parameter[index] = old
                worst = max(worst, abs((plus-minus)/(2*eps)-analytic[name][index]))
        require(worst < 1e-7, f'{kind} gradient check')
        errors[kind] = worst
    pos = np.ones((1, 2, 2), np.float32); state = np.zeros((1, 2), np.float32)
    rejected = [expect_rejection('unknown_language', lambda: tokens(pos, state, ['move green'])),
                expect_rejection('unknown_intervention', lambda: tokens(pos, state, [LANGUAGES[0]], 'all')),
                expect_rejection('wrong_state_dtype', lambda: tokens(pos, state.astype(np.float64), [LANGUAGES[0]])),
                expect_rejection('wrong_visual_shape', lambda: tokens(pos[:, :1], state, [LANGUAGES[0]])),
                expect_rejection('nonfinite_token', lambda: forward(np.full((1, 4, 4), np.nan), initialize('concat', 7), 'concat'))]
    return dict(gradient_max_absolute_errors=errors, rejected_cases=rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1600)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--kick-x', type=float, default=0.)
    parser.add_argument('--output', type=Path, default=Path('tmp/s1_4'))
    args = parser.parse_args()
    require(args.epochs > 0 and np.isfinite(args.kick_x) and abs(args.kick_x) <= .12 and not args.output.exists(),
            'positive epochs, finite kick in [-.12,.12] m and fresh output')
    verification = checks()
    args.output.mkdir(parents=True)
    data = vision.make_data(args.output)
    cnn, _, cnn_history = vision.train(data['train'], 400, args.seed)
    frozen = [w.copy() for w in cnn]
    np.savez_compressed(args.output/'frozen_cnn.npz', **dict(zip(vision.KEYS, cnn)))
    visual = {split:visual_positions(batch['rgb'], cnn) for split,batch in data.items()}
    batches = make_batches(data, visual, args.output)
    models = {}
    references = dict(oracle=reference_rollouts(data['validation'], visual['validation'], args.kick_x, True),
                      frozen_visual=reference_rollouts(data['validation'], visual['validation'], args.kick_x, False))
    for kind in ('concat', 'transformer'):
        weights, history = fit(batches['train']['tokens'], batches['train']['proposal_label_m'], kind, args.epochs, args.seed)
        np.savez_compressed(args.output/(kind+'_weights.npz'), **weights)
        with np.load(args.output/(kind+'_weights.npz'), allow_pickle=False) as saved:
            loaded = {key:saved[key] for key in saved.files}
        np.testing.assert_array_equal(forward(batches['validation']['tokens'], weights, kind)[0],
                                      forward(batches['validation']['tokens'], loaded, kind)[0])
        conditions = {}
        for condition in CONDITIONS:
            conditions[condition] = dict(offline=metrics(batches['validation'], weights, kind, condition),
                rollout=rollout(data['validation'], weights, cnn, kind, condition, args.kick_x))
        pair_state = np.array([[0, -.12], [0, -.12]], np.float32)
        pair = tokens(np.repeat(visual['validation'][:1], 2, axis=0), pair_state, LANGUAGES)
        paired = forward(pair, weights, kind)[0]*SCALE_M
        trace = dict(input_tokens=list(pair.shape), output=list(paired.shape))
        cache = forward(pair, weights, kind)[1]
        if kind == 'transformer':
            trace.update(projected=list(cache[0].shape), attention=list(cache[6].shape), ffn_hidden=list(cache[11].shape))
            np.savez_compressed(args.output/'attention_probe.npz', tokens=pair, attention=cache[6], proposal_m=paired)
        else:
            trace.update(flatten=list(cache[0].shape), hidden=list(cache[1].shape))
        models[kind] = dict(parameters=sum(w.size for w in weights.values()), history=history, shape_trace=trace,
                            train=metrics(batches['train'], weights, kind, 'full'), conditions=conditions,
                            same_rgb_state_different_language_proposals_m=paired.tolist())
        require(forward(pair, weights, kind)[0].dtype == np.float32, 'runtime fusion must remain float32')
        print(kind, json.dumps({name:f"{value['rollout']['successes']}/48" for name,value in conditions.items()}), flush=True)
    for before, after in zip(frozen, cnn): np.testing.assert_array_equal(before, after)
    manifest = dict(schema_version='multimodal_fusion_examples/1.0',
        source='S1.3 synthetic scenes; current state samples seed23, same reset pair stays in split',
        split={name:batch['scene_id'].tolist() for name,batch in data.items()},
        language=list(LANGUAGES), language_encoding='fixed literal one-hot; no pretrained language semantics',
        visual='frozen train-only red CNN; blue from same detector with explicit R/B symmetry prior',
        tokens='N,4,4 float32 dimensionless; red,blue,language,state; object/current state XY divided by .3m',
        label='proposal_label_m=selected true goal-current state; unbounded displacement proposal, not executed action',
        execution='clip proposal each XY component to +/-.06m once per .1s; H=E=1, ideal point',
        raw='train/validation.npz preserve source RGB/physical coordinates; fusion labels stored separately',
        intervention='post-encoding zero content, retains slot/type markers; inference only, no retraining',
        normalization='fixed /.3m; no fitted validation statistics', learned='CNN then frozen, independently trained fusion models')
    (args.output/'fusion_contract.json').write_text(json.dumps(manifest, indent=2)+'\n')
    report = dict(schema_version='multimodal_fusion_experiment/1.0', environment=dict(python=sys.version,
        executable=sys.executable, numpy=np.__version__, numpy_path=np.__file__), epochs=args.epochs, seed=args.seed,
        kick_x_m=args.kick_x, checks=verification, cnn_frozen=True, cnn_history=cnn_history, models=models, references=references,
        limitations='Synthetic two literal color tasks, frozen localization with fixed channel-swap prior, one seed; object summary tokens not full image patches. Different parameter counts; no Transformer superiority, attention explanation, end-to-end VLA or real robot claim.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    canvas = np.concatenate([np.repeat(np.repeat(image, 4, 0), 4, 1) for image in data['validation']['rgb'][:4]], axis=1)
    vision.png(args.output/'scenes.png', canvas)
    print(json.dumps(dict(checks=verification, parameters={name:m['parameters'] for name,m in models.items()},
                          validation={name:m['conditions']['full']['offline'] for name,m in models.items()}), indent=2))
    print(f'PASS gradients / frozen CNN / split / model and data round-trip; {args.output}')


if __name__ == '__main__':
    main()
