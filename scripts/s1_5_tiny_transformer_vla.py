"""Tiny CPU patch/word/proprio Transformer -> parallel actual displacement chunk."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require, toy_episode, validate
from s0_2_sequence_dataset import make_windows, audit_windows, masked_mse
from s0_5_action_representation import round_trip, expect_rejection
from s1_4_multimodal_fusion import layer_norm, norm_backward
from s1_3_visual_encoder import png

DIM = 16
STEP = np.float32(.06)
HORIZON = 3
LENGTH = 8
LANGUAGES = ('move to red', 'move to blue')


def render(centers):
    require(np.asarray(centers).shape == (2, 2), 'two object pixel centers')
    rgb = np.full((32, 32, 3), 8, np.uint8)
    for (u, v), color in zip(centers, ([240, 20, 20], [20, 20, 240])):
        require(u in (4, 12, 20, 28) and v in (4, 12, 20, 28), 'patch-center object positions')
        rgb[v-1:v+2, u-1:u+2] = color
    return rgb


def world(uv):
    uv = np.asarray(uv, dtype=np.float32)
    return np.stack(((uv[..., 0]-16)*.02, (16-uv[..., 1])*.02), axis=-1)


def image_tokens(rgb):
    require(rgb.ndim == 4 and rgb.shape[1:] == (32, 32, 3) and rgb.dtype == np.uint8 and len(rgb) > 0,
            'nonempty current RGB (N,32,32,3) uint8')
    cells = rgb.astype(np.float32).reshape(-1, 4, 8, 4, 8, 3).mean((2, 4))/np.float32(255)
    v, u = np.meshgrid(np.arange(4, 32, 8), np.arange(4, 32, 8), indexing='ij')
    xy = np.stack(((u-16)/16, (16-v)/16), axis=-1).astype(np.float32).reshape(16, 2)
    return np.concatenate((cells.reshape(-1, 16, 3), np.broadcast_to(xy, (len(rgb), 16, 2))), axis=-1)


def fit_vocabulary(texts):
    return {'<PAD>': 0, '<UNK>': 1, **{word:i+2 for i, word in enumerate(sorted({w for t in texts for w in t.lower().split()}))}}


def encode_text(texts, vocabulary):
    require(all(isinstance(t, str) and t.lower().split() in [s.split() for s in LANGUAGES] for t in texts),
            'only the two trained three-word instructions; unknown/reordered text rejected')
    ids = np.array([[vocabulary[w] for w in text.lower().split()] for text in texts], dtype=np.int64)
    require(ids.shape == (len(texts), 3), 'three word tokens')
    return ids


def initialize(vocabulary_size, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    def matrix(a, b): return rng.normal(0, .15, (a, b)).astype(dtype)
    embedding = matrix(vocabulary_size, DIM); embedding[:2] = 0
    return dict(image=matrix(5, DIM), word=embedding, state=matrix(2, DIM), types=matrix(3, DIM),
                q=matrix(DIM, DIM), k=matrix(DIM, DIM), v=matrix(DIM, DIM), wo=matrix(DIM, DIM),
                f1=matrix(DIM, 32), b1=np.zeros(32, dtype), f2=matrix(32, DIM), b2=np.zeros(DIM, dtype),
                out=matrix(DIM, HORIZON*2), bout=np.zeros(HORIZON*2, dtype))


def forward(patches, ids, state, weights):
    require(patches.shape == (len(state), 16, 5) and np.isfinite(patches).all(), 'finite 16 patch features')
    require(ids.shape == (len(state), 3) and ids.dtype == np.int64 and np.all((ids >= 2) & (ids < len(weights['word']))),
            'three known token IDs; no PAD/UNK in current context')
    require(state.shape == (len(ids), 2) and np.isfinite(state).all(), 'finite current XY')
    embedded = np.concatenate((patches @ weights['image']+weights['types'][0],
                              weights['word'][ids]+weights['types'][1],
                              (state @ weights['state']+weights['types'][2])[:, None]), axis=1)
    norm1, inv1 = layer_norm(embedded)
    query = norm1 @ weights['q']; key = norm1 @ weights['k']; value = norm1 @ weights['v']
    logits = (query @ key.transpose(0, 2, 1))/(DIM**.5)
    attention = np.exp(logits-logits.max(-1, keepdims=True)); attention /= attention.sum(-1, keepdims=True)
    context = attention @ value
    residual = embedded+context @ weights['wo']
    norm2, inv2 = layer_norm(residual)
    hidden = np.tanh(norm2 @ weights['f1']+weights['b1'])
    final = residual+hidden @ weights['f2']+weights['b2']
    output = (final[:, 18] @ weights['out']+weights['bout']).reshape(-1, HORIZON, 2)
    return output, (embedded, norm1, inv1, query, key, value, attention, context, norm2, inv2, hidden, final)


def gradients(patches, ids, state, target, mask, weights):
    predicted, cache = forward(patches, ids, state, weights)
    require(target.shape == predicted.shape and mask.shape == predicted.shape[:2] and mask.dtype == bool and mask.any(), 'valid chunk target/mask')
    difference = predicted[mask]-target[mask]
    require(np.isfinite(difference).all(), 'finite valid labels')
    loss = float(np.mean(difference**2))
    derivative = np.zeros_like(predicted); derivative[mask] = 2*difference/difference.size
    delta = derivative.reshape(len(ids), -1)
    embedded, norm1, inv1, query, key, value, attention, context, norm2, inv2, hidden, final = cache
    grad = dict(out=final[:, 18].T @ delta, bout=delta.sum(0))
    dfinal = np.zeros_like(final); dfinal[:, 18] = delta @ weights['out'].T
    grad['f2'] = hidden.reshape(-1, 32).T @ dfinal.reshape(-1, DIM); grad['b2'] = dfinal.sum((0, 1))
    dhidden = (dfinal @ weights['f2'].T)*(1-hidden**2)
    grad['f1'] = norm2.reshape(-1, DIM).T @ dhidden.reshape(-1, 32); grad['b1'] = dhidden.sum((0, 1))
    dresidual = dfinal+norm_backward(dhidden @ weights['f1'].T, norm2, inv2)
    grad['wo'] = context.reshape(-1, DIM).T @ dresidual.reshape(-1, DIM)
    dcontext = dresidual @ weights['wo'].T
    dattention = dcontext @ value.transpose(0, 2, 1)
    dvalue = attention.transpose(0, 2, 1) @ dcontext
    dlogits = attention*(dattention-(dattention*attention).sum(-1, keepdims=True))
    dquery = dlogits @ key/(DIM**.5); dkey = dlogits.transpose(0, 2, 1) @ query/(DIM**.5)
    dnorm1 = np.zeros_like(norm1)
    for name, intermediate in [('q', dquery), ('k', dkey), ('v', dvalue)]:
        grad[name] = norm1.reshape(-1, DIM).T @ intermediate.reshape(-1, DIM)
        dnorm1 += intermediate @ weights[name].T
    dembedded = dresidual+norm_backward(dnorm1, norm1, inv1)
    grad['image'] = patches.reshape(-1, 5).T @ dembedded[:, :16].reshape(-1, DIM)
    grad['word'] = np.zeros_like(weights['word'])
    np.add.at(grad['word'], ids.ravel(), dembedded[:, 16:19].reshape(-1, DIM))
    grad['state'] = state.T @ dembedded[:, -1]
    grad['types'] = np.stack((dembedded[:, :16].sum((0, 1)), dembedded[:, 16:19].sum((0, 1)), dembedded[:, -1].sum(0)))
    return loss, grad


def make_dataset(output):
    coordinates = np.array([(u, v) for v in (4, 12, 20, 28) for u in (4, 12, 20, 28)], dtype=np.int64)
    layouts = [(red, blue) for red in range(16) for blue in range(16) if red != blue]
    selected = np.random.default_rng(11).permutation(len(layouts))[:40]
    episodes = {'train': [], 'validation': []}
    scenes = {'train': [], 'validation': []}
    for group, layout in enumerate(selected):
        split = 'train' if group < 32 else 'validation'
        centers = coordinates[list(layouts[layout])]
        rgb = render(centers)
        scene = dict(scene_id=f'layout_{layout}', centers_uv=centers.tolist())
        scenes[split].append(scene)
        for color, language in enumerate(LANGUAGES):
            goal = world(centers[color]); states = [np.array([0, -.12], np.float32)]; actions = []
            for _ in range(LENGTH):
                action = np.clip(goal-states[-1], -STEP, STEP)
                actions.append(action); states.append(states[-1]+action)
            arrays, meta = toy_episode(1)
            arrays.update(proprioception=np.stack(states), action=np.stack(actions),
                          timestamp=np.arange(LENGTH+1, dtype=np.float64)*.1,
                          terminated=np.zeros(LENGTH, bool), truncated=np.array([False]*(LENGTH-1)+[True]),
                          image=np.repeat(rgb[None], LENGTH+1, axis=0))
            meta.update(episode_id=f'vla_{layout}_{color}', language_instruction=language, task_name='patch_grid_color_goal',
                        success=bool(np.max(np.abs(states[-1]-goal)) <= .01),
                        success_definition='max final world XY goal error <= .01 m', embodiment='ideal_world_xy_patch_grid_v1')
            meta['provenance'].update(source='S1.5 synthetic expert', group_id=scene['scene_id'], split=split)
            for key, value in arrays.items():
                if key == 'image':
                    meta['modalities'][key] = dict(shape=list(value.shape), dtype='uint8', color_space='RGB',
                        semantic='current static red/blue squares, no agent marker', temporal='observation at t including T',
                        frame='fixed world XY orthographic synthetic camera', calibration='u=16+x/.02;v=16-y/.02; centers4,12,20,28',
                        source='S1.5 deterministic raster')
                else:
                    meta['modalities'][key]['shape'] = list(value.shape)
            validate(arrays, meta)
            np.testing.assert_allclose(arrays['proprioception'][1:], arrays['proprioception'][:-1]+arrays['action'])
            episodes[split].append(round_trip(output/'episodes'/meta['episode_id'], arrays, meta))
    require(not {s['scene_id'] for s in scenes['train']} & {s['scene_id'] for s in scenes['validation']}, 'layout overlap')
    batches = {}
    for split, source in episodes.items():
        batch = make_windows(source, 1, HORIZON); audit_windows(batch, source, 1, HORIZON)
        raw = {m['episode_id']:a for a, m in source}
        batch['image'] = np.stack([raw[e]['image'][t] for e,t in zip(batch['episode_id'], batch['anchor'])])
        batch['patches'] = image_tokens(batch['image'])
        batches[split] = batch
    vocabulary = fit_vocabulary(batches['train']['language_instruction'])
    for split, batch in batches.items():
        batch['word_ids'] = encode_text(batch['language_instruction'], vocabulary)
        np.savez_compressed(output/(split+'_chunks.npz'), **batch)
        with np.load(output/(split+'_chunks.npz'), allow_pickle=False) as saved:
            for key, value in batch.items(): np.testing.assert_array_equal(saved[key], value)
    return episodes, scenes, batches, vocabulary


def train(batch, vocabulary, steps, seed):
    weights = initialize(len(vocabulary), seed)
    first = {key:np.zeros_like(value) for key,value in weights.items()}; second = {key:value.copy() for key,value in first.items()}
    rng = np.random.default_rng(seed+1); history = []
    for step in range(1, steps+1):
        indices = rng.choice(len(batch['anchor']), 32, replace=False)
        _, grad = gradients(batch['patches'][indices], batch['word_ids'][indices], batch['observation'][indices, -1]/np.float32(.3),
                            batch['action'][indices]/STEP, batch['action_mask'][indices], weights)
        for key in weights:
            first[key] = .9*first[key]+.1*grad[key]; second[key] = .999*second[key]+.001*grad[key]**2
            weights[key] -= .002*(first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+1e-8)
        if step == 1 or step % 400 == 0 or step == steps:
            predicted = forward(batch['patches'], batch['word_ids'], batch['observation'][:, -1]/np.float32(.3), weights)[0]*STEP
            history.append(dict(step=step, train_masked_mse_m2=masked_mse(predicted, batch['action'], batch['action_mask'])))
            print('train', history[-1], flush=True)
    np.testing.assert_array_equal(weights['word'][:2], 0)
    return weights, history


def predict(rgb, state, text, weights, vocabulary):
    state = np.asarray(state)
    require(state.shape == (2,) and state.dtype == np.float32 and np.isfinite(state).all(), 'finite current float32 XY')
    ids = encode_text([text], vocabulary)
    output, _ = forward(image_tokens(rgb[None]), ids, state[None]/np.float32(.3), weights)
    require(output.dtype == np.float32, 'float32 inference')
    return np.clip(output[0]*STEP, -STEP, STEP)


def rollout(scene, text, weights, vocabulary, execute, kick_x, oracle=False):
    centers = np.array(scene['centers_uv'], dtype=np.int64); rgb = render(centers)
    goal = world(centers[LANGUAGES.index(text)])
    state = np.array([0, -.12], np.float32); trace = [state.tolist()]; commands = []; decisions = []; timings = []
    tick = 0
    while tick < LENGTH:
        before = perf_counter_ns()
        if oracle:
            simulated = state.copy(); chunk = []
            for _ in range(HORIZON):
                action = np.clip(goal-simulated, -STEP, STEP); chunk.append(action); simulated += action
            chunk = np.stack(chunk)
        else:
            chunk = predict(rgb, state, text, weights, vocabulary)
        timings.append((perf_counter_ns()-before)/1e6)
        count = min(execute, LENGTH-tick)
        decisions.append(dict(observation_tick=tick, state_m=state.tolist(), predicted_chunk_m=chunk.tolist(), executed_prefix=count))
        for offset in range(count):
            action = chunk[offset]; state += action; tick += 1
            if tick == 2: state[0] += kick_x
            commands.append(action.tolist()); trace.append(state.tolist())
    error = float(np.max(np.abs(state-goal))); success = error <= .01
    return dict(scene_id=scene['scene_id'], language=text, execute=execute, success=success, final_error_m=error,
                failure_stage=None if success else 'final goal error after 8-step budget', truncated=True,
                kick_x_m=kick_x, kick_after_transition=2, goal_world_m=goal.tolist(), trace=trace, commands=commands,
                decisions=decisions, policy_calls=len(decisions), latency_ms=dict(median=float(np.median(timings)),
                p95=float(np.percentile(timings, 95)), samples=len(timings),
                scope='patch preprocessing + tokenizer + Transformer + clipping; excludes render/environment; oracle skips model'))


def checks(vocabulary):
    rng = np.random.default_rng(3)
    patches = rng.normal(0, .1, (2, 16, 5)); state = rng.normal(0, .1, (2, 2))
    ids = encode_text(LANGUAGES, vocabulary)
    target = rng.normal(0, .1, (2, HORIZON, 2)); mask = np.array([[True, True, True], [True, False, False]])
    weights = initialize(len(vocabulary), 3, np.float64)
    loss, analytic = gradients(patches, ids, state, target, mask, weights)
    errors = {}
    # Every parameter entry, including word embeddings and all horizon channels.
    for name, parameter in weights.items():
        worst = 0.
        for flat_index in range(parameter.size):
            index = np.unravel_index(flat_index, parameter.shape); old = parameter[index]; eps = 1e-5
            parameter[index] = old+eps; plus = gradients(patches, ids, state, target, mask, weights)[0]
            parameter[index] = old-eps; minus = gradients(patches, ids, state, target, mask, weights)[0]
            parameter[index] = old
            worst = max(worst, abs((plus-minus)/(2*eps)-analytic[name][index]))
        require(worst < 1e-7, f'{name} gradient probe')
        errors[name] = worst
    poisoned = target.copy(); poisoned[~mask] = np.nan
    changed_loss, changed_grad = gradients(patches, ids, state, poisoned, mask, weights)
    require(loss == changed_loss, 'padding NaN affected loss')
    for name in analytic: np.testing.assert_array_equal(analytic[name], changed_grad[name])
    rejected = [expect_rejection('unknown_text', lambda: encode_text(['move to green'], vocabulary)),
                expect_rejection('empty_text', lambda: encode_text([''], vocabulary)),
                expect_rejection('float_image', lambda: image_tokens(np.zeros((1, 32, 32, 3), np.float32))),
                expect_rejection('wrong_resolution', lambda: image_tokens(np.zeros((1, 16, 16, 3), np.uint8))),
                expect_rejection('empty_label_mask', lambda: gradients(patches, ids, state, target, np.zeros_like(mask), weights))]
    return dict(finite_difference_all_parameters_per_tensor=errors, padding_nan_loss_and_gradient_invariance=True, rejected_cases=rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--execute', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('tmp/s1_5'))
    args = parser.parse_args()
    require(args.steps > 0 and 1 <= args.execute <= HORIZON and not args.output.exists(), 'positive steps, 1<=E<=H3, fresh output')
    args.output.mkdir(parents=True)
    episodes, scenes, batches, vocabulary = make_dataset(args.output)
    verification = checks(vocabulary)
    # Current input must be isolated from future arrays/results, even before training.
    source = episodes['train'][0][0]
    source_copy = {key:value.copy() for key,value in source.items()}
    source_copy['proprioception'][1:] += .07; source_copy['image'][1:] = 0; source_copy['action'] *= -1
    np.testing.assert_array_equal(source_copy['proprioception'][0], source['proprioception'][0])
    np.testing.assert_array_equal(image_tokens(source_copy['image'][:1]), image_tokens(source['image'][:1]))
    verification['anchor0_future_array_input_isolation'] = True
    weights, history = train(batches['train'], vocabulary, args.steps, args.seed)
    np.savez_compressed(args.output/'weights.npz', **weights)
    (args.output/'tokenizer.json').write_text(json.dumps(dict(vocabulary=vocabulary, normalization='lower().split()', known_instructions=LANGUAGES,
        unknown_policy='reject', fit_split='train', context_words=3, pad_unk='reserved zero rows, never present in context'), indent=2)+'\n')
    with np.load(args.output/'weights.npz', allow_pickle=False) as saved: loaded = {key:saved[key] for key in saved.files}
    metrics = {}
    for split, batch in batches.items():
        prediction, _ = forward(batch['patches'], batch['word_ids'], batch['observation'][:, -1]/np.float32(.3), weights)
        np.testing.assert_array_equal(prediction, forward(batch['patches'], batch['word_ids'], batch['observation'][:, -1]/np.float32(.3), loaded)[0])
        metrics[split] = dict(episodes=len(episodes[split]), anchors=len(batch['anchor']), valid_action_tokens=int(batch['action_mask'].sum()),
            raw_masked_mse_m2=masked_mse(prediction*STEP, batch['action'], batch['action_mask']),
            clipped_masked_mse_m2=masked_mse(np.clip(prediction*STEP, -STEP, STEP), batch['action'], batch['action_mask']),
            zero_action_baseline_mse_m2=masked_mse(np.zeros_like(prediction), batch['action'], batch['action_mask']))
    results = {}
    for name, kick in [('nominal', 0.), ('disturbed', .12)]:
        for oracle in (False, True):
            rows = [rollout(scene, text, loaded, vocabulary, args.execute, kick, oracle) for scene in scenes['validation'] for text in LANGUAGES]
            results[name+('_oracle' if oracle else '')] = dict(successes=sum(row['success'] for row in rows), denominator=len(rows), episodes=rows)
    batch = batches['validation']; first = np.flatnonzero(batch['anchor'] == 0)[:2]
    pair_prediction, cache = forward(batch['patches'][first], batch['word_ids'][first], batch['observation'][first, -1]/np.float32(.3), weights)
    np.savez_compressed(args.output/'attention_probe.npz', attention=cache[6], word_ids=batch['word_ids'][first],
                        input_patches=batch['patches'][first], current_state_m=batch['observation'][first, -1])
    np.testing.assert_array_equal(batch['image'][first[0]], batch['image'][first[1]])
    np.testing.assert_array_equal(batch['observation'][first[0]], batch['observation'][first[1]])
    report = dict(paired_first_anchor_chunks_m=(pair_prediction*STEP).tolist(), schema_version='tiny_transformer_vla/1.0', environment=dict(python=sys.version, executable=sys.executable,
        numpy=np.__version__, numpy_path=np.__file__), steps=args.steps, seed=args.seed, horizon=HORIZON, execute=args.execute,
        parameters=sum(w.size for w in weights.values()), checks=verification, history=history, metrics=metrics, rollouts=results,
        shape_trace=dict(rgb=['N',32,32,3], patches=['N',16,5], word_ids=['N',3], context=['N',20,DIM],
                         attention=['N',20,20], ffn=['N',20,32], chunk=['N',HORIZON,2]),
        limitations='One seed, grid-centered colored squares, fixed average patch preprocessing and spatial coordinates; only two training instructions, static RGB. Direct parallel chunk regression, not autoregressive language/action generation or large foundation model. No pretrained encoder, oracle object positions, channel-swap detector or learned world model in policy.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    manifest = dict(schema_version='tiny_patch_vla_contract/1.0', split=scenes, raw_schema='embodied_episode/1.0',
        raw='T8 executed world delta[m], T+1 state/image/clock, dt=.1s, final budget truncated, success independent',
        input='only current anchor RGB/state and episode text; no future/source action mask/success/goal labels',
        patches='4x4 row-major 8x8 average RGB /255 plus fixed [world-aligned x,y] normalized patch centers; no within-patch detail',
        trained='image projection, word embedding, state projection, modality types, one pre-LN attention+FFN block; color-word token index18 readout; parallel H3 delta head',
        action='output x.06m, clip each component to +/-.06m; H3 output/E prefix/dt.1s separate',
        split_scope='paired layout groups disjoint; individual object centers seen in both splits; no OOD position claim')
    (args.output/'model_contract.json').write_text(json.dumps(manifest, indent=2)+'\n')
    previews = [np.repeat(np.repeat(render(np.array(scene['centers_uv'])), 4, 0), 4, 1) for scene in scenes['validation'][:4]]
    png(args.output/'scenes.png', np.concatenate(previews, axis=1))
    print(json.dumps(dict(metrics=metrics, parameters=report['parameters'], execute=args.execute,
        successes={name:f"{value['successes']}/{value['denominator']}" for name,value in results.items()}), indent=2))
    print(f'PASS raw/window audits / all-parameter gradient checks / mask invariance / model reload; {args.output}')


if __name__ == '__main__':
    main()
