"""CPU NumPy 3x3 CNN -> spatial features -> target heatmap -> world XY."""
import argparse
import json
from pathlib import Path
import struct
import sys
from time import perf_counter_ns
import zlib

import numpy as np

from s0_1_data_contract import require
from s0_5_action_representation import expect_rejection

PIXEL_M = .02
STEP_M = .06
FEATURES = 4
KEYS = ('kernel', 'bias', 'heatmap_weight')


def pixel_to_world(uv):
    uv = np.asarray(uv, dtype=np.float32)
    return np.stack(((uv[..., 0]-16)*PIXEL_M, (16-uv[..., 1])*PIXEL_M), axis=-1)


def render(target_uv, distractor_uv, observation='visible'):
    require(observation in ('visible', 'occluded', 'blank'), 'known observation condition')
    image = np.full((32, 32, 3), 8, dtype=np.uint8)
    for center, color in [(distractor_uv, [20, 20, 240]), (target_uv, [240, 20, 20])]:
        center = np.asarray(center)
        require(center.shape == (2,) and np.issubdtype(center.dtype, np.integer)
                and np.all((center >= 3) & (center <= 27)), 'integer pixel center in [3,27]')
        u, v = center
        if observation == 'blank' or (observation == 'occluded' and np.array_equal(center, target_uv)):
            continue
        image[v-1:v+2, u-1:u+2] = color
    return image


def patches(rgb):
    require(rgb.ndim == 4 and rgb.shape[1:] == (32, 32, 3) and rgb.dtype == np.uint8 and len(rgb) > 0,
            'nonempty RGB NHWC (N,32,32,3) uint8')
    normalized = rgb.astype(np.float32)/np.float32(255)
    # Flatten each valid 3x3 RGB window in (dy,dx,channel) order; stride=2, no padding.
    windows = [normalized[:, dy:dy+30:2, dx:dx+30:2] for dy in range(3) for dx in range(3)]
    return np.concatenate(windows, axis=-1).reshape(len(rgb), 225, 27)


def initialize(seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, .2, (27, FEATURES)).astype(dtype), np.zeros(FEATURES, dtype),
            rng.normal(0, .2, FEATURES).astype(dtype)]


def forward(windows, weights):
    kernel, bias, heatmap_weight = weights
    features = np.tanh(windows @ kernel+bias)
    logits = features @ heatmap_weight
    shifted = logits-logits.max(axis=1, keepdims=True)
    log_probability = shifted-np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return np.exp(log_probability), features, log_probability


def gradients(windows, target_indices, weights):
    probability, features, log_probability = forward(windows, weights)
    loss = -np.mean(log_probability[np.arange(len(windows)), target_indices])
    delta = probability.copy()
    delta[np.arange(len(windows)), target_indices] -= 1
    delta /= len(windows)
    dhead = features.reshape(-1, FEATURES).T @ delta.reshape(-1)
    back = delta[..., None]*weights[2]*(1-features**2)
    return float(loss), [windows.reshape(-1, 27).T @ back.reshape(-1, FEATURES),
                         back.sum(axis=(0, 1)), dhead]


def coordinates(probability):
    grid_v, grid_u = np.meshgrid(np.arange(1, 30, 2), np.arange(1, 30, 2), indexing='ij')
    grid = np.stack((grid_u.ravel(), grid_v.ravel()), axis=1).astype(np.float32)
    return probability @ grid


def predict(rgb, weights):
    probability, features, _ = forward(patches(rgb), weights)
    uv = coordinates(probability)
    return pixel_to_world(uv), uv, probability, features


def make_data(output):
    rng = np.random.default_rng(11)
    centers = np.array([(u, v) for v in range(3, 28, 2) for u in range(3, 28, 2)], dtype=np.int64)
    selected = rng.permutation(len(centers))[:120]
    data = {}
    for split, indices in [('train', selected[:96]), ('validation', selected[96:])]:
        target = centers[indices]
        distractor = np.stack([centers[rng.choice(np.flatnonzero(np.max(np.abs(centers-p), axis=1) >= 6))]
                               for p in target])
        rgb = np.stack([render(t, d) for t, d in zip(target, distractor)])
        batch = dict(rgb=rgb, target_uv=target, distractor_uv=distractor, target_world_m=pixel_to_world(target),
                     target_index=((target[:, 1]-1)//2)*15+(target[:, 0]-1)//2,
                     scene_id=np.array([f'scene_{i}' for i in indices]))
        file = output/(split+'.npz')
        np.savez_compressed(file, **batch)
        with np.load(file, allow_pickle=False) as saved:
            for key, value in batch.items():
                np.testing.assert_array_equal(saved[key], value)
        data[split] = batch
    require(not set(data['train']['scene_id']) & set(data['validation']['scene_id']), 'scene overlap')
    require(not set(map(tuple, data['train']['target_uv'])) & set(map(tuple, data['validation']['target_uv'])),
            'target center leakage')
    contract = dict(schema_version='synthetic_visual_scene/1.0', source='S1.3 deterministic renderer, seed11',
                    temporal='independent static scene observations, not trajectories; no future labels in CNN input',
                    rgb=dict(shape='N,32,32,3', dtype='uint8', color_space='RGB', range=[0, 255]),
                    calibration='u=16+x/.02, v=16-y/.02; fixed world XY camera; synthetic orthographic',
                    target='red 3x3 square center; blue square distractor; non-overlapping; no agent in RGB',
                    target_uv='N,2 int64; u right, v down; centers odd integers 3..27',
                    target_world_m='N,2 float32 world x right,y up; label only',
                    target_index='N int64; flattened 15x15 grid, row-major; label only',
                    split={name:batch['scene_id'].tolist() for name,batch in data.items()},
                    raw_normalized=False, rollout='dt=.1s, ideal point world delta clip +/-.06m; budget10; final max error<=.01m')
    (output/'dataset_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    return data


def train(batch, epochs, seed):
    windows = patches(batch['rgb'])
    weights = initialize(seed)
    initial = [w.copy() for w in weights]
    first = [np.zeros_like(w) for w in weights]; second = [w.copy() for w in first]
    history = []
    for step in range(1, epochs+1):
        _, gradient = gradients(windows, batch['target_index'], weights)
        for i, grad in enumerate(gradient):
            first[i] = .9*first[i]+.1*grad
            second[i] = .999*second[i]+.001*grad**2
            weights[i] -= .03*(first[i]/(1-.9**step))/(np.sqrt(second[i]/(1-.999**step))+1e-8)
        if step == 1 or step % 50 == 0 or step == epochs:
            loss, _ = gradients(windows, batch['target_index'], weights)
            history.append(dict(epoch=step, train_cross_entropy=loss))
    require(np.linalg.norm(weights[0]-initial[0]) > 0, 'CNN kernel unchanged')
    return weights, initial, history


def evaluate(batch, rgb, weights):
    world, uv, probability, features = predict(rgb, weights)
    return dict(samples=len(rgb), world_mse_m2=float(np.mean((world-batch['target_world_m'])**2)),
                mean_euclidean_error_m=float(np.linalg.norm(world-batch['target_world_m'], axis=1).mean()),
                mean_pixel_error=float(np.linalg.norm(uv-batch['target_uv'], axis=1).mean()),
                peak_cell_accuracy=float(np.mean(probability.argmax(1) == batch['target_index'])),
                mean_peak_probability=float(probability.max(1).mean())), (world, uv, probability, features)


def rollouts(batch, weights, observation, oracle=False):
    results = []; timings = []
    for target, distractor, goal, scene in zip(batch['target_uv'], batch['distractor_uv'], batch['target_world_m'], batch['scene_id']):
        position = np.array([0, -.12], dtype=np.float32)
        trace = [position.tolist()]; commands = []; estimates = []
        for _ in range(10):
            image = render(target, distractor, observation)
            before = perf_counter_ns()
            estimate = goal.copy() if oracle else predict(image[None], weights)[0][0]
            action = np.clip(estimate-position, -STEP_M, STEP_M)
            timings.append((perf_counter_ns()-before)/1e6)
            position += action
            trace.append(position.tolist()); commands.append(action.tolist()); estimates.append(estimate.tolist())
        error = float(np.max(np.abs(position-goal)))
        success = error <= .01
        results.append(dict(scene_id=str(scene), success=success, final_error_m=error,
                            failure_stage=None if success else 'final goal error after 10-step budget',
                            truncated=True, goal_world_m=goal.tolist(), target_uv=target.tolist(),
                            trace=trace, commands=commands, estimates_world_m=estimates))
    return dict(successes=sum(r['success'] for r in results), denominator=len(results), episodes=results,
                latency_ms=dict(median=float(np.median(timings)), p95=float(np.percentile(timings, 95)), samples=len(timings),
                    scope='CNN preprocessing/inference + fixed delta controller; excludes rendering/environment; oracle skips CNN'))


def checks():
    rng = np.random.default_rng(3)
    weights = initialize(3, np.float64)
    windows = rng.normal(size=(2, 5, 27))*.2
    targets = np.array([1, 3])
    _, analytic = gradients(windows, targets, weights)
    worst = 0.
    for parameter, grad in zip(weights, analytic):
        for index in np.ndindex(parameter.shape):
            old = parameter[index]; eps = 1e-5
            parameter[index] = old+eps; plus = gradients(windows, targets, weights)[0]
            parameter[index] = old-eps; minus = gradients(windows, targets, weights)[0]
            parameter[index] = old
            worst = max(worst, abs((plus-minus)/(2*eps)-grad[index]))
    require(worst < 1e-8, 'CNN finite difference')
    rgb = np.arange(32*32*3, dtype=np.int64).reshape(32, 32, 3).astype(np.uint8)[None]
    window = patches(rgb).reshape(1, 15, 15, 27)[0, 2, 3]
    np.testing.assert_array_equal(window, (rgb[0, 4:7, 6:9].astype(np.float32)/np.float32(255)).ravel())
    probability = np.zeros((1, 225), np.float32); probability[0, 2*15+3] = 1
    np.testing.assert_array_equal(coordinates(probability), [[7, 5]])
    np.testing.assert_allclose(pixel_to_world([[7, 5]]), [[-.18, .22]], atol=1e-7)
    bad = [('wrong_dtype', rgb.astype(np.float32)), ('wrong_channels', rgb[..., :1]),
           ('wrong_resolution', rgb[:, :16]), ('empty_batch', rgb[:0])]
    rejected = [expect_rejection(name, lambda: patches(value)) for name, value in bad]
    rejected.append(expect_rejection('unknown_observation', lambda: render([3, 3], [21, 21], 'hidden')))
    return dict(gradient_max_absolute_error=worst, patch_order_and_stride=True, pixel_world_oracle=True, rejected_cases=rejected)


def png(path, rgb):
    """Write an RGB preview with stdlib only; no imaging dependency."""
    height, width, _ = rgb.shape
    def chunk(tag, payload):
        return struct.pack('!I', len(payload))+tag+payload+struct.pack('!I', zlib.crc32(tag+payload) & 0xffffffff)
    scanlines = b''.join(b'\x00'+row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
                     +chunk(b'IDAT', zlib.compress(scanlines))+chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--observation', choices=('visible', 'occluded'), default='visible')
    parser.add_argument('--output', type=Path, default=Path('tmp/s1_3'))
    args = parser.parse_args()
    require(args.epochs > 0 and not args.output.exists(), 'positive epochs and fresh output')
    verification = checks()
    args.output.mkdir(parents=True)
    data = make_data(args.output)
    weights, initial, history = train(data['train'], args.epochs, args.seed)
    np.savez_compressed(args.output/'weights.npz', **dict(zip(KEYS, weights)), initial_kernel=initial[0])
    with np.load(args.output/'weights.npz', allow_pickle=False) as saved:
        loaded = [saved[key] for key in KEYS]
    np.testing.assert_array_equal(predict(data['validation']['rgb'], weights)[0], predict(data['validation']['rgb'], loaded)[0])
    metrics = {}; results = {}
    for name, batch in data.items():
        metrics[name], (world, uv, probability, features) = evaluate(batch, batch['rgb'], weights)
        arrays = dict(scene_id=batch['scene_id'], feature_map=features.reshape(-1, 15, 15, FEATURES),
                      tokens=features, global_mean=features.mean(1), heatmap=probability.reshape(-1, 15, 15),
                      predicted_uv=uv, predicted_world_m=world)
        np.savez_compressed(args.output/(name+'_features.npz'), **arrays)
        with np.load(args.output/(name+'_features.npz'), allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
    validation = data['validation']
    for condition in ('visible', 'occluded', 'blank'):
        rgb = np.stack([render(t, d, condition) for t, d in zip(validation['target_uv'], validation['distractor_uv'])])
        metric, _ = evaluate(validation, rgb, weights)
        results[condition] = dict(metrics=metric, rollout=rollouts(validation, weights, condition))
    mean_goal = data['train']['target_world_m'].mean(0)
    baseline = dict(fit_split='train', predicted_world_m=mean_goal.tolist(),
                    validation_mse_m2=float(np.mean((validation['target_world_m']-mean_goal)**2)))
    # Interior non-overlapping object translations: global pooling loses location here.
    pair = np.stack([render([5, 5], [23, 23]), render([11, 5], [23, 23])])
    world, _, _, features = predict(pair, weights)
    gap_difference = float(np.max(np.abs(features[0].mean(0)-features[1].mean(0))))
    require(gap_difference < 1e-6, 'global mean unexpectedly encodes this interior translation')
    swapped_world = predict(pair[:1, ..., ::-1].copy(), weights)[0][0]
    swapped_error = float(np.max(np.abs(swapped_world-pixel_to_world([23, 23]))))
    preview_rows = []
    for i in range(6):
        image = validation['rgb'][i]
        heat = predict(image[None], weights)[2][0].reshape(15, 15)
        heat_rgb = np.zeros((15, 15, 3), dtype=np.uint8)
        heat_rgb[..., 0] = np.rint(255*heat/max(float(heat.max()), 1e-9)).astype(np.uint8)
        resized_heat = np.repeat(np.repeat(heat_rgb, 8, 0), 8, 1)
        heat_panel = np.full((128, 128, 3), 8, np.uint8); heat_panel[4:124, 4:124] = resized_heat
        preview_rows.append(np.concatenate((np.repeat(np.repeat(image, 4, 0), 4, 1), heat_panel,
            np.repeat(np.repeat(render(validation['target_uv'][i], validation['distractor_uv'][i], 'occluded'), 4, 0), 4, 1)), axis=1))
    png(args.output/'preview.png', np.concatenate(preview_rows, axis=0))
    report = dict(schema_version='tiny_visual_encoder/1.0', environment=dict(python=sys.version, executable=sys.executable,
                  numpy=np.__version__, numpy_path=np.__file__), epochs=args.epochs, seed=args.seed, checks=verification,
                  selected_observation=args.observation, selected_successes=results[args.observation]['rollout']['successes'],
                  parameters=sum(w.size for w in weights), history=history, metrics=metrics, conditions=results,
                  train_mean_baseline=baseline, oracle_rollout=rollouts(validation, weights, 'visible', oracle=True),
                  spatial_probe=dict(global_mean_max_difference=gap_difference, predicted_world_m=world.tolist(),
                                     color_swap_predicted_world_m=swapped_world.tolist(), color_swap_max_error_m=swapped_error),
                  contract=dict(rgb='current scene RGB NHWC uint8, red target/blue distractor, fixed camera',
                                calibration='u=16+x/.02; v=16-y/.02; object centers odd pixels 3..27',
                                features='N,15,15,4 or N,225,4 float32; spatial cells, not temporal tokens',
                                preprocessing='float32 /255; no fitted statistics', target='single red square center, no language',
                                evaluation='24 held-out unique target centers; static scene each tick, measured point state separate'),
                  limitations='One seed, 3x3 colored squares, no perspective/real camera/contact. CNN is learned localization; controller, camera calibration, soft-argmax grid and point dynamics are fixed. Softmax confidence is not visibility detection or calibrated uncertainty.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(metrics=metrics, checks=verification, parameters=report['parameters'],
        successes={key:f"{value['rollout']['successes']}/24" for key, value in results.items()},
        oracle_success=f"{report['oracle_rollout']['successes']}/24", selected=args.observation,
        train_mean_baseline=baseline, spatial_probe=report['spatial_probe']), indent=2))
    print(f'PASS CNN gradient / patch order / camera / scene split / raw RGB and weights round-trip; {args.output}')


if __name__ == '__main__':
    main()
