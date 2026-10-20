"""Analytic action-conditioned branching video probe; no learned video model."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

STEPS = 8
DT = .1
DRIFT = .0075
BACKGROUND = np.array([12, 16, 24], dtype=np.uint8)
OBJECT = np.array([245, 100, 60], dtype=np.uint8)


def positions(initial, actions, modes):
    """modes [...,8] chooses branch at each absolute future time; constant is valid."""
    base = initial.astype(np.float64)+np.cumsum(actions.astype(np.float64), axis=-2)
    result = np.broadcast_to(base, modes.shape+(2,)).copy()
    result[..., 1] += modes*DRIFT*np.arange(1, STEPS+1)
    start = np.broadcast_to(initial, result.shape[:-2]+(1, 2))
    return np.concatenate([start, result], axis=-2)


def render(state):
    result = np.empty(state.shape[:-1]+(32, 32, 3), dtype=np.uint8)
    result[...] = BACKGROUND
    flat = result.reshape(-1, 32, 32, 3)
    for frame, p in zip(flat, state.reshape(-1, 2)):
        col = int(np.rint((p[0]+.25)/.5*31))
        row = int(np.rint((.25-p[1])/.5*31))
        if not 1 <= row < 31 or not 1 <= col < 31:
            raise ValueError('out of renderer range; no silent clipping')
        frame[row-1:row+2, col-1:col+2] = OBJECT
    return result


def temporal_violation(state, action):
    delta = np.diff(state, axis=-2)-action
    return ((np.abs(delta[..., 0]) > 1e-8) | (np.abs(np.abs(delta[..., 1])-DRIFT) > 1e-8))


def pixel_mse(prediction, truth):
    return float(np.mean(((prediction.astype(np.float64)-truth)/255)**2))


def component_count(frame):
    mask = np.any(np.abs(frame.astype(float)-BACKGROUND) > 1, axis=-1)
    visited = np.zeros_like(mask)
    count = 0
    for r, c in np.argwhere(mask):
        if visited[r, c]:
            continue
        count += 1
        stack = [(int(r), int(c))]
        visited[r, c] = True
        while stack:
            rr, cc = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr+dr, cc+dc
                if 0 <= nr < 32 and 0 <= nc < 32 and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
    return count


def build(samples):
    initial = np.random.default_rng(380).uniform(-.015, .015, (20, 2)).astype(np.float32)
    actions = np.zeros((20, 2, STEPS, 2), dtype=np.float32)
    actions[:, 0, :, 0] = -.015
    actions[:, 1, :, 0] = .015
    truth, persistent, flicker, mean_state, ignored = [], [], [], [], []
    signs_all = []
    for group in range(20):
        signs = np.stack([np.where(np.random.default_rng(38000+group*1000+k).random(STEPS) < .5, -1, 1)
                          for k in range(samples)])
        signs_all.append(signs)
        one_truth, one_persistent, one_flicker, one_mean, one_ignored = [], [], [], [], []
        for direction in range(2):
            a = actions[group, direction]
            one_truth.append(positions(initial[group], a, np.tile(np.array([-1, 1])[:, None], (1, STEPS))))
            fixed = np.repeat(signs[:, :1], STEPS, axis=1)
            one_persistent.append(positions(initial[group], a, fixed))
            one_flicker.append(positions(initial[group], a, signs))
            one_mean.append(positions(initial[group], a, np.zeros(STEPS)))
            one_ignored.append(positions(initial[group], np.zeros_like(a), fixed))
        truth.append(one_truth); persistent.append(one_persistent); flicker.append(one_flicker)
        mean_state.append(one_mean); ignored.append(one_ignored)
    return {'initial_state_m': initial, 'action_m': actions, 'truth_state_m': np.asarray(truth),
            'persistent_state_m': np.asarray(persistent), 'frame_resampled_state_m': np.asarray(flicker),
            'mean_state_m': np.asarray(mean_state), 'action_ignored_state_m': np.asarray(ignored),
            'sample_signs': np.asarray(signs_all, dtype=np.int8)}


def sample_summary(prediction, truth, action):
    # Average across ALL truth branches/samples, and an explicitly privileged best-of-K whole path.
    error = prediction[:, :, None, :, 1:]-truth[:, :, :, None, 1:]
    path_mse = np.mean(error**2, axis=(-1, -2))
    matches = np.max(np.abs(error), axis=(-1, -2)) <= 1e-8
    violation = temporal_violation(prediction, action[:, :, None])
    return {'all_sample_path_mse_m2': float(path_mse.mean()),
            'best_of_k_path_mse_m2': float(path_mse.min(-1).mean()),
            'exact_support_coverage': float(matches.any(-1).mean()),
            'temporal_violation_fraction': float(violation.mean()),
            'any_invalid_trajectory_fraction': float(violation.any(-1).mean())}


def previews(output, truth_images, persistent_images, flicker_images, mean_images, mean_pixel_images):
    # Fixed reset0/right-action visual, never pick the best generated sample using truth.
    rows = [('truth branch -', truth_images[0, 1, 0]), ('truth branch +', truth_images[0, 1, 1]),
            ('sample0 persistent', persistent_images[0, 1, 0]), ('sample0 frame resample', flicker_images[0, 1, 0]),
            ('mean latent -> frame', mean_images[0, 1]), ('conditional mean pixels', mean_pixel_images[0, 1])]
    width, scale = 180, 3
    sheet = Image.new('RGB', (width+9*32*scale, len(rows)*(32*scale+8)), '#20242c')
    draw = ImageDraw.Draw(sheet)
    for row, (label, sequence) in enumerate(rows):
        draw.text((5, row*104+35), label, fill='white')
        for t, frame in enumerate(sequence):
            tile = Image.fromarray(np.rint(frame).astype(np.uint8)).resize((96, 96), Image.Resampling.NEAREST)
            sheet.paste(tile, (width+t*96, row*104))
    sheet.save(output/'comparison.png')
    frames = []
    for t in range(9):
        frame = Image.new('RGB', (width+96, len(rows)*104), '#20242c')
        draw = ImageDraw.Draw(frame)
        for row, (label, sequence) in enumerate(rows):
            draw.text((5, row*104+35), label, fill='white')
            frame.paste(Image.fromarray(np.rint(sequence[t]).astype(np.uint8)).resize((96, 96), Image.Resampling.NEAREST), (width, row*104))
        frames.append(frame)
    frames[0].save(output/'comparison.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--output', type=Path, default=Path('tmp/s3_8'))
    args = parser.parse_args()
    if not 1 <= args.samples <= 64:
        parser.error('samples must be 1..64')
    args.output.mkdir(parents=True, exist_ok=False)
    arrays = build(args.samples)
    truth, action = arrays['truth_state_m'], arrays['action_m']
    truth_images = render(truth)
    persistent_images = render(arrays['persistent_state_m'])
    flicker_images = render(arrays['frame_resampled_state_m'])
    mean_images = render(arrays['mean_state_m'])
    mean_pixel_images = truth_images.astype(np.float64).mean(2)
    copy_images = np.broadcast_to(truth_images[:, :, :1, :1], truth_images.shape)
    mean_state_mse = float(np.mean((arrays['mean_state_m'][:, :, None, 1:]-truth[:, :, :, 1:])**2))
    report = {'schema': 's3_8_generative_concepts/1.0', 'samples': args.samples,
              'evaluation': {'resets': 20, 'action_contexts': 40, 'true_trajectories': 80, 'future_steps': 8},
              'persistent': sample_summary(arrays['persistent_state_m'], truth, action),
              'frame_resampled': sample_summary(arrays['frame_resampled_state_m'], truth, action),
              'action_ignored': sample_summary(arrays['action_ignored_state_m'], truth, action),
              'mean_state_path_mse_m2': mean_state_mse,
              'mean_state_temporal_violation_fraction': float(temporal_violation(arrays['mean_state_m'], action).mean()),
              'pixel_mse_0_1': {
                  'conditional_mean_pixels': pixel_mse(mean_pixel_images[:, :, None, 1:], truth_images[:, :, :, 1:]),
                  'mean_state_render': pixel_mse(mean_images[:, :, None, 1:], truth_images[:, :, :, 1:]),
                  'copy_initial_frame': pixel_mse(copy_images[:, :, :, 1:], truth_images[:, :, :, 1:])},
              'last_frame_components': {'truth': component_count(truth_images[0, 1, 0, -1]),
                                        'mean_pixels': component_count(mean_pixel_images[0, 1, -1])},
              'paired_action_endpoint_separation_m': float(np.mean(np.linalg.norm(truth[:, 1, :, -1]-truth[:, 0, :, -1], axis=-1))),
              'persistent_plus_branch_fraction': float(np.mean(arrays['sample_signs'][:, :, 0] == 1)),
              'limitations': 'Known analytic sampler/renderer and two-point hidden-mode prior; no learned encoder/dynamics/video generator, no robot success or calibrated physical risk.'}
    np.savez_compressed(args.output/'trajectories.npz', **arrays, timestamp_s=np.arange(9, dtype=np.float64)*DT,
                        truth_image_rgb=truth_images, conditional_mean_image=mean_pixel_images)
    previews(args.output, truth_images, persistent_images, flicker_images, mean_images, mean_pixel_images)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    contract = {'schema': 's3_8_branching_video/1.0', 'state': 'world XY positions[m], float64 analytic rollout from float32 initial state',
                'action': 'float32 [20,2,8,2], commanded world XY displacement[m]; ideal x displacement; dt=.1s',
                'truth': 'float64 [20,2 actions,2 hidden branches,9,2]; branch b=-1/+1 fixed for whole episode, equal prior',
                'dynamics': 'p_h=p0+sum actions[:h]+[0,b*.0075m*h]; hidden b not visible at initial frame',
                'samples': f'float64 [20,2,{args.samples},9,2]; no future truth input; seed38000+reset*1000+sample',
                'stochastic_meaning': 'hidden fixed episode mode, not per-step iid process noise or model ensemble disagreement',
                'images': 'uint8 [20,2,2,9,32,32,3], 0..255 RGB; world bounds +/- .25m, +x right/+y up, one 3x3 square',
                'renderer': 'handwritten known state->pixels; no learned visual encoder or neural decoder',
                'context': 'initial frame identical across actions/branches within a reset; action affects t->t+1',
                'boundary': '8 transitions/9 observations, time truncation, no task success labels',
                'evaluation': 'all20 reset probes, both actions/branches; no training or fitted normalization; best-of-K uses true future and is diagnostic only',
                'mean_pixels': 'exact two-branch conditional pixel mean, analytical reference not deployable learned output',
                'scope': 'conceptual counterexamples only; no Dreamer/GAIA/Genie inference/training or fidelity claim'}
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
