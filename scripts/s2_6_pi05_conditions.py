"""CPU information-path probes; no trained VLA, tokenizer vocabulary or robot rollout."""
import argparse
import json
from pathlib import Path

import numpy as np


def state_bins(state):
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError('state must be a finite vector')
    # Exact selected upstream expression: deliberately no silent clipping.
    return np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1


def hybrid_mask(prefix=3, discrete=2, continuous=3):
    """Rows query, columns key; discrete slots stand for shifted AR inputs."""
    n = prefix + discrete + continuous
    mask = np.zeros((n, n), dtype=bool)
    mask[:prefix, :prefix] = True
    mask[prefix:prefix + discrete, :prefix] = True
    mask[prefix:prefix + discrete, prefix:prefix + discrete] = np.tril(np.ones((discrete, discrete), bool))
    mask[prefix + discrete:, :prefix] = True
    mask[prefix + discrete:, prefix + discrete:] = True
    return mask


def mix(x, mask):
    return x + mask @ x / mask.sum(axis=1, keepdims=True)


def adaptive_norm(x, cond, weight):
    scale, shift, gate = np.split(cond @ weight, 3)
    norm = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    return norm * (1 + scale) + shift, gate


def run(delta, output):
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('delta must be finite and positive')
    if output.exists():
        raise FileExistsError(output)
    raw = np.array([0.0, 0.0])  # world XY metres, current observation only
    changed = raw + np.array([delta, 0.0])
    # Fixed teaching scale 1m, not fitted data statistics.
    b0, b1 = state_bins(raw / 1.0), state_bins(changed / 1.0)
    mask = hybrid_mask()
    x = np.arange(32, dtype=np.float64).reshape(8, 4) / 32
    perturbed = x.copy()
    perturbed[3:5] += 100  # privileged FAST-label proxy
    y = mix(mix(x, mask), mask)
    yp = mix(mix(perturbed, mask), mask)
    isolated = float(np.max(np.abs(y[5:] - yp[5:])))
    assert isolated == 0
    wrong = mask.copy()
    wrong[5:, 3:5] = True
    leaked = float(np.max(np.abs(mix(mix(x, wrong), wrong)[5:] - mix(mix(perturbed, wrong), wrong)[5:])))
    assert leaked > 0
    # Seeded nonzero modulation illustrates a possible learned dependency, not trained weights.
    weight = np.random.default_rng(41).normal(0, .1, (4, 12))
    c0, c1 = np.zeros(4), np.ones(4)
    z0, _ = adaptive_norm(x[5:], c0, weight)
    z1, _ = adaptive_norm(x[5:], c1, weight)
    zi0, gi0 = adaptive_norm(x[5:], c0, np.zeros_like(weight))
    zi1, gi1 = adaptive_norm(x[5:], c1, np.zeros_like(weight))
    assert np.array_equal(zi0, zi1) and np.all(gi0 == 0) and np.all(gi1 == 0)
    assert np.max(np.abs(z1 - z0)) > 0
    rejected = 0
    for bad in ([np.nan], [[0.0]], [np.inf]):
        try:
            state_bins(bad)
        except ValueError:
            rejected += 1
    assert rejected == 3
    report = dict(delta_m=delta, raw_state_m=raw.tolist(), changed_state_m=changed.tolist(),
                  state_bins=b0.tolist(), changed_bins=b1.tolist(),
                  boundary_probe=state_bins(np.array([-1.01, -1, 0, 1, 1.01])).tolist(),
                  mask_edges=int(mask.sum()), isolated_max_difference=isolated,
                  wrong_mask_leak_difference=leaked,
                  nonzero_modulation_difference=float(np.max(np.abs(z1-z0))),
                  zero_init_condition_difference=float(np.max(np.abs(zi1-zi0))),
                  rejected_states=rejected,
                  scope='Handwritten probes only; no text IDs, trained policy, robot success or latency.')
    output.mkdir(parents=True)
    arrays = dict(raw_state_m=raw, changed_state_m=changed, mask=mask,
                  wrong_mask=wrong, inputs=x, outputs=y, perturbed_outputs=yp,
                  norm_condition0=z0, norm_condition1=z1)
    np.savez(output / 'trace.npz', **arrays)
    with np.load(output / 'trace.npz', allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for key, value in arrays.items():
            assert np.array_equal(loaded[key], value)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta', type=float, default=.001, help='world X perturbation in metres')
    parser.add_argument('--output', type=Path, default=Path('tmp/s2_6'))
    args = parser.parse_args()
    run(args.delta, args.output)
