"""Octo-inspired mask/readout and diffusion algebra probes, not an Octo policy."""
import argparse
import json
from pathlib import Path

import numpy as np


def fixture(wrist):
    # task, then primary/wrist/readout at t=0 and t=1; embeddings have no physical units.
    x = np.random.default_rng(7).normal(size=(7, 4)).astype(np.float32)
    kinds = ['task', 'obs', 'obs', 'readout', 'obs', 'obs', 'readout']
    times = np.array([-1, 0, 0, 0, 1, 1, 1])
    valid = np.ones(7, dtype=bool)
    if wrist == 'missing':
        valid[[2, 5]] = False
    mask = np.zeros((7, 7), dtype=bool)
    for q, kind in enumerate(kinds):
        for k, key_kind in enumerate(kinds):
            if kind == 'task':
                allowed = key_kind == 'task'
            elif kind == 'obs':
                allowed = key_kind in ('task', 'obs') and times[k] <= times[q]
            else:
                allowed = times[k] <= times[q]  # own readout group may attend causally to itself
            mask[q, k] = valid[q] and valid[k] and allowed
    return x, valid, mask


def backbone(x, valid, mask):
    """Two untrained identity-QKV residual attention blocks; invalid payload is sanitized first."""
    x = np.where(valid[:, None], x, np.float32(0))
    if not np.isfinite(x).all():
        raise ValueError('nonfinite valid token')
    for _ in range(2):
        score = x @ x.T / np.sqrt(np.float32(x.shape[-1]))
        weights = np.zeros_like(score)
        for q in range(len(x)):
            keys = mask[q]
            if keys.any():
                logits = score[q, keys]
                exps = np.exp(logits - logits.max())
                weights[q, keys] = exps / exps.sum()
        x = np.where(valid[:, None], x + weights @ x, np.float32(0))
    return x, weights


def diffusion_probe():
    # A physical fixture, not a policy target learned by the random backbone.
    raw = np.array([[[[.01, -.02], [.02, 0], [0, .03]],
                     [[.02, .01], [0, .02], [-.01, 0]]]], dtype=np.float32)
    scale_m = np.float32(.06)  # declared fixed unit conversion, not a fitted statistic
    clean = raw / scale_m
    noise = np.random.default_rng(11).normal(size=clean.shape).astype(np.float32)
    alpha_bar = np.float32(.64)
    noisy = np.sqrt(alpha_bar)*clean + np.sqrt(1-alpha_bar)*noise
    # Oracle knows injected noise: validates forward algebra only, cannot sample a learned policy.
    recovered = (noisy - np.sqrt(1-alpha_bar)*noise) / np.sqrt(alpha_bar)
    np.testing.assert_allclose(recovered, clean, atol=1e-7)
    return dict(raw_action_m=raw, normalized_action=clean, injected_noise=noise,
                noisy_action=noisy, oracle_recovered=recovered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrist', choices=('present', 'missing'), default='present')
    parser.add_argument('--output', type=Path, default=Path('tmp/s2_2'))
    args = parser.parse_args()
    outputs, cases = {}, {}
    for wrist in ('present', 'missing'):
        x, valid, mask = fixture(wrist)
        out, weights = backbone(x, valid, mask)
        future = x.copy(); future[4:6] += 2
        future_out, _ = backbone(future, valid, mask)
        np.testing.assert_array_equal(out[:4], future_out[:4])
        readout = x.copy(); readout[[3, 6]] += 2
        readout_out, _ = backbone(readout, valid, mask)
        np.testing.assert_array_equal(out[[0, 1, 2, 4, 5]], readout_out[[0, 1, 2, 4, 5]])
        task = x.copy(); task[0] += .2
        task_out, _ = backbone(task, valid, mask)
        assert not np.allclose(out[6], task_out[6])
        assert np.all(weights[~mask] == 0)
        if wrist == 'missing':
            poison = x.copy(); poison[[2, 5]] = np.nan
            poisoned_out, _ = backbone(poison, valid, mask)
            np.testing.assert_array_equal(out, poisoned_out)
        rng = np.random.default_rng(13)
        head2 = rng.normal(size=(4, 3*2)).astype(np.float32)
        head3 = rng.normal(size=(4, 3*3)).astype(np.float32)
        # These are unitless random projection probes, not physical actions or diffusion heads.
        probe2 = (out[6] @ head2).reshape(3, 2)
        probe3 = (out[6] @ head3).reshape(3, 3)
        outputs.update({f'{wrist}_{k}': v for k, v in dict(
            input_tokens=x, valid=valid, attention_mask=mask, embeddings=out,
            last_layer_attention=weights, head2_probe=probe2, head3_probe=probe3).items()})
        cases[wrist] = dict(valid_tokens=int(valid.sum()), token_shape=list(x.shape),
                            allowed_edges=int(mask.sum()), readout_t1=out[6].tolist(),
                            head2_shape=list(probe2.shape), head3_shape=list(probe3.shape))
    assert not np.allclose(outputs['present_embeddings'][6], outputs['missing_embeddings'][6])
    x, valid, mask = fixture('present'); x[1] = np.nan
    try:
        backbone(x, valid, mask)
    except ValueError:
        pass
    else:
        raise AssertionError('valid NaN accepted')
    outputs.update(diffusion_probe())
    report = dict(schema_version='octo_concept_trace/1.0', learned_policy=False,
                  selected_wrist=args.wrist, selected=cases[args.wrist], cases=cases,
                  seed=dict(tokens=7, projection_heads=13, diffusion_noise=11),
                  history=2, action_horizon=3, token_dim=4,
                  checks=['future isolation', 'readout cannot affect observations',
                          'task influences readout', 'masked attention zero',
                          'missing NaN payload invariance', 'valid NaN rejection',
                          'head dimension adaptation shape', 'oracle forward-noise inversion'],
                  evidence_limit='untrained attention and algebra only; no diffusion sampling, transfer, control or model latency')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'trace.npz', **outputs)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    with np.load(args.output/'trace.npz', allow_pickle=False) as saved:
        for k, v in outputs.items():
            np.testing.assert_array_equal(saved[k], v)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
