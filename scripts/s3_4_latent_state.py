"""Train-only PCA observation encoder + small residual latent dynamics, CPU NumPy."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from s3_1_state_dynamics import dataset, forward, gradients, metrics, transition

# Six known linear sensor projections: o = s @ SENSOR.T, all lengths in world metres.
SENSOR = np.array([[1., .3], [-.4, 1.], [.8, .6], [.6, -.8], [-.8, .6], [.3, 1.]])
SENSOR_INVERSE = np.linalg.pinv(SENSOR.T)


def observe(state):
    return (state @ SENSOR.T).astype(np.float32)


def fit_encoder(train_observations, dimension):
    if dimension not in (1, 2):
        raise ValueError('latent dimension must be 1 or 2')
    x = np.asarray(train_observations, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all():
        raise ValueError('finite [N,6] train observations required')
    mean, scale = x.mean(0), x.std(0)
    if np.any(scale <= 0):
        raise ValueError('constant observation channel')
    _, singular, vt = np.linalg.svd((x-mean)/scale, full_matrices=False)
    # Canonicalize each axis sign; sign itself has no physical meaning.
    for row in vt:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return {'observation_mean': mean, 'observation_std': scale,
            'components': vt[:dimension].T.copy(), 'singular_values': singular}


def encode(observation, encoder):
    return ((observation-encoder['observation_mean'])/encoder['observation_std']) @ encoder['components']


def decode(latent, encoder):
    return (latent @ encoder['components'].T)*encoder['observation_std']+encoder['observation_mean']


def physical_state(observation):
    """Known synthetic measurement inverse, for physical evaluation only."""
    return observation @ SENSOR_INVERSE


def fit_dynamics(latent, actions, epochs):
    z, a = latent[:, :-1].reshape(-1, latent.shape[-1]), actions.reshape(-1, 2)
    dz = (latent[:, 1:]-latent[:, :-1]).reshape(z.shape)
    x = np.concatenate([z, a], axis=1)
    stats = {'input_mean': x.mean(0), 'input_std': x.std(0),
             'delta_mean': dz.mean(0), 'delta_std': dz.std(0)}
    if np.any(stats['input_std'] <= 0) or np.any(stats['delta_std'] <= 0):
        raise ValueError('degenerate latent dynamics statistics')
    x = (x-stats['input_mean'])/stats['input_std']
    y = (dz-stats['delta_mean'])/stats['delta_std']
    dimension = latent.shape[-1]
    rng = np.random.default_rng(340)
    weights = [rng.normal(0, .2, (dimension+2, 32)), np.zeros(32),
               rng.normal(0, .15, (32, dimension)), np.zeros(dimension)]
    # Check one active coordinate per tensor against finite differences before training.
    analytic = gradients(x[:8], y[:8], weights)
    errors = []
    for w, grad in zip(weights, analytic):
        index = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
        original = w[index]
        w[index] = original+1e-5
        plus = np.mean((forward(x[:8], weights)[0]-y[:8])**2)
        w[index] = original-1e-5
        minus = np.mean((forward(x[:8], weights)[0]-y[:8])**2)
        w[index] = original
        errors.append(float(abs((plus-minus)/2e-5-grad[index])))
    assert max(errors) < 1e-7
    m, v = [np.zeros_like(w) for w in weights], [np.zeros_like(w) for w in weights]
    losses = []
    for step in range(1, epochs+1):
        for i, grad in enumerate(gradients(x, y, weights)):
            m[i] = .9*m[i]+.1*grad
            v[i] = .999*v[i]+.001*grad*grad
            weights[i] -= .01*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
        if step == 1 or step % 100 == 0 or step == epochs:
            losses.append([step, float(np.mean((forward(x, weights)[0]-y)**2))])
    return weights, stats, losses, errors


def predict_latent(latent, action, weights, stats):
    x = np.concatenate([latent, action], axis=-1)
    delta = forward((x-stats['input_mean'])/stats['input_std'], weights)[0]
    return latent+delta*stats['delta_std']+stats['delta_mean']


def rollout(initial_latent, actions, weights, stats):
    z = np.empty((len(actions), actions.shape[1]+1, initial_latent.shape[-1]), dtype=np.float64)
    z[:, 0] = initial_latent
    for t in range(actions.shape[1]):
        z[:, t+1] = predict_latent(z[:, t], actions[:, t], weights, stats)
    return z


def evaluate(states, observations, actions, encoder, weights, stats):
    z = encode(observations, encoder)
    reconstructed = decode(z, encoder)
    current = z[:, :-1].reshape(-1, z.shape[-1])
    command = actions.reshape(-1, 2)
    target_z = z[:, 1:].reshape(current.shape)
    predicted = predict_latent(current, command, weights, stats)
    zero_action = predict_latent(current, np.zeros_like(command), weights, stats)
    target_obs, target_state = observations[:, 1:].reshape(-1, 6), states[:, 1:].reshape(-1, 2)
    decoded = decode(predicted, encoder)
    ar = rollout(z[:, 0], actions, weights, stats)
    ar_obs = decode(ar, encoder)
    report = {
        'episodes': len(states), 'observations': len(states)*21, 'transitions': len(states)*20,
        'reconstruction_observation': metrics(reconstructed, observations),
        'reconstruction_physical': metrics(physical_state(reconstructed), states),
        'latent_prediction_mse_dimensionless': float(np.mean((predicted-target_z)**2)),
        'latent_persistence_mse_dimensionless': float(np.mean((current-target_z)**2)),
        'decoded_next_observation': metrics(decoded, target_obs),
        'decoded_next_physical': metrics(physical_state(decoded), target_state),
        'future_encoding_reconstruction_reference': metrics(physical_state(decode(target_z, encoder)), target_state),
        'latent_persistence_decoded_physical': metrics(physical_state(decode(current, encoder)), target_state),
        'zero_action_ablation_physical': metrics(physical_state(decode(zero_action, encoder)), target_state),
        'privileged_ideal_addition': metrics(states[:, :-1].reshape(-1, 2)+command, target_state),
        'ar20_endpoint_physical': metrics(physical_state(ar_obs[:, -1]), states[:, -1]),
        'ar20_prefix_physical': metrics(physical_state(ar_obs[:, 1:]), states[:, 1:])}
    arrays = {'latent': z, 'reconstructed_observation_m': reconstructed,
              'predicted_next_latent': predicted, 'target_next_latent': target_z,
              'predicted_next_observation_m': decoded,
              'predicted_next_state_m': physical_state(decoded), 'ar_latent': ar,
              'ar_observation_m': ar_obs, 'ar_state_m': physical_state(ar_obs)}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('nonfinite model evaluation')
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latent-dim', type=int, choices=(1, 2), default=2)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=Path('tmp/s3_4'))
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10000:
        parser.error('epochs must be 1..10000')
    args.output.mkdir(parents=True, exist_ok=False)
    states, actions = dataset()
    observations = observe(states)
    ids = np.random.default_rng(32).permutation(40)
    train_ids, val_ids = ids[:30], ids[30:]
    assert not set(train_ids) & set(val_ids)
    # Each original train observation, including terminal, appears exactly once here.
    encoder = fit_encoder(observations[train_ids].reshape(-1, 6), args.latent_dim)
    train_z = encode(observations[train_ids], encoder)
    weights, stats, losses, gradient_errors = fit_dynamics(train_z, actions[train_ids], args.epochs)
    np.savez_compressed(args.output/'model.npz', **encoder, **stats,
                        **{k: w for k, w in zip(('w1', 'b1', 'w2', 'b2'), weights)},
                        sensor_matrix=SENSOR)
    np.savez_compressed(args.output/'episodes.npz', state_m=states, observation_m=observations,
                        action_m=actions, train_episode_ids=train_ids, validation_episode_ids=val_ids,
                        timestamp_s=np.arange(21, dtype=np.float64)*.1)
    report = {'schema': 's3_4_latent_state/1.0', 'latent_dim': args.latent_dim, 'epochs': args.epochs,
              'dynamics_parameters': sum(w.size for w in weights),
              'encoder_fit_observations': 630, 'dynamics_fit_transitions': 600,
              'train_explained_variance_fraction': float(np.sum(encoder['singular_values'][:args.latent_dim]**2)
                                                       /np.sum(encoder['singular_values']**2)),
              'gradient_absolute_errors': gradient_errors, 'normalized_train_losses': losses,
              'evaluation': {}}
    arrays = {}
    for name, indices in [('train', train_ids), ('validation', val_ids)]:
        report['evaluation'][name], values = evaluate(states[indices], observations[indices], actions[indices],
                                                      encoder, weights, stats)
        if name == 'validation':
            arrays = values
    # Representation ambiguity: one removed PCA direction gives equal z but different physical states.
    if args.latent_dim == 1:
        # Null direction in physical XY of this one-dimensional encoder.
        direction = SENSOR.T @ (encoder['components'][:, 0]/encoder['observation_std'])
        null = np.array([-direction[1], direction[0]])
        null /= np.linalg.norm(null)
        paired = np.array([-.05*null, .05*null])
        exact_obs = paired @ SENSOR.T  # float64 avoids rendering-rounding ambiguity in this algebra probe.
        paired_z = encode(exact_obs, encoder)
        assert np.max(np.abs(paired_z[0]-paired_z[1])) < 1e-12
        common_action = np.tile([.03, .02], (2, 1)).astype(np.float32)
        true_next = transition(paired.astype(np.float32), common_action)
        report['aliasing_probe'] = {'states_m': paired.tolist(), 'latent': paired_z.tolist(),
                                    'same_action_m': common_action.tolist(),
                                    'true_next_state_m': true_next.tolist(),
                                    'next_latent_difference': float(np.linalg.norm(encode(observe(true_next), encoder)[0]
                                                                                  -encode(observe(true_next), encoder)[1]))}
    rx, ra = observations[val_ids[0], :1], actions[val_ids[0], :1]
    def inference():
        return decode(predict_latent(encode(rx, encoder), ra, weights, stats), encoder)
    for _ in range(20):
        inference()
    times = []
    for _ in range(200):
        start = perf_counter_ns()
        inference()
        times.append((perf_counter_ns()-start)/1e6)
    report['batch1_encode_predict_decode_ms'] = {'p50': float(np.percentile(times, 50)),
                                                'p95': float(np.percentile(times, 95)), 'warmup': 20, 'samples': 200}
    report['limitations'] = 'Linear noiseless known sensor map; PCA is not visual representation learning; no robot success.'
    np.savez_compressed(args.output/'validation_predictions.npz', **arrays)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    contract = {'schema': 's3_4_linear_sensor_latent/1.0', 'dynamics': 's3_1_xy_command_dynamics/1.0 nominal generator',
                'observation': 'float32 [40,21,6], known linear world-XY projections [m], o=s@SENSOR.T',
                'state': 'float32 [40,21,2], world XY [m], only generator/evaluation; not model input',
                'action': 'float32 [40,20,2], commanded world XY displacement [m], dt=.1s',
                'time': 'float64 [21], relative seconds;20 transitions/21 observations; time truncation, no success label',
                'split': 'data seed31, episode split seed32, train30/validation10; same as S3.1',
                'encoder': 'train630 unique observations normalized per channel; PCA top d=1 or2; frozen before dynamics',
                'decoder': 'inverse PCA projection and observation normalization, output six sensor lengths [m]',
                'latent': 'float64 [episode,21,d], dimensionless, learned axes without direct metre/frame meaning',
                'dynamics_model': 'float64 [z,action] -> normalized delta_z -> z_next; input/delta stats fit train600 only',
                'physical_evaluation': 'decoded sensor vector @ known SENSOR pseudoinverse; privileged synthetic geometry',
                'modality': 'six numeric sensor projections only; no image/language/force/velocity/noise',
                'prediction_boundary': 'future encoding only target/diagnostic; AR initialized from o0, no future feedback'}
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    v = report['evaluation']['validation']
    print('latent dimension / explained variance:', args.latent_dim, report['train_explained_variance_fraction'])
    for key in ('reconstruction_physical', 'decoded_next_physical', 'future_encoding_reconstruction_reference',
                'latent_persistence_decoded_physical', 'zero_action_ablation_physical', 'ar20_endpoint_physical'):
        print(key, 'RMSE [mm]', v[key]['rmse_m']*1000)
    print('latent prediction/persistence MSE', v['latent_prediction_mse_dimensionless'], v['latent_persistence_mse_dimensionless'])
    print('gradient errors', gradient_errors)


if __name__ == '__main__':
    main()
