"""Tiny learned word embeddings with explicit PAD, UNK and masked mean pooling."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require
from s0_5_action_representation import expect_rejection
from s1_1_language_state_policy import dataset, GOALS, STEP

PAD, UNK = 0, 1
MAX_TOKENS = 4
DIM = 4
KEYS = ('embedding', 'w1', 'b1', 'w2', 'b2')
PROBES = [('move left', 0), ('move right', 1)]
UNKNOWN = [('move port', 0), ('move starboard', 1), ('please move left', 0), ('please move right', 1)]


def words(text):
    require(isinstance(text, str), 'instruction must be string')
    tokens = text.lower().split()
    require(0 < len(tokens) <= MAX_TOKENS, 'nonempty instruction, at most four whitespace tokens; no silent truncation')
    return tokens


def fit_vocab(train_texts):
    return {'<PAD>': PAD, '<UNK>': UNK, **{word: i+2 for i, word in enumerate(sorted({w for text in train_texts for w in words(text)}))}}


def tokenize(texts, vocab, unk_policy='reject'):
    require(unk_policy in ('reject', 'map'), 'known UNK policy')
    ids = np.full((len(texts), MAX_TOKENS), PAD, dtype=np.int64)
    mask = np.zeros_like(ids, dtype=bool)
    unknowns = []
    for row, text in enumerate(texts):
        tokens = words(text)
        unknown = [w for w in tokens if w not in vocab]
        require(unk_policy == 'map' or not unknown, f'unknown words: {unknown}')
        for column, word in enumerate(tokens):
            ids[row, column] = vocab.get(word, UNK)
            mask[row, column] = True
        unknowns.append(unknown)
    return ids, mask, unknowns


def forward(state_scaled, ids, mask, weights):
    embedding, w1, b1, w2, b2 = weights
    require(ids.dtype == np.int64 and ids.ndim == 2 and mask.shape == ids.shape and mask.dtype == bool, 'int64 IDs and bool mask')
    require(np.all((ids >= 0) & (ids < len(embedding))), 'token ID out of vocabulary')
    require(np.all(mask.sum(1) > 0) and np.array_equal(mask, ids != PAD), 'PAD mask mismatch or empty sentence')
    require(state_scaled.shape == (len(ids), 2) and np.isfinite(state_scaled).all(), 'finite batch XY')
    looked_up = embedding[ids]
    # Select before arithmetic: even NaN in PAD row must not contaminate pooling.
    pooled = np.where(mask[..., None], looked_up, 0).sum(1)/mask.sum(1, keepdims=True)
    pooled = pooled.astype(embedding.dtype)
    fused = np.concatenate((state_scaled, pooled), axis=1)
    hidden = np.tanh(fused @ w1+b1)
    return hidden @ w2+b2, (fused, hidden, pooled)


def gradients(state, ids, mask, target, weights):
    predicted, (fused, hidden, _) = forward(state, ids, mask, weights)
    delta = 2*(predicted-target)/target.size
    back = (delta @ weights[3].T)*(1-hidden*hidden)
    dpool = (back @ weights[1].T)[:, 2:]
    dembedding = np.zeros_like(weights[0])
    per_token = dpool/mask.sum(1, keepdims=True)
    rows, columns = np.where(mask)
    # Repeated IDs must accumulate gradients, including repeated words in one sentence.
    np.add.at(dembedding, ids[rows, columns], per_token[rows])
    return [dembedding, fused.T @ back, back.sum(0), hidden.T @ delta, delta.sum(0)]


def initialize(vocab_size, seed, dtype=np.float32, hidden=32):
    rng = np.random.default_rng(seed)
    embedding = rng.normal(0, .2, (vocab_size, DIM)).astype(dtype)
    embedding[[PAD, UNK]] = 0
    return [embedding, rng.normal(0, .3, (2+DIM, hidden)).astype(dtype), np.zeros(hidden, dtype),
            rng.normal(0, .1, (hidden, 2)).astype(dtype), np.zeros(2, dtype)]


def train(batch, vocab, epochs, seed):
    state, action, texts, _ = batch
    ids, mask, _ = tokenize(texts, vocab)
    weights = initialize(len(vocab), seed)
    initial_embedding = weights[0].copy()
    first = [np.zeros_like(w) for w in weights]; second = [g.copy() for g in first]
    history = []
    for step in range(1, epochs+1):
        for i, grad in enumerate(gradients(state/.3, ids, mask, action/STEP, weights)):
            first[i] = .9*first[i]+.1*grad
            second[i] = .999*second[i]+.001*grad**2
            weights[i] -= .01*(first[i]/(1-.9**step))/(np.sqrt(second[i]/(1-.999**step))+1e-8)
        if step == 1 or step % 100 == 0 or step == epochs:
            prediction = forward(state/.3, ids, mask, weights)[0]*STEP
            history.append(dict(epoch=step, train_raw_mse_m2=float(np.mean((prediction-action)**2))))
    np.testing.assert_array_equal(weights[0][[PAD, UNK]], initial_embedding[[PAD, UNK]])
    require(np.linalg.norm(weights[0][2:]-initial_embedding[2:]) > 0, 'embedding did not learn')
    return weights, history, initial_embedding


def read_batches(output):
    batches = {}
    for split in ('train', 'validation'):
        state = []; actions = []; texts = []; groups = []
        for file in sorted((output/'episodes').glob('*/episode.json')):
            meta = json.loads(file.read_text())
            if meta['provenance']['split'] != split:
                continue
            with np.load(file.with_suffix('.npz'), allow_pickle=False) as arrays:
                length = len(arrays['action'])
                state.extend(arrays['proprioception'][:-1]); actions.extend(arrays['action'])
                texts.extend([meta['language_instruction']]*length)
                groups.extend([meta['provenance']['group_id']]*length)
        batches[split] = (np.asarray(state), np.asarray(actions), texts, groups)
    require(not set(batches['train'][3]) & set(batches['validation'][3]), 'group split overlap')
    return batches


def rollout(starts, probes, vocab, weights, unk_policy):
    results = []; latencies = []
    for group, start in enumerate(starts, 24):
        for text, goal_index in probes:
            state = start.copy(); trace = [state.tolist()]; commands = []
            try:
                for _ in range(10):
                    before = perf_counter_ns()
                    ids, mask, unknown = tokenize([text], vocab, unk_policy)
                    action = np.clip(forward(state[None]/.3, ids, mask, weights)[0][0]*STEP, -STEP, STEP)
                    latencies.append((perf_counter_ns()-before)/1e6)
                    state += action; commands.append(action.tolist()); trace.append(state.tolist())
                stage = None
            except ValueError as error:
                stage = 'tokenization rejected unknown word'
                rejection = str(error)
            error_m = float(np.max(np.abs(state-GOALS[goal_index])))
            success = stage is None and error_m <= .01
            results.append(dict(group=group, text=text, evaluator_goal_index=goal_index,
                                success=success, final_error_m=error_m,
                                failure_stage=stage or (None if success else 'final goal error after 10-step budget'),
                                rejection=rejection if stage else None, truncated=stage is None,
                                trace=trace, commands=commands))
    timing = None if not latencies else dict(median=float(np.median(latencies)), p95=float(np.percentile(latencies, 95)),
        samples=len(latencies), scope='tokenization + embedding/pooling + MLP + clipping; excludes environment; successful calls only')
    return dict(successes=sum(r['success'] for r in results), denominator=len(results),
                rejected=sum(r['rejection'] is not None for r in results), latency_ms=timing, episodes=results)


def checks(vocab):
    weights = initialize(len(vocab), 2, np.float64, hidden=3)
    ids, mask, _ = tokenize(['move left left', 'move right'], vocab)
    state = np.array([[.1, -.2], [.2, .3]])
    target = np.array([[.2, .1], [-.1, .2]])
    analytic = gradients(state, ids, mask, target, weights)
    worst = 0.
    for parameter, grad in zip(weights, analytic):
        for index in np.ndindex(parameter.shape):
            old = parameter[index]; eps = 1e-5
            parameter[index] = old+eps; plus = np.mean((forward(state, ids, mask, weights)[0]-target)**2)
            parameter[index] = old-eps; minus = np.mean((forward(state, ids, mask, weights)[0]-target)**2)
            parameter[index] = old
            worst = max(worst, abs((plus-minus)/(2*eps)-grad[index]))
    require(worst < 1e-8, 'embedding/MLP finite difference')
    reference = forward(state, ids, mask, weights)[0]
    weights[0][PAD] = np.nan
    np.testing.assert_array_equal(reference, forward(state, ids, mask, weights)[0])
    permutation = np.array([1, 0, 3, 2])
    np.testing.assert_allclose(reference, forward(state, ids[:, permutation], mask[:, permutation], weights)[0], atol=1e-15)
    clean_ids, clean_mask, _ = tokenize(['move left'], vocab)
    extra_ids, extra_mask, _ = tokenize(['please move left'], vocab, 'map')
    clean_sentence = forward(state[:1], clean_ids, clean_mask, weights)[1][2]
    extra_sentence = forward(state[:1], extra_ids, extra_mask, weights)[1][2]
    np.testing.assert_allclose(extra_sentence, clean_sentence*2/3, atol=1e-15)
    mapped, _, _ = tokenize(['move port', 'move starboard'], vocab, 'map')
    np.testing.assert_array_equal(mapped[0], mapped[1])
    rejected = []
    for name, text in [('unknown_word', 'move port'), ('empty', '  '), ('too_long', 'move left left left left'), ('non_string', 7)]:
        rejected.append(expect_rejection(name, lambda: tokenize([text], vocab)))
    bad_ids = ids.copy(); bad_ids[0, 0] = len(vocab)
    rejected.append(expect_rejection('invalid_id', lambda: forward(state, bad_ids, mask, weights)))
    rejected.append(expect_rejection('empty_mask', lambda: forward(state, np.zeros_like(ids), np.zeros_like(mask), weights)))
    rejected.append(expect_rejection('wrong_mask', lambda: forward(state, ids, ~mask, weights)))
    return dict(gradient_max_absolute_error=worst, pad_nan_invariance=True, order_invariance=True,
                unk_collision=True, zero_unk_changes_mean=True, rejected_cases=rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--unk-policy', choices=('reject', 'map'), default='reject')
    parser.add_argument('--output', type=Path, default=Path('tmp/s1_2'))
    args = parser.parse_args()
    require(args.epochs > 0 and not args.output.exists(), 'positive epochs and fresh output')
    args.output.mkdir(parents=True)
    # Reuse the S1.1 raw generator unchanged; its one-hot exports are context, not S1.2 model inputs.
    _, starts = dataset(args.output)
    batches = read_batches(args.output)
    vocab = fit_vocab(batches['train'][2])
    verification = checks(vocab)
    tokenizer = dict(schema_version='tiny_word_tokenizer/1.0', vocab=vocab, max_tokens=MAX_TOKENS,
                     normalization='str.lower().split(); punctuation is retained', pad_id=PAD, unk_id=UNK,
                     pooling='masked mean', embedding_dim=DIM, fit_split='train',
                     deployment_unknown_policy=args.unk_policy, unknown_embedding='zero, never trained; diagnostic only')
    (args.output/'tokenizer.json').write_text(json.dumps(tokenizer, indent=2)+'\n')
    loaded_vocab = json.loads((args.output/'tokenizer.json').read_text())['vocab']
    require(loaded_vocab == vocab, 'tokenizer reload')
    weights, history, initial = train(batches['train'], vocab, args.epochs, args.seed)
    np.savez_compressed(args.output/'weights.npz', **dict(zip(KEYS, weights)), initial_embedding=initial)
    with np.load(args.output/'weights.npz', allow_pickle=False) as saved:
        loaded_weights = [saved[key] for key in KEYS]
    metrics = {}
    for split, (state, action, texts, groups) in batches.items():
        ids, mask, _ = tokenize(texts, vocab)
        raw, (_, _, pooled) = forward(state/.3, ids, mask, weights)
        np.testing.assert_array_equal(raw, forward(state/.3, ids, mask, loaded_weights)[0])
        physical = raw*STEP
        np.savez_compressed(args.output/(split+'_tokens.npz'), state_m=state, action_m=action,
                            text=np.asarray(texts), group=np.asarray(groups), ids=ids, mask=mask, sentence=pooled)
        with np.load(args.output/(split+'_tokens.npz'), allow_pickle=False) as saved:
            for key, value in dict(state_m=state, action_m=action, text=np.asarray(texts), group=np.asarray(groups),
                                   ids=ids, mask=mask, sentence=pooled).items():
                np.testing.assert_array_equal(saved[key], value)
        metrics[split] = dict(samples=len(state), ids_shape=list(ids.shape), sentence_shape=list(pooled.shape),
                             raw_mse_m2=float(np.mean((physical-action)**2)),
                             executed_mse_m2=float(np.mean((np.clip(physical, -STEP, STEP)-action)**2)))
    diagnostics = []
    for text in ['move left', ' MOVE  LEFT ', 'left move', 'move left left', 'move port', 'move starboard', 'please move right', 'move right!']:
        ids, mask, unknown = tokenize([text], vocab, 'map')
        raw, (_, _, pooled) = forward(np.zeros((1, 2), np.float32), ids, mask, weights)
        diagnostics.append(dict(text=text, ids=ids[0].tolist(), mask=mask[0].tolist(), unknown_words=unknown[0],
                                sentence=pooled[0].tolist(), diagnostic_mapped_action_m=np.clip(raw[0]*STEP, -STEP, STEP).tolist(),
                                selected_policy_accepts=not unknown[0] or args.unk_policy == 'map'))
    known = rollout(starts, PROBES, loaded_vocab, loaded_weights, args.unk_policy)
    unknown = rollout(starts, UNKNOWN, loaded_vocab, loaded_weights, args.unk_policy)
    report = dict(schema_version='tiny_language_representation/1.0', environment=dict(python=sys.version,
                  executable=sys.executable, numpy=np.__version__, numpy_path=np.__file__), seed=args.seed,
                  epochs=args.epochs, unk_policy=args.unk_policy, tokenizer=tokenizer, checks=verification,
                  parameters=sum(w.size for w in weights), embedding_change_l2=float(np.linalg.norm(weights[0]-initial)),
                  history=history, metrics=metrics, diagnostics=diagnostics, known_rollout=known, unknown_rollout=unknown,
                  limitations='Only move left/right training text, static goals and ideal point; mean pooling loses order. UNK has no supervised examples. OOV probe goals are evaluator annotations, never policy input; no synonym understanding evidence.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(metrics=metrics, checks=verification, parameters=report['parameters'],
                          known_success=f"{known['successes']}/{known['denominator']}",
                          unknown_success=f"{unknown['successes']}/{unknown['denominator']}",
                          unknown_rejected=unknown['rejected'], unk_policy=args.unk_policy), indent=2))
    print(f'PASS embedding gradients / padding / collision / raw split / tokenizer and model reload; {args.output}')


if __name__ == '__main__':
    main()
