"""Train-only tiny text goal selector; held-out phrasing and isolated oracle-control bridge."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from s0_1_data_contract import require
from s0_5_action_representation import expect_rejection
from s1_5_tiny_transformer_vla import render, world
from s1_3_visual_encoder import png

MAX_WORDS = 8
DIM = 4
TRAIN = [('pick red cube', 0), ('pick blue cube', 1), ('move red object', 0), ('move blue object', 1)]
EVALUATION = dict(
    seen=TRAIN,
    recombination=[('move red cube', 0), ('move blue cube', 1), ('pick red object', 0), ('pick blue object', 1)],
    oov_verb_noun=[('grab red block', 0), ('grab blue block', 1), ('move red item', 0), ('move blue item', 1)],
    oov_color=[('pick crimson cube', 0), ('pick cobalt cube', 1), ('move crimson object', 0), ('move cobalt object', 1)],
    order_stress=[('move red object instead of blue', 0), ('move blue object instead of red', 1)])


def words(text):
    require(isinstance(text, str), 'text must be string')
    result = text.lower().split()
    require(0 < len(result) <= MAX_WORDS, 'nonempty text with at most eight whitespace words; no truncation')
    return result


def fit_vocab(texts):
    return {'<PAD>': 0, '<UNK>': 1, **{word:i+2 for i,word in enumerate(sorted({w for text in texts for w in words(text)}))}}


def tokenize(texts, vocabulary, unk_policy):
    require(unk_policy in ('reject', 'map'), 'known unknown-word policy')
    ids = np.zeros((len(texts), MAX_WORDS), np.int64); mask = np.zeros_like(ids, dtype=bool); unknown = []
    for row, text in enumerate(texts):
        tokens = words(text); oov = [word for word in tokens if word not in vocabulary]
        require(unk_policy == 'map' or not oov, f'unknown words: {oov}')
        ids[row, :len(tokens)] = [vocabulary.get(word, 1) for word in tokens]
        mask[row, :len(tokens)] = True; unknown.append(oov)
    return ids, mask, unknown


def initialize(vocabulary, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    embedding = rng.normal(0, .2, (len(vocabulary), DIM)).astype(dtype); embedding[:2] = 0
    return dict(embedding=embedding, head=rng.normal(0, .2, (DIM, 2)).astype(dtype), bias=np.zeros(2, dtype))


def forward(ids, mask, weights):
    require(ids.ndim == 2 and ids.dtype == np.int64 and mask.shape == ids.shape and mask.dtype == bool,
            'int64 batch IDs and bool mask')
    require(np.all((ids >= 0) & (ids < len(weights['embedding']))) and np.array_equal(mask, ids != 0)
            and np.all(mask.sum(1) > 0), 'valid IDs, nonempty and consistent PAD mask')
    counts = mask.sum(1, keepdims=True).astype(weights['embedding'].dtype)
    sentence = np.where(mask[..., None], weights['embedding'][ids], 0).sum(1)/counts
    logits = sentence @ weights['head']+weights['bias']
    shifted = logits-logits.max(1, keepdims=True)
    log_probability = shifted-np.log(np.exp(shifted).sum(1, keepdims=True))
    return np.exp(log_probability), sentence, log_probability


def gradients(ids, mask, target, weights):
    probability, sentence, log_probability = forward(ids, mask, weights)
    loss = float(-log_probability[np.arange(len(target)), target].mean())
    delta = probability.copy(); delta[np.arange(len(target)), target] -= 1; delta /= len(target)
    dsentence = delta @ weights['head'].T
    dword = dsentence/mask.sum(1, keepdims=True).astype(dsentence.dtype)
    dembedding = np.zeros_like(weights['embedding'])
    row, column = np.where(mask)
    np.add.at(dembedding, ids[row, column], dword[row])
    return loss, dict(embedding=dembedding, head=sentence.T @ delta, bias=delta.sum(0))


def train(vocabulary, steps, seed):
    texts, target = zip(*TRAIN); target = np.array(target, dtype=np.int64)
    ids, mask, _ = tokenize(texts, vocabulary, 'reject')
    weights = initialize(vocabulary, seed)
    first = {key:np.zeros_like(value) for key,value in weights.items()}; second = {key:value.copy() for key,value in first.items()}
    history = []
    for step in range(1, steps+1):
        _, grad = gradients(ids, mask, target, weights)
        for key in weights:
            first[key] = .9*first[key]+.1*grad[key]; second[key] = .999*second[key]+.001*grad[key]**2
            weights[key] -= .03*(first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+1e-8)
        if step == 1 or step % 100 == 0 or step == steps:
            loss, _ = gradients(ids, mask, target, weights)
            history.append(dict(step=step, train_cross_entropy=loss))
    np.testing.assert_array_equal(weights['embedding'][:2], 0)
    return weights, history


def decide(text, vocabulary, weights, unk_policy):
    before = perf_counter_ns()
    ids, mask, unknown = tokenize([text], vocabulary, unk_policy)
    probability, sentence, _ = forward(ids, mask, weights)
    elapsed = (perf_counter_ns()-before)/1e6
    require(probability.dtype == np.float32, 'float32 inference')
    return dict(predicted_class=int(probability[0].argmax()), ids=ids[0].tolist(), mask=mask[0].tolist(),
                unknown_words=unknown[0], probabilities=probability[0].tolist(), sentence=sentence[0].tolist(),
                latency_ms=elapsed)


def bridge(scene, predicted_class, expected_class, rejection_stage='tokenization_rejected'):
    """Fixed controller with oracle object geometry; expected class is evaluator-only."""
    state = np.array([0, -.12], np.float32); trace = [state.tolist()]; actions = []
    if predicted_class is not None:
        selected_goal = world(scene['centers_uv'][predicted_class])
        for _ in range(8):
            action = np.clip(selected_goal-state, -.06, .06)
            state += action; trace.append(state.tolist()); actions.append(action.tolist())
    expected_goal = world(scene['centers_uv'][expected_class])
    error = float(np.max(np.abs(state-expected_goal)))
    success = predicted_class is not None and error <= .01
    stage = None if success else (rejection_stage if predicted_class is None else
                                  'language_target_selection' if predicted_class != expected_class else 'controller_final_goal_error')
    return dict(scene_id=scene['scene_id'], predicted_class=predicted_class, expected_class=expected_class,
                success=success, final_error_m=error, failure_stage=stage, truncated=predicted_class is not None,
                trace=trace, commands=actions, expected_goal_world_m=expected_goal.tolist())


def make_scenes():
    positions = [(u,v) for v in (4,12,20,28) for u in (4,12,20,28)]
    pairs = [(a,b) for a in range(16) for b in range(16) if a != b]
    choices = np.random.default_rng(41).permutation(len(pairs))[:6]
    return [dict(scene_id=f'language_bridge_{int(i)}', centers_uv=[positions[pairs[i][0]], positions[pairs[i][1]]]) for i in choices]


def evaluate(vocabulary, weights, unk_policy, scenes):
    results = {}; lookup = dict(TRAIN)
    for name, probes in EVALUATION.items():
        records = []; episodes = []; exact = []; latencies = []; oracle = []
        for text, expected in probes:
            try:
                decision = decide(text, vocabulary, weights, unk_policy)
                latencies.append(decision['latency_ms']); predicted = decision['predicted_class']; rejection = None
            except ValueError as error:
                predicted = None; decision = None; rejection = str(error)
            exact_prediction = lookup.get(' '.join(words(text)))
            record = dict(text=text, expected_class=expected, decision=decision, rejection=rejection,
                          correct=predicted == expected, exact_lookup_correct=exact_prediction == expected)
            records.append(record)
            for scene in scenes:
                ep = bridge(scene, predicted, expected); ep['text'] = text; episodes.append(ep)
                baseline = bridge(scene, exact_prediction, expected, 'exact_instruction_lookup_miss'); baseline['text'] = text; exact.append(baseline)
                reference = bridge(scene, expected, expected); reference['text'] = text; oracle.append(reference)
        results[name] = dict(phrases=len(probes), target_correct=sum(row['correct'] for row in records),
            rejected=sum(row['rejection'] is not None for row in records),
            exact_lookup_correct=sum(row['exact_lookup_correct'] for row in records), records=records,
            bridge=dict(successes=sum(ep['success'] for ep in episodes), denominator=len(episodes), episodes=episodes),
            exact_lookup_bridge=dict(successes=sum(ep['success'] for ep in exact), denominator=len(exact), episodes=exact),
            oracle_bridge=dict(successes=sum(ep['success'] for ep in oracle), denominator=len(oracle), episodes=oracle),
            latency_ms=None if not latencies else dict(median=float(np.median(latencies)), p95=float(np.percentile(latencies,95)),
                samples=len(latencies), scope='successful text tokenization+mean embedding+classifier; excludes bridge, no deployment claim'))
    return results


def checks(vocabulary):
    weights = initialize(vocabulary, 3, np.float64)
    ids, mask, _ = tokenize(['pick red red cube', 'move blue object'], vocabulary, 'reject')
    target = np.array([0,1]); _, analytic = gradients(ids, mask, target, weights)
    worst = 0.
    for name, parameter in weights.items():
        for index in np.ndindex(parameter.shape):
            old = parameter[index]; eps = 1e-5
            parameter[index] = old+eps; plus = gradients(ids, mask, target, weights)[0]
            parameter[index] = old-eps; minus = gradients(ids, mask, target, weights)[0]
            parameter[index] = old
            worst = max(worst, abs((plus-minus)/(2*eps)-analytic[name][index]))
    require(worst < 1e-8, 'full language classifier gradient')
    clean = forward(ids, mask, weights)[0]; weights['embedding'][0] = np.nan
    np.testing.assert_array_equal(clean, forward(ids, mask, weights)[0])
    collision_ids, collision_mask, _ = tokenize([text for text,_ in EVALUATION['oov_color'][:2]], vocabulary, 'map')
    np.testing.assert_array_equal(collision_ids[0],collision_ids[1])
    probability = forward(collision_ids, collision_mask, weights)[0]
    np.testing.assert_array_equal(probability[0],probability[1])
    stress_ids, stress_mask, _ = tokenize([text for text,_ in EVALUATION['order_stress']],vocabulary,'map')
    probability = forward(stress_ids, stress_mask, weights)[0]
    np.testing.assert_allclose(probability[0],probability[1],atol=1e-14)
    train_set = {' '.join(words(t)) for t,_ in TRAIN}
    for group, probes in EVALUATION.items():
        if group != 'seen': require(not train_set & {' '.join(words(t)) for t,_ in probes}, 'held-out phrase leakage')
    require(not {'grab','block','item','crimson','cobalt','instead','of'} & vocabulary.keys(), 'probe vocabulary leakage')
    rejected = [expect_rejection('unknown_reject', lambda: tokenize(['grab red block'],vocabulary,'reject')),
                expect_rejection('empty', lambda: tokenize([''],vocabulary,'map')),
                expect_rejection('overflow', lambda: tokenize(['red '*9],vocabulary,'map')),
                expect_rejection('nonstring', lambda: tokenize([None],vocabulary,'map')),
                expect_rejection('bad_policy', lambda: tokenize(['pick red cube'],vocabulary,'silent'))]
    return dict(gradient_max_absolute_error=worst, pad_nan_invariance=True, oov_color_collision=True,
                order_stress_mean_collision=True, normalized_train_test_overlap=0, heldout_vocabulary_leakage=False,
                rejected_cases=rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=800)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--unk-policy',choices=('reject','map'),default='reject')
    parser.add_argument('--output',type=Path,default=Path('tmp/s1_6'))
    args = parser.parse_args()
    require(args.steps > 0 and not args.output.exists(),'positive training steps and fresh output')
    vocabulary = fit_vocab([text for text,_ in TRAIN]); verification = checks(vocabulary)
    weights, history = train(vocabulary,args.steps,args.seed)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'weights.npz',**weights)
    with np.load(args.output/'weights.npz',allow_pickle=False) as saved: loaded = {key:saved[key] for key in saved.files}
    all_texts = [text for probes in EVALUATION.values() for text,_ in probes]
    ids, mask, unknown = tokenize(all_texts,vocabulary,'map')
    original = forward(ids,mask,weights); reloaded = forward(ids,mask,loaded)
    for a,b in zip(original,reloaded): np.testing.assert_array_equal(a,b)
    arrays = dict(text=np.array(all_texts),ids=ids,mask=mask,sentence=original[1],mapped_probability=original[0])
    np.savez_compressed(args.output/'text_diagnostics.npz', **arrays)
    with np.load(args.output/'text_diagnostics.npz',allow_pickle=False) as saved:
        for key,value in arrays.items(): np.testing.assert_array_equal(saved[key],value)
    tokenizer = dict(vocabulary=vocabulary,fit_split='train four phrases only',normalization='lower().split(); punctuation retained',
                     max_tokens=MAX_WORDS,pad_id=0,unk_id=1,unknown_policy=args.unk_policy,
                     unknown_embedding='zero; never trained; map is diagnostic, not synonym learning')
    (args.output/'tokenizer.json').write_text(json.dumps(tokenizer,indent=2)+'\n')
    manifest = dict(schema_version='tiny_language_probe/1.0',train=TRAIN,evaluation=EVALUATION,
        task_semantics='All verbs mean go to annotated color goal in this synthetic probe; no physical pick/grab manipulation.',
        classes=['red','blue'],oov_color_annotations='crimson/cobalt evaluator labels only, never supplied to tokenizer/classifier',
        partition='exact normalized phrases disjoint outside seen; recombination uses only train words; other probes contain OOV',
        frozen_protocol='fixed phrase lists/seed/800 default updates; no fitting vocabulary or tuning checkpoint on probes',
        bridge='oracle red/blue world XY + fixed ideal delta controller, isolates language selector; not S1.5 VLA repair',
        physical='world XY m, .02m/pixel, measured start[0,-.12], dt.1s, clip each delta +/-.06m, budget8, success maxerror<=.01m',
        scene_scope='six deterministic layouts for evaluation only; classifier is text-only, no learned scene generalization claim')
    (args.output/'probe_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    loaded_vocabulary = json.loads((args.output/'tokenizer.json').read_text())['vocabulary']
    require(loaded_vocabulary == vocabulary, 'tokenizer round-trip')
    scenes = make_scenes(); results = evaluate(loaded_vocabulary,loaded,args.unk_policy,scenes)
    (args.output/'scenes.json').write_text(json.dumps(scenes,indent=2)+'\n')
    png(args.output/'scenes.png',np.concatenate([np.repeat(np.repeat(render(np.array(s['centers_uv'])),4,0),4,1) for s in scenes],axis=1))
    report = dict(schema_version='language_generalization/1.0',environment=dict(python=sys.version,executable=sys.executable,
        numpy=np.__version__,numpy_path=np.__file__),steps=args.steps,seed=args.seed,unk_policy=args.unk_policy,
        parameters=sum(w.size for w in weights.values()),checks=verification,history=history,results=results,
        limitations='Four training phrases, mean embedding color selector, one seed; successful OOV verbs/nouns can follow known color shortcut. Unknown color and word-order collision expose information loss. Oracle perception/controller bridge is not learned VLA or real robot success. S1.5 remains failed.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(parameters=report['parameters'],checks=verification,unk_policy=args.unk_policy,
        results={name:dict(target=f"{value['target_correct']}/{value['phrases']}",rejected=value['rejected'],
            exact=f"{value['exact_lookup_correct']}/{value['phrases']}",
            oracle_control_bridge=f"{value['bridge']['successes']}/{value['bridge']['denominator']}") for name,value in results.items()}),indent=2))
    print(f'PASS full gradient / train-only vocab / phrase splits / collisions / weight reload; {args.output}')


if __name__ == '__main__':
    main()
