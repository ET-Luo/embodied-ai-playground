"""Frozen tiny VLA proposals; explicit physical contract, no world-model ranking."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

import s1_5_tiny_transformer_vla as vla
from s0_1_data_contract import require
from s1_7_closed_loop_evaluation import load_checkpoint, sha256, SOURCE_FILES, offline_metrics, run_case, summarize

CONTRACT = dict(schema_version='vla_proposal/1.0', embodiment='ideal_world_xy_patch_grid_v1',
    representation='commanded_displacement', frame='world_xy', unit='m', control_dt_s=.1,
    horizon=3, action_dim=2, dtype='float32', component_limit_m=.06,
    temporal='slot h commands transition t+h to t+h+1, conditioned only on observation t',
    provenance='candidate0 frozen learned output; later candidates hand-designed constant XY biases then clip',
    execution='proposal only; no world-model score, selection, acceptance or execution implied')


def validate_proposals(actions, contract):
    require(contract == CONTRACT, 'incompatible proposal semantics/version')
    require(isinstance(actions, np.ndarray) and actions.ndim == 3 and actions.shape[1:] == (3, 2)
            and 1 <= len(actions) <= 5, 'candidate shape Kx3x2, K1..5')
    require(actions.dtype == np.float32 and np.isfinite(actions).all(), 'finite float32 physical actions')
    require(np.all(np.abs(actions) <= vla.STEP), 'component displacement limit')


def propose(rgb, state, text, weights, vocabulary, candidates):
    require(type(candidates) is int and 1 <= candidates <= 5, 'K1..5')
    base = vla.predict(rgb, state, text, weights, vocabulary)
    # Fixed nested order makes K a bank-size change, never a change to candidate0.
    biases = np.array([[0, 0], [.01, 0], [-.01, 0], [0, .01], [0, -.01]], np.float32)[:candidates]
    bank = np.clip(base[None] + biases[:, None], -vla.STEP, vla.STEP)
    validate_proposals(bank, CONTRACT)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('tmp/s4_1_source_s1_5'))
    parser.add_argument('--candidates', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--output', type=Path, default=Path('tmp/s4_1'))
    args = parser.parse_args()
    require(not args.output.exists(), 'fresh output directory required')
    weights, vocabulary, source = load_checkpoint(args.checkpoint)
    hashes = {name: sha256(args.checkpoint/name) for name in SOURCE_FILES}
    scenes = source['split']['validation']
    rgb0 = vla.render(np.array(scenes[0]['centers_uv'], np.int64))
    for _ in range(5):
        propose(rgb0, np.array([0, -.12], np.float32), vla.LANGUAGES[0], weights, vocabulary, args.candidates)
    banks = []; images = []; states = []; texts = []; rows = []; times = []
    for scene in scenes:
        rgb = vla.render(np.array(scene['centers_uv'], np.int64))
        for color, text in enumerate(vla.LANGUAGES):
            state = np.array([0, -.12], np.float32)
            start = perf_counter_ns()
            bank = propose(rgb, state, text, weights, vocabulary, args.candidates)
            times.append((perf_counter_ns()-start)/1e6)
            banks.append(bank); images.append(rgb); states.append(state); texts.append(text)
            rows.append(dict(scene_id=scene['scene_id'], text=text, candidate_count=len(bank),
                             unique_candidates=len(np.unique(bank.reshape(len(bank), -1), axis=0))))
    # Input refusal is preserved as a failed attempt, with no fabricated chunk.
    rejected = []
    for text in ('grab red block', 'grab blue block'):
        try:
            propose(rgb0, np.array([0, -.12], np.float32), text, weights, vocabulary, args.candidates)
        except ValueError as error:
            rejected.append(dict(text=text, failure_stage='policy_input_rejection', reason=str(error)))
    require(len(rejected) == 2, 'unsupported text must remain rejected')
    # Separate baseline deployment diagnosis: always original learned candidate0, E1.
    records = []
    for scene in scenes:
        for color, text in enumerate(vla.LANGUAGES):
            case = dict(case_id=f'{scene["scene_id"]}_{color}', condition='nominal',
                centers_uv=scene['centers_uv'], evaluator_color=color, text=text,
                start_m=[0, -.12], kick_x_m=0., kick_after_transition=2)
            records.append(run_case(case, 'learned', weights, vocabulary, 1))
    require(hashes == {name: sha256(args.checkpoint/name) for name in SOURCE_FILES}, 'source mutation')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'proposals.npz', action=np.stack(banks), image=np.stack(images),
        state=np.stack(states), text=np.array(texts), observation_tick=np.zeros(len(banks), np.int64))
    with np.load(args.output/'proposals.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['action'], np.stack(banks))
        for bank in saved['action']: validate_proposals(bank, CONTRACT)
    (args.output/'contract.json').write_text(json.dumps(CONTRACT, indent=2)+'\n')
    report = dict(schema_version='vla_proposal_report/1.0', environment=dict(python=sys.version,
        executable=sys.executable, numpy=np.__version__, numpy_path=np.__file__),
        checkpoint=str(args.checkpoint.resolve()), source_sha256=hashes, trained_during_proposal=False,
        candidates=args.candidates, proposal_attempts=18, accepted=16, rejected=rejected, cases=rows,
        proposal_latency_ms=dict(count=len(times), median=float(np.median(times)), p95=float(np.percentile(times, 95)),
            scope='patch/text/state inference + bank construction + contract; excludes render, IO; five warmups excluded'),
        offline=offline_metrics(args.checkpoint, weights, vocabulary),
        candidate0_closed_loop=summarize(records), closed_loop_records=records,
        limitations='Reused eight validation layouts, no independent test. Biases are hand-designed, not learned samples. Contract validity is not safety or success. No WM, ranking or fallback. Closed-loop diagnostic executes only candidate0 in ideal point plant.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(accepted=16, rejected=len(rejected), shape=list(np.stack(banks).shape),
        candidate0_successes=report['candidate0_closed_loop']['successes'],
        proposal_latency_ms=report['proposal_latency_ms']), indent=2))


if __name__ == '__main__':
    main()
