"""Deterministic hypothetical deployment budget; no model inference or speed benchmark."""
import argparse
import json
import math
from pathlib import Path


def calculate(horizon=4, execute=1, action_dim=7, fast_tokens=8, steps=10, dt=.1):
    for name, value in [('horizon', horizon), ('execute', execute), ('action_dim', action_dim),
                        ('fast_tokens', fast_tokens), ('steps', steps)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if execute > horizon or not math.isfinite(dt) or dt <= 0:
        raise ValueError('require 1 <= execute <= horizon and finite dt > 0')
    # Teaching assumptions only. These costs are NOT measured on any VLA or hardware.
    costs = dict(prefix_ms=40., ar_token_ms=5., expert_step_ms=4.,
                 small_head_step_ms=.5, io_decode_ms=5.)
    paths = [
        ('scalar_chunk_ar_hypothesis', horizon * action_dim, costs['ar_token_ms']),
        ('fast_chunk_ar_hypothesis', fast_tokens, costs['ar_token_ms']),
        ('flow_expert_hypothesis', steps, costs['expert_step_ms']),
        ('small_diffusion_head_hypothesis', 20, costs['small_head_step_ms']),
    ]
    budget_ms = execute * dt * 1000
    rows = []
    for name, calls, cost in paths:
        total = costs['prefix_ms'] + calls * cost + costs['io_decode_ms']
        rows.append(dict(path=name, sequential_calls=calls, assumed_latency_ms=total,
                         budget_ms=budget_ms, margin_ms=budget_ms-total,
                         steady_state_budget_pass=total <= budget_ms))
    return dict(scope='Hypothetical arithmetic only; not OpenVLA/Octo/pi0 timings or robot success.',
                horizon=horizon, execute=execute, action_dim=action_dim,
                fast_tokens=fast_tokens, flow_steps=steps, robot_dt_s=dt,
                planned_horizon_s=horizon*dt, execution_prefix_s=execute*dt,
                assumed_costs=costs, rows=rows,
                qualifications=['Scalar chunk AR is hypothetical: original OpenVLA uses H1.',
                                'FAST token count is an input assumption, not H*A divided by a guaranteed ratio.',
                                'Steps and per-step costs differ between models; shared prefix cost is a teaching control.',
                                'E*dt is a steady-state overlap budget; startup and observation age still matter.',
                                'Passing a budget is neither a measured deadline guarantee nor closed-loop success.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', type=int, default=1)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('tmp/s2_7'))
    args = parser.parse_args()
    result = calculate(execute=args.execute, steps=args.steps)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    header = '| hypothetical path | serial calls | assumed ms | budget ms | margin ms | pass |\n|---|---:|---:|---:|---:|---|\n'
    lines = [f"| {r['path']} | {r['sequential_calls']} | {r['assumed_latency_ms']:g} | {r['budget_ms']:g} | {r['margin_ms']:g} | {r['steady_state_budget_pass']} |" for r in result['rows']]
    (args.output/'budget.md').write_text('Hypothetical arithmetic; no model measured.\n\n'+header+'\n'.join(lines)+'\n')
    assert json.loads((args.output/'report.json').read_text()) == result
    print(header+'\n'.join(lines))


if __name__ == '__main__':
    main()
