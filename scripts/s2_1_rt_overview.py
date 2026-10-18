"""S2.1: executable codec comparison; no RT model or learned policy."""
import argparse
import json
from pathlib import Path

import numpy as np

from s0_5_action_representation import episode, views
from s0_6_action_tokenization import fit_tokenizer, encode, decode


def parse_action(text, bins):
    """Local decimal-field protocol, not a PaLI-X/PaLM-E tokenizer."""
    fields = text.split()
    if len(fields) != 2 or any(not s.isascii() or not s.isdecimal() for s in fields):
        raise ValueError('expected two unsigned decimal fields')
    ids = np.array([int(s) for s in fields], dtype=np.int64)
    if not ((0 <= ids) & (ids < bins)).all():
        raise ValueError('action bin outside vocabulary')
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bins', type=int, choices=(8, 16, 256), default=256)
    parser.add_argument('--output', type=Path, default=Path('tmp/s2_1'))
    args = parser.parse_args()
    # Reuse the established joint-reference data and train-only ranges.
    train = [episode(i, np.array([i*.02, -i*.01], dtype=np.float32), 'train', .5) for i in range(3)]
    spec = fit_tokenizer(train, args.bins)
    signature = views(*train[0])[1]['joint_absolute']
    raw = train[0][0]['action']
    channel_ids, saturation, _ = encode(raw, spec, signature)
    bins = channel_ids - np.arange(2) * args.bins
    strings = [' '.join(map(str, row)) for row in bins]
    restored = np.stack([parse_action(s, args.bins) for s in strings])
    np.testing.assert_array_equal(restored, bins)
    decoded = decode(restored + np.arange(2) * args.bins, spec, signature)
    width = (np.array(spec['high']) - spec['low']) / args.bins
    error = np.abs(decoded.astype(np.float64) - raw)
    assert not saturation.any() and (error <= width / 2 + 1e-7).all()
    rejected = 0
    for bad in ('hello robot', '1', '1 2 3', '-1 0', f'{args.bins} 0', 'nan 0'):
        try:
            parse_action(bad, args.bins)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(f'accepted {bad!r}')
    # Synthetic logits: a text-only ID wins without vocabulary restriction.
    logits = np.zeros(args.bins + 1)
    logits[args.bins] = 10
    logits[3] = 5
    unrestricted, restricted = int(logits.argmax()), int(logits[:args.bins].argmax())
    assert unrestricted == args.bins and restricted == 3 and rejected == 6
    report = dict(
        schema_version='rt_overview_concept/1.0', learned_model=False,
        source='S0.5 first train episode; joint absolute reference; not RT robot data',
        action_signature=signature, tokenizer=spec,
        raw_shape=list(raw.shape), raw_dtype=str(raw.dtype), bins=args.bins,
        shared_bin_ids=bins.tolist(), channel_ids=channel_ids.tolist(), strings=strings,
        max_absolute_error_per_joint_rad=error.max(axis=0).tolist(),
        half_bin_bound_rad=(width/2).tolist(), rejection_checks=rejected,
        unrestricted_id=unrestricted, restricted_id=restricted,
        output_fields_per_action=2, future_horizon=1,
        evidence_limit='codec round-trip and lexical constraint only; no semantic transfer, model latency or robot success')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'codec.npz', raw=raw, bin_ids=bins,
                        channel_ids=channel_ids, decoded=decoded)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    with np.load(args.output/'codec.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['raw'], raw)
        np.testing.assert_array_equal(saved['decoded'], decoded)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
