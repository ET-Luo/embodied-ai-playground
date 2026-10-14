"""Inspect one pinned real Bridge mirror episode; never export a training episode."""
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import sys
import urllib.request

import av
import numpy as np
import PIL
from PIL import Image, ImageDraw
import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST = Path(__file__).resolve().parents[1] / 'data/adapters/bridge_sample_manifest.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_bytes(payload, entry):
    require(len(payload) == entry['bytes'], f"size mismatch: {entry['path']}")
    require(hashlib.sha256(payload).hexdigest() == entry['sha256'],
            f"SHA256 mismatch: {entry['path']}")


def prepare_source(root, manifest, download):
    """Bounded, pinned downloads; existing raw files are verified, never overwritten."""
    total = sum(entry['bytes'] for entry in manifest['files'])
    require(total < 3_000_000, 'sample download budget exceeded')
    base = f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/{manifest['revision']}/"
    print(f'Source budget: {total} bytes; download enabled={download}')
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.exists():
            require(download, f'missing {path}; use --download once, or provide verified --source')
            with urllib.request.urlopen(base + entry['path'], timeout=30) as response:
                payload = response.read(entry['bytes'] + 1)
            verify_bytes(payload, entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as target:
                target.write(payload)
        require(path.stat().st_size == entry['bytes'], f'raw file size mismatch: {path}')
        verify_bytes(path.read_bytes(), entry)
    return total


def audit_rows(rows, info, tasks, episode):
    """Structural evidence only. No action semantics or boundary inferred here."""
    expected = {'observation.state', 'action', 'camera_present', 'timestamp',
                'frame_index', 'episode_index', 'index', 'task_index'}
    require(len(rows) >= 2, 'too few rows')
    require(all(set(row) == expected for row in rows), 'unexpected/missing row fields')
    features = info['features']
    arrays = {}
    for key in ('observation.state', 'action'):
        require(features[key]['shape'] == [7] and features[key]['dtype'] == 'float32', 'schema changed')
        arrays[key] = np.asarray([row[key] for row in rows], dtype=np.float32)
        require(arrays[key].shape == (len(rows), 7), f'{key} shape')
        require(np.isfinite(arrays[key]).all(), f'{key} nonfinite')
    arrays['timestamp'] = np.asarray([row['timestamp'] for row in rows], dtype=np.float32)
    require(np.isfinite(arrays['timestamp']).all(), 'nonfinite timestamp')
    expected_time = np.arange(len(rows), dtype=np.float64) / info['fps']
    require(np.allclose(arrays['timestamp'], expected_time, rtol=0, atol=1e-6), 'clock/frame cadence')
    require(all(row['frame_index'] == i for i, row in enumerate(rows)), 'frame order')
    require(all(row['episode_index'] == episode for row in rows), 'episode mixing')
    require(all(rows[i+1]['index'] == rows[i]['index'] + 1 for i in range(len(rows)-1)), 'global index order')
    camera = np.asarray([row['camera_present'] for row in rows])
    require(camera.shape == (len(rows), 4) and camera.dtype == np.bool_, 'camera mask')
    require(camera[:, 0].all(), 'selected camera absent')
    task_ids = sorted({row['task_index'] for row in rows})
    require(len(task_ids) == 1 and task_ids[0] in tasks, 'missing/changing task text')
    require(isinstance(tasks[task_ids[0]], str) and bool(tasks[task_ids[0]].strip()), 'empty task text')
    arrays['camera_present'] = camera
    return arrays, tasks[task_ids[0]], task_ids[0]


def audit_arrow(table):
    for key in ('observation.state', 'action', 'camera_present'):
        typ = table.schema.field(key).type
        width = 4 if key == 'camera_present' else 7
        value_type = pa.bool_() if key == 'camera_present' else pa.float32()
        require(pa.types.is_fixed_size_list(typ) and typ.list_size == width
                and typ.value_type == value_type, f'actual Arrow dtype mismatch: {key}')
    require(table.schema.field('timestamp').type == pa.float32(), 'timestamp dtype')
    for key in ('index', 'frame_index', 'episode_index', 'task_index'):
        require(table.schema.field(key).type == pa.int64(), f'{key} dtype')
    require(all(table[key].null_count == 0 for key in table.column_names), 'null column')


def decode_video(path, count, times):
    images, pts = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        rate = float(stream.average_rate)
        for frame in container.decode(stream):
            require(len(images) < count, 'video has extra frames')
            require(frame.pts is not None, 'missing video PTS')
            rgb = frame.to_ndarray(format='rgb24')
            require(rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8, 'decoded RGB shape/dtype')
            images.append(rgb)
            pts.append(float(frame.pts * frame.time_base))
    require(len(images) == count, 'video/table row mismatch')
    require(np.allclose(pts, times, rtol=0, atol=1e-6), 'video/table presentation timestamp mismatch')
    return images, dict(codec=codec, fps=rate, frames=len(images), decoded_shape=[count, 256, 256, 3],
                        dtype='uint8', color_space='RGB', pts_seconds=pts,
                        acquisition_sync='unknown; PTS match only')


def alignment_probe(state, action):
    """First six unnamed-unit components only; diagnostic, not a relabeling rule."""
    delta = np.diff(state[:, :6].astype(np.float64), axis=0)
    probes = []
    for offset in (-1, 0, 1):
        i = np.arange(len(delta))
        j = i + offset
        keep = (j >= 0) & (j < len(action))
        residual = action[j[keep], :6] - delta[i[keep]]
        probes.append(dict(action_row_offset=offset, compared_pairs=int(keep.sum()),
                           per_component_mae=np.abs(residual).mean(axis=0).tolist()))
    return dict(equation='action[i+offset, :6] versus state[i+1, :6]-state[i, :6]',
                caveat='Unknown units/frame/rotation/controller; MAE is per-component raw numeric diagnostic, not physical error or proof of alignment.',
                probes=probes)


def rejection_checks(rows, info, tasks, manifest, source):
    names = ['nonfinite_action', 'clock', 'frame_order', 'episode_mix', 'vector_shape', 'camera_absent', 'missing_language']
    for name in names:
        bad = copy.deepcopy(rows)
        bad_tasks = dict(tasks)
        if name == 'nonfinite_action': bad[0]['action'][0] = float('nan')
        if name == 'clock': bad[1]['timestamp'] = 9.
        if name == 'frame_order': bad[1]['frame_index'] = 99
        if name == 'episode_mix': bad[1]['episode_index'] = 1
        if name == 'vector_shape': bad[0]['observation.state'].pop()
        if name == 'camera_absent': bad[0]['camera_present'][0] = False
        if name == 'missing_language': bad_tasks.clear()
        try:
            audit_rows(bad, info, bad_tasks, manifest['episode_index'])
        except ValueError:
            pass
        else:
            raise RuntimeError(f'corrupt sample accepted: {name}')
    entry = manifest['files'][2]
    payload = bytearray((source / entry['path']).read_bytes())
    payload[0] ^= 1
    try:
        verify_bytes(payload, entry)
    except ValueError:
        pass
    else:
        raise RuntimeError('corrupt raw hash accepted')
    return names + ['raw_sha256']


def write_preview(output, images, selected, rows, report):
    sheet = Image.new('RGB', (256 * len(selected), 288), 'white')
    draw = ImageDraw.Draw(sheet)
    for slot, i in enumerate(selected):
        sheet.paste(Image.fromarray(images[i]), (slot * 256, 32))
        draw.text((slot * 256 + 8, 8), f'row {i} | {rows[i]["timestamp"]:.2f} s', fill='black')
    sheet.save(output / 'contact_sheet.png')
    table_rows = []
    for i, row in enumerate(rows):
        state = ', '.join(f'{x:.6g}' for x in row['observation.state'])
        action = ', '.join(f'{x:.6g}' for x in row['action'])
        table_rows.append(f'<tr><td>{i}</td><td>{row["timestamp"]:.6g}</td><td>{state}</td><td>{action}</td></tr>')
    text = '<!doctype html><meta charset="utf-8"><title>S0.4 real sample inspection</title>'
    text += '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}td,th{border:1px solid #ddd;padding:6px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'
    text += '<h1>S0.4 — Bridge mirror episode 0</h1><p>' + html.escape(report['language']) + '</p>'
    text += '<p>Preview stride changes displayed rows only. Raw state/action units, frame, controller and causal alignment remain unverified. No training export.</p>'
    text += '<img src="contact_sheet.png" alt="Selected real robot camera frames">'
    text += '<h2>All raw numerical rows</h2><p>Channel names: x, y, z, roll, pitch, yaw, gripper. Names do not establish units or absolute/delta semantics.</p>'
    text += '<table><tr><th>row</th><th>stored time (s)</th><th>state</th><th>action</th></tr>' + ''.join(table_rows) + '</table>'
    text += '<h2>Inspection report</h2><pre>' + html.escape(json.dumps(report, indent=2, ensure_ascii=False)) + '</pre>'
    (output / 'inspection.html').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('tmp/s0_4_source'))
    parser.add_argument('--download', action='store_true', help='download only the four pinned files, <3 MB')
    parser.add_argument('--output', type=Path, default=Path('tmp/s0_4'))
    parser.add_argument('--preview-stride', type=int, default=7)
    args = parser.parse_args()
    require(args.preview_stride > 0, 'preview stride must be positive')
    require(not args.output.exists(), 'output already exists; choose a new directory')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    total = prepare_source(args.source, manifest, args.download)
    info = json.loads((args.source / 'meta/info.json').read_text(encoding='utf-8'))
    tasks = {}
    for line in (args.source / 'meta/tasks.jsonl').read_text(encoding='utf-8').splitlines():
        task = json.loads(line)
        require(task['task_index'] not in tasks, 'duplicate task ID')
        tasks[task['task_index']] = task['task']
    table = pq.read_table(args.source / manifest['files'][2]['path'])
    audit_arrow(table)
    rows = table.to_pylist()
    arrays, language, task_id = audit_rows(rows, info, tasks, manifest['episode_index'])
    images, video = decode_video(args.source / manifest['files'][3]['path'], len(rows), arrays['timestamp'])
    selected = sorted(set(range(0, len(rows), args.preview_stride)) | {len(rows)-1})
    blockers = [
        'N observations and N actions; final outgoing-action validity and post-action observation unverified',
        'No terminal/truncated flags; cannot infer boundary kind from file ending',
        'No independent success annotation in fetched files',
        'Units, coordinate frame, action anchor/rotation convention/controller not specified by mirror metadata',
        'Conversion code revision, original episode/reset group and latency handling unknown',
        'Video PTS matches record clock, but sensor acquisition causality/calibration unknown',
    ]
    stats = {}
    for key in ('observation.state', 'action'):
        x = arrays[key]
        stats[key] = dict(shape=list(x.shape), dtype=str(x.dtype), fields=info['features'][key]['names'],
                          min=x.min(axis=0).tolist(), max=x.max(axis=0).tolist(),
                          first=x[0].tolist(), last=x[-1].tolist(), units_frame='unknown')
    report = dict(
        schema_version='robot_sample_inspection_report/1.0', source=manifest, downloaded_file_bytes=total,
        environment=dict(python=sys.version.split()[0], executable=sys.executable,
                         versions={m.__name__: m.__version__ for m in (np, pa, av, PIL)},
                         module_paths={m.__name__: m.__file__ for m in (np, pa, av, PIL)}),
        rows=len(rows), episode_index=0, task_index=task_id, language=language, arrays=stats,
        arrow_schema=str(table.schema), timestamp_dtype=str(arrays['timestamp'].dtype),
        timestamp_range_seconds=[float(arrays['timestamp'][0]), float(arrays['timestamp'][-1])],
        declared_fps=info['fps'], max_clock_error_seconds=float(np.max(np.abs(arrays['timestamp']-np.arange(len(rows))/info['fps']))),
        camera_present_patterns=np.unique(arrays['camera_present'], axis=0).tolist(), video=video,
        preview_stride=args.preview_stride, preview_rows=selected,
        alignment_diagnostic=alignment_probe(arrays['observation.state'], arrays['action']),
        local_contract=dict(version='embodied_episode/1.0', export_allowed=False, blockers=blockers),
        rejected_cases=rejection_checks(rows, info, tasks, manifest, args.source),
        limitations='One community-converted episode and one camera; no original RLDS/TFDS decode, producer audit, training, robot rollout, success evaluation or generalization claim.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    write_preview(args.output, images, selected, rows, report)
    print(json.dumps({key: report[key] for key in ('rows', 'language', 'timestamp_range_seconds', 'preview_rows', 'local_contract', 'rejected_cases')}, indent=2))
    print(f'PASS raw hashes / actual Arrow schema / numerical audit / video RGB+PTS / rejection checks; inspect {args.output}/inspection.html')


if __name__ == '__main__':
    main()
