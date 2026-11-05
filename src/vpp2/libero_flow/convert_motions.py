"""Create validated, atomic lossless training sidecars from completed motion GT."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time
import uuid
import numpy as np
from .motion_records import (CAMERAS, MotionRecords, cache_path, phase_order,
                             source_signature, write_records)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def metadata_path(source):
    source = Path(source)
    return source.parents[2] / 'meta/motions' / (source.stem + '.json')


def converted_metadata_path(source):
    source = Path(source)
    return source.parents[2] / 'meta/motions_fast' / (source.stem + '.json')


def is_current(source):
    dest = cache_path(source)
    meta = converted_metadata_path(source)
    try:
        report = json.loads(meta.read_text())
        return (report.get('source_signature') == source_signature(source)
            and report.get('source_metadata_sha256') == hashlib.sha256(metadata_path(source).read_bytes()).hexdigest()
            and report.get('cache_bytes') == dest.stat().st_size
            and report.get('verified_all_raw_frames') is True
            and report.get('schema') == 'vpp2.motion_records.v1')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def convert_one(source, *, scratch_dir=None, force=False):
    import h5py
    from vpp2.libero.motion_data import load_motion_clip
    source = Path(source).resolve()
    dest = cache_path(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Shared filesystem lock; two workers/watchers never publish the same file.
    with dest.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not force and is_current(source):
            return dict(source=str(source), destination=str(dest), status='already_current')
        began = time.perf_counter()
        metadata_bytes = metadata_path(source).read_bytes()
        source_meta = json.loads(metadata_bytes)
        signature = source_signature(source)
        if source_meta.get('replay_version') != 3 or not source_meta.get('complete_episode'):
            raise ValueError('Only completed replay-v3 motion may be converted')
        with h5py.File(source, 'r') as src:
            n = len(src['frame_index'])
            h, w = src['image/point_map'].shape[1:3]
            if source_meta['frames'] != n or (h, w) != (224, 224) or src.attrs.get('replay_version') != 3:
                raise ValueError('Source metadata/geometry contract mismatch')
            if not np.array_equal(src['frame_index'][:], np.arange(n)):
                raise ValueError('Unexpected source frame indices')
            fingerprint = dict(source_signature=signature,
                source_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                source_parquet_sha256=str(src.attrs['parquet_sha256']),
                source_replay_sha256=str(src.attrs['replay_sha256']),
                source_name=source.name, source_dataset=source.parents[2].name, replay_version=3,
                coordinate_frame=str(src.attrs['coordinate_frame']))
        scratch = Path(scratch_dir or tempfile.gettempdir())
        scratch.mkdir(parents=True, exist_ok=True)
        raw_bytes = n * 2 * h * w * (9 * 4 + 3)
        if shutil.disk_usage(scratch).free < raw_bytes + 1024**3:
            raise OSError('Insufficient temporary space for one lossless episode')
        if shutil.disk_usage(dest.parent).free < max(raw_bytes, 4 * 1024**3):
            raise OSError('Insufficient destination free space')
        temporary = dest.with_name('.' + dest.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with tempfile.TemporaryDirectory(prefix='vpp2_motion_pack_', dir=scratch) as staging:
                perm = phase_order(n)
                values = np.lib.format.open_memmap(Path(staging)/'values.npy', mode='w+',
                    dtype='<f4', shape=(n, 2, 3, h, w, 3))
                masks = np.lib.format.open_memmap(Path(staging)/'masks.npy', mode='w+',
                    dtype=np.uint8, shape=(n, 2, 3, h, w))
                # Each original dataset is scanned sequentially once. The large
                # staging arrays stay in local temporary storage, not shared GT.
                with h5py.File(source, 'r') as src:
                    for view, camera in enumerate(CAMERAS):
                        for field, key in enumerate(('point_map', 'flow_dt1/scene_flow', 'flow_dt4/scene_flow')):
                            raw = src[f'{camera}/{key}'][:]
                            if raw.dtype != np.dtype('float32'):
                                raise ValueError('Source GT must remain float32')
                            values[:, view, field] = raw[perm]
                            del raw
                        for field, key in enumerate(('valid_mask', 'flow_dt1/valid_mask', 'flow_dt4/valid_mask')):
                            raw = src[f'{camera}/{key}'][:]
                            masks[:, view, field] = raw[perm]
                            del raw
                values.flush(); masks.flush()
                meta = write_records(temporary, values, masks, fingerprint, shuffle=True)
                with MotionRecords(temporary) as packed:
                    for first in range(0, n, 8):
                        count = min(8, n-first)
                        v, m = packed.read_rows(first, count)
                        if not np.array_equal(v, values[first:first+count]) or not np.array_equal(m, masks[first:first+count]):
                            raise AssertionError('Raw motion values changed during conversion')
                    checked = []
                    if n >= 33:
                        for start in sorted({0, (n-33)//2, n-33}):
                            for dt in (1, 4):
                                ref = load_motion_clip(source, start=start, length=8, stride=4,
                                    flow_dt=dt, prediction_horizon=32)
                                fast = packed.clip(start, flow_dt=dt)
                                for camera in CAMERAS:
                                    for key in ref['cameras'][camera]:
                                        if not np.array_equal(ref['cameras'][camera][key], fast['cameras'][camera][key]):
                                            raise AssertionError((start, dt, camera, key))
                                if not np.array_equal(ref['center'], fast['center']) or ref['scale'] != fast['scale']:
                                    raise AssertionError('Clip normalization changed')
                                checked.append(dict(start=start, dt=dt, exact_equal=True))
                del values, masks
            if source_signature(source) != signature or metadata_path(source).read_bytes() != metadata_bytes:
                raise RuntimeError('Source changed while converting; not publishing')
            temporary.replace(dest)
            report = dict(meta, source=str(source), destination=str(dest),
                status='converted', cache_bytes=dest.stat().st_size,
                verified_all_raw_frames=True, normalized_clip_checks=checked,
                elapsed_s=time.perf_counter()-began, completed_at=datetime.now(timezone.utc).isoformat())
            atomic_json(converted_metadata_path(source), report)
            atomic_json(converted_metadata_path(source).parent / 'schema.json', dict(
                schema='vpp2.motion_records.v1',
                path='motions_fast/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mrec',
                container_magic='VPPMOT01', record_order='source indices grouped by index modulo4',
                values='float32 metric point_map, flow_dt1, flow_dt4; both cameras',
                validity='lossless bit-packed source and flow masks',
                normalization='same per-clip center and mean radius as canonical HDF5 loader, on read',
                codec='independent LZ4 frames with byte shuffle and content checksum',
                source='canonical motions/*.hdf5 and meta/motions completion metadata remain authoritative'))
            return report
        finally:
            temporary.unlink(missing_ok=True)


def completed_sources(root, suite=None):
    found = []
    for dataset in sorted(Path(root).glob('libero_*_no_noops_lerobot')):
        if suite and dataset.name != f'libero_{suite}_no_noops_lerobot': continue
        for path in sorted((dataset/'meta/motions').glob('episode_*.json')):
            if path.stem.endswith('_validation'): continue
            try: meta = json.loads(path.read_text())
            except (json.JSONDecodeError, FileNotFoundError): continue  # producer is publishing
            if meta.get('replay_version') != 3 or not meta.get('complete_episode'): continue
            episode = int(path.stem.split('_')[-1])
            source = dataset/f'motions/chunk-{episode//1000:03d}/{path.stem}.hdf5'
            if source.is_file(): found.append(source)
    return sorted(found, key=lambda p: (p.name, p.parents[2].name))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data/libero')
    p.add_argument('--suite', choices=('spatial','object','goal','10'))
    p.add_argument('--source', nargs='+', help='Convert only these complete episode files')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--poll-seconds', type=float, default=60)
    p.add_argument('--scratch-dir')
    p.add_argument('--force', action='store_true', help='Rebuild requested sidecars; incompatible with watch')
    p.add_argument('--run-dir', default='runs/libero_flow_motion_conversion')
    a = p.parse_args()
    if a.workers < 1 or a.poll_seconds < 1: p.error('Positive worker/poll settings required')
    if a.force and a.watch: p.error('--force is for bounded repairs, not a recurring watcher')
    run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=True)
    with (run/'watcher.lock').open('a') as watcher_lock:
        fcntl.flock(watcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(pid=os.getpid(), host=socket.gethostname(), workers=a.workers,
            started_at=datetime.now(timezone.utc).isoformat(), converted_this_run=0, failed=[], state='scanning')
        failures = {}
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            while True:
                sources = [Path(s) for s in a.source] if a.source else completed_sources(a.root, a.suite)
                ready = [s for s in sources if is_current(s)] if not a.force else []
                ready_set = set(ready)
                pending = [s for s in sources if s not in ready_set and time.monotonic()-failures.get(str(s),-1e9)>600]
                state.update(available_completed_gt=len(sources), cache_ready=len(ready), pending=len(pending),
                    state='converting' if pending else 'watching' if a.watch else 'complete',
                    updated_at=datetime.now(timezone.utc).isoformat())
                atomic_json(run/'status.json', state)
                # Bounded waves avoid queueing stale work while generation continues.
                for offset in range(0, len(pending), a.workers):
                    wave = pending[offset:offset+a.workers]
                    state['current'] = [str(s) for s in wave]
                    atomic_json(run/'status.json', state)
                    futures = {pool.submit(convert_one, str(s), scratch_dir=a.scratch_dir, force=a.force):s for s in wave}
                    for future in as_completed(futures):
                        source = futures[future]
                        try:
                            report = future.result()
                            state['converted_this_run'] += report['status'] == 'converted'
                            state['cache_ready'] += 1
                            state['failed'] = [e for e in state['failed'] if e['source'] != str(source)]
                        except Exception as e:
                            failures[str(source)] = time.monotonic()
                            report = dict(source=str(source), status='error', error=f'{type(e).__name__}: {e}')
                            state['failed'] = [e for e in state['failed'] if e['source'] != str(source)] + [report]
                        with (run/'events.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')
                        print(json.dumps({k:v for k,v in report.items() if k in ('source','destination','status','elapsed_s','cache_bytes','error')}), flush=True)
                        state['pending'] -= 1
                        state['updated_at'] = datetime.now(timezone.utc).isoformat()
                        atomic_json(run/'status.json', state)
                state.pop('current', None)
                state['state'] = 'watching' if a.watch else 'finished_with_errors' if state['failed'] else 'complete'
                atomic_json(run/'status.json', state)
                if not a.watch: break
                time.sleep(a.poll_seconds)
        if state['failed'] and not a.watch: raise SystemExit(1)


if __name__ == '__main__': main()
