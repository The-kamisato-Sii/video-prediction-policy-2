#!/usr/bin/env python3
"""Supervise isolated episode jobs, preserving an auditable resumable status."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempts', type=int, default=2)
    parser.add_argument('--suite', choices=['object', 'goal', 'spatial', '10'])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    datasets = sorted((repo / 'data/libero').glob('*_no_noops_lerobot'),
                      key=lambda p: (p.name != 'libero_object_no_noops_lerobot', p.name))
    if args.suite:
        datasets = [p for p in datasets if p.name == f'libero_{args.suite}_no_noops_lerobot']
    locks = []
    for dataset in datasets:
        lock = (repo / 'data/libero_motion_sources' / f'{dataset.name}.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    jobs = [(p, e['episode_index']) for p in datasets
            for e in map(json.loads, (p / 'meta/episodes.jsonl').read_text().splitlines())]
    status = dict(host=socket.gethostname(), pid=os.getpid(), started=now(),
                  total=len(jobs), completed=0, failed=[], current=None, state='running')

    def save():
        status['updated'] = now()
        temporary = run / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(run / 'status.json')

    save()
    for dataset, episode in jobs:
        name = f'{dataset.name}_episode_{episode:06d}'
        succeeded = False
        for attempt in range(1, args.attempts + 1):
            status['current'] = dict(dataset=dataset.name, episode=episode, attempt=attempt)
            save()
            log = run / f'{name}_attempt{attempt}.log'
            command = [sys.executable, '-u', '-m', 'vpp2.libero.motion_pipeline',
                       '--dataset', str(dataset), '--episodes', str(episode)]
            print(now(), 'START', name, 'attempt', attempt, flush=True)
            started = time.monotonic()
            with log.open('a') as out:
                try:
                    result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, timeout=5400)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            event = dict(dataset=dataset.name, episode=episode, attempt=attempt,
                         returncode=code, seconds=time.monotonic()-started, log=str(log), time=now())
            with (run / 'events.jsonl').open('a') as out:
                out.write(json.dumps(event) + '\n')
            print(now(), 'DONE' if code == 0 else 'FAILED', name, code, flush=True)
            if code == 0:
                succeeded = True
                break
            tail = log.read_text(errors='replace')[-3000:]
            if any(message in tail for message in ('RGB alignment failed:', 'Replay mismatch at frame',
                                                    'Source indices disagree')):
                print('Deterministic validation failure; preserve log without identical retry', flush=True)
                break
        if succeeded:
            status['completed'] += 1
        else:
            status['failed'].append(dict(dataset=dataset.name, episode=episode, log=str(log)))
        save()
    status['current'] = None
    status['state'] = 'complete' if not status['failed'] else 'finished_with_failures'
    save()


if __name__ == '__main__':
    main()
