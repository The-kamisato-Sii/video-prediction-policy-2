#!/usr/bin/env bash
set -euo pipefail
suite="${1:?suite required}"
case "$suite" in object|goal|spatial|10) ;; *) exit 2;; esac
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
# Only terminate the old worker recorded on this exact host, plus its episode child.
/usr/bin/python3 - "$suite" <<'PY'
import json,os,signal,socket,sys
from pathlib import Path
suite=sys.argv[1]
p=Path('runs/libero_motion_suites_20261022')/suite/'status.json'
if p.exists():
 s=json.loads(p.read_text());pid=s['pid'];proc=Path(f'/proc/{pid}')
 if s['host']==socket.gethostname() and proc.exists():
  command=(proc/'cmdline').read_bytes().replace(b'\0',b' ')
  if b'generate_all_motions.py' in command and f'--suite {suite}'.encode() in command:
   children=(proc/f'task/{pid}/children').read_text().split()
   for child in children:
    try:os.kill(int(child),signal.SIGTERM)
    except ProcessLookupError:pass
   os.kill(pid,signal.SIGTERM)
   s['state']='superseded_by_replay_v3';p.write_text(json.dumps(s,indent=2))
PY
if command -v tmux >/dev/null; then tmux_bin="$(command -v tmux)"; else tmux_bin="$(dirname "$repo")/tools/tmux-portable/tmux"; fi
"$tmux_bin" kill-session -t "motion-$suite" 2>/dev/null || true
exec bash scripts/libero/start_motion_suite.sh "$suite"
