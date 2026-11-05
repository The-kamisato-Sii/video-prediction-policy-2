"""Summarize the sampled replay audit from persisted per-episode reports."""
import json,sys
from pathlib import Path
root=Path('runs/motion_validation_20261023')
version=sys.argv[1] if len(sys.argv)>1 else ''
selection=json.loads((root/'selection.json').read_text())
audit=json.loads((root/'noop_audit.json').read_text())
summary=dict(noop_audited=len(audit),noop_mismatches=sum(not x['matches_noop_rule'] for x in audit),suites={})
for suite,rows in selection.items():
 reports=[json.loads(p.read_text()) for p in (root/version/suite).glob('episode_*.json')]
 successful=[r for r in reports if r['passed']]
 replay_pass=[r for r in reports if 'rgb' in r]
 summary['suites'][suite]=dict(planned=len(rows),finished=len(reports),passed=len(successful),
  tasks=len({r['task'] for r in reports}),replay_passed=len(replay_pass),frames=sum(r['frames'] for r in replay_pass),
  max_rgb_mae=max((x['max'] for r in replay_pass for x in r['rgb'].values()),default=None),
  max_joint_error=max((r['max_proprio_error'][0] for r in replay_pass),default=None),
  failures=[dict(episode=r['episode'],error=r.get('error'),rgb=r.get('rgb')) for r in reports if not r['passed']])
(root/version/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
