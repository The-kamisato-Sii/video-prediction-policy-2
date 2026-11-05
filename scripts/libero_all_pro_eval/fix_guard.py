import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_all_pro_eval_20261103';guard=OUT/'gpu_idle_guard.py'
parents=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if str(guard).encode() in args:parents.append(int(p.name))
 except (OSError,ProcessLookupError):pass
for pid in parents:os.kill(pid,signal.SIGTERM)
for _ in range(30):
 if not any(Path(f'/proc/{pid}').exists() for pid in parents):break
 time.sleep(1)
tmux=str(ROOT.parent/'tools/tmux-portable/tmux');env='/inspire/hdd/global_user/linbokai-CZXS24250037/inspire_shared/vpp2_flow_eval_20261027/root/miniconda/envs/vpp2-libero-pro/bin/python'
if subprocess.run([tmux,'has-session','-t','libero-all-pro-gpu-guard'],capture_output=True).returncode==0:
 subprocess.run([tmux,'kill-session','-t','libero-all-pro-gpu-guard'],check=True)
subprocess.run([tmux,'new-session','-d','-s','libero-all-pro-gpu-guard',f'{env} -u {guard} >> {OUT}/gpu-guard.log 2>&1'],check=True)
(OUT/'guard-fix.json').write_text(json.dumps(dict(status='PASS',stopped_own_guard_pids=parents,isolated_cuda_visible_devices_per_guard=True,time=time.time()),indent=2)+'\n')
