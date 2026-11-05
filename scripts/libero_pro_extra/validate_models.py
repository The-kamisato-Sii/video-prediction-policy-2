from pathlib import Path
import json,sys,gc
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_pro_extra_20261030'
sys.path.insert(0,str(ROOT/'src'))
from omegaconf import OmegaConf
from vpp2.libero_pro_extra.evaluate import validate_checkpoint
report={}
for name,m in json.loads((OUT/'queue.json').read_text())['models'].items():
    if not Path(m['config']['action_checkpoint']).is_file():
        report[name]=dict(status='WAIT_CHECKPOINT');continue
    try:
        cfg=OmegaConf.load(OUT/'jobs'/f'{name}__object'/'eval.yaml')
        validate_checkpoint(cfg)
        report[name]=dict(status='PASS',checkpoint=cfg.action_checkpoint,nfe=cfg.action_nfe,kind=cfg.extra_model_kind)
        print('MODEL_PASS',name,flush=True)
    except Exception as e:
        report[name]=dict(status='FAIL',error=repr(e));print('MODEL_FAIL',name,repr(e),flush=True)
    gc.collect()
    (OUT/'model-validation.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'model-validation.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(x['status'] in ('PASS','WAIT_CHECKPOINT') for x in report.values()),report
