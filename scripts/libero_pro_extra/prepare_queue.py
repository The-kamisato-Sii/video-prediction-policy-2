"""Inventory previously evaluated final checkpoints and create three jobs each."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/libero_pro_extra_20261030'

def saved(path):
    p=ROOT/path;data=json.loads(p.read_text());return data.get('config',data)

def main():
    if (OUT/'queue.json').exists():raise RuntimeError('Queue already exists; never overwrite active state')
    models={}
    def add(name,path,kind='baseline',group='',fraction='all',tokens=32):
        c=saved(path)
        for k in ('action_checkpoint','video_checkpoint','dataset_stats','wan_model_dir','training_config'):
            c[k]=str((ROOT/c[k]).resolve())
        models[name]=dict(config=c,kind=kind,group=group,action_fraction=fraction,tokens=tokens,source_config=str(ROOT/path),checkpoint_exists=Path(c['action_checkpoint']).exists())
    for d in (8,4,2):
        add(f'baseline-a1of{d}',f'runs/libero_fraction_eval_20261025/1of{d}/RUN_CONFIG.json',group='Action baseline',fraction=f'1/{d}')
        add(f'flow32-a1of{d}',f'runs/libero_flow_eval_20261030/flow32-a1of{d}/eval.yaml' if d==4 else f'runs/libero_flow_eval_20261027/1of{d}/eval.yaml','flow','Flow FM',f'1/{d}',32)
        add(f'flow128-a1of{d}',f'runs/libero_flow_eval_20261030/flow128-a1of{d}/eval.yaml' if d==4 else f'runs/libero_flow128_eval_20261027/1of{d}/eval.yaml','flow','Flow FM',f'1/{d}',128)
        add(f'recon32-mse100-a1of{d}',f'runs/libero_flow_eval_20261030/recon32-mse100-a1of{d}/eval.yaml','flow','Flow reconstruction MSE100',f'1/{d}',32)
        if d!=2:add(f'flow64-a1of{d}',f'runs/libero_flow_eval_20261030/flow64-a1of{d}/eval.yaml','flow','Flow FM',f'1/{d}',64)
    add('baseline-all','runs/libero_baseline_eval_20261022/baseline/RUN_CONFIG.json',group='Action baseline')
    for name,path,label in (
        ('direct-mse','runs/libero_eval_20261021/action-mse/RUN_CONFIG.json','Direct action MSE'),
        ('query-mse','runs/libero_ablations_eval_20261026/direct_action_learnable_mse/eval.yaml','Query action MSE'),
        ('meanflow-adaptive','runs/libero_meanflow_eval_20261022/meanflow/RUN_CONFIG.json','MeanFlow adaptive (original)'),
        ('meanflow-diagonal','runs/libero_ablations_eval_20261026/meanflow_diagonal_fix/eval.yaml','MeanFlow diagonal fix')):
        add(name,path,group=label)
    # Reserve, but never evaluate an intermediate checkpoint under a final-model label.
    m=json.loads(json.dumps(models['flow64-a1of4']));run=ROOT/'runs/libero_flow/libero-flow64-a1of2-job-34380189-5e35-4bcf-8f88-dc978bb982c3-round0'
    m['config']['action_checkpoint']=str(run/'checkpoints/weights/step_030000.pt');m['config']['training_config']=str(run/'resolved_config.yaml')
    m.update(action_fraction='1/2',checkpoint_exists=Path(m['config']['action_checkpoint']).exists(),source_config='reserved from final training run')
    m['config']['output_dir']=str(ROOT/'runs/libero_flow_eval_20261030/flow64-a1of2/evaluation')
    models['flow64-a1of2']=m
    jobs={}
    for axis in ('object','lan','env'):
        for name,m in models.items():
            key=name+'__'+axis;folder=OUT/'jobs'/key;folder.mkdir(parents=True,exist_ok=True)
            c=dict(m['config']);c.update(protocol='extra1200_v1',benchmark_root=str(OUT/'runtime'),runtime_config_dir=str(OUT/'runtime_config'),
                 output_dir=str(folder/'evaluation'),extra_axis=axis,extra_model_kind=m['kind'],
                 extra_assets_definition=str(OUT/'assets-definition.json'),extra_asset_check=str(OUT/'asset_checks'/f'{axis}.json'))
            (folder/'eval.yaml').write_text(json.dumps(c,indent=2)+'\n')
            jobs[key]=dict(model=name,axis=axis,kind=m['kind'],state='WAIT_ASSETS' if m['checkpoint_exists'] else 'WAIT_CHECKPOINT',config=str(folder/'eval.yaml'),expected_episodes=400,episodes=0,successes=0)
    state=dict(protocol='extra1200_v1',models=models,jobs=jobs,nodes={'h200-a':dict(notebook_id='7afff45e-8a19-40c0-85c1-e4599d9ce4b2'),'h200-b':dict(notebook_id='3b0549af-8ef8-473d-acd9-63c34d338f4c')},gpu_count=4)
    (OUT/'queue.json').write_text(json.dumps(state,indent=2)+'\n')
    print('MODELS',len(models),'FINAL',sum(m['checkpoint_exists'] for m in models.values()),'JOBS',len(jobs),'ROLLOUTS',len(jobs)*400)
if __name__=='__main__':main()
