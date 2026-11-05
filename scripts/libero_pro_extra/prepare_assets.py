"""Freeze official Object/Language assets and the official Environment generator output."""
from pathlib import Path
import json, shutil, importlib.util, sys, re
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/libero_pro_extra_20261030'
SOURCE = ROOT/'third_party/libero_pro_runtime'
RUNTIME = OUT/'runtime'
SUITES = ('libero_spatial','libero_object','libero_goal','libero_10')

def main():
    RUNTIME.mkdir(exist_ok=True)
    for p in SOURCE.iterdir():
        d=RUNTIME/p.name
        if d.exists(): continue
        if p.name=='liberopro':
            shutil.copytree(p,d,ignore=shutil.ignore_patterns('assets','bddl_files','init_files','__pycache__'))
        else: d.symlink_to(p.resolve(),target_is_directory=p.is_dir())
    core=RUNTIME/'liberopro/liberopro'; old=SOURCE/'liberopro/liberopro'
    for kind in ('bddl_files','init_files'):
        (core/kind).mkdir(exist_ok=True)
        for p in (old/kind).iterdir():
            d=core/kind/p.name
            if not d.exists(): d.symlink_to(p.resolve(),target_is_directory=p.is_dir())
    if not (core/'assets').exists(): (core/'assets').symlink_to((old/'assets').resolve(),target_is_directory=True)
    spec=importlib.util.spec_from_file_location('official_pro_perturbation',OUT/'upstream/perturbation.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    records=[]
    for suite in SUITES:
        d=core/'bddl_files'/f'{suite}_env'; d.mkdir(exist_ok=True)
        for p in sorted((old/'bddl_files'/suite).glob('*.bddl')):
            text=p.read_text()
            perturb=mod.EnvironmentReplacePerturbator(mod.BDDLParser(text),str(OUT/'upstream/ood_environment.yaml'))
            result=perturb.perturb(suite,p.stem,seed=42)
            dest=d/p.name
            if dest.exists(): assert dest.read_text()==result
            else:dest.write_text(result)
            old_env=perturb._extract_current_env(suite,p.stem)
            records.append(dict(suite=suite,task=p.stem,old_environment=old_env,new_environment='living_room_table',
                                effective_shift=old_env!='living_room_table',bddl=str(dest)))
        assert len(list(d.glob('*.bddl')))==10
        (core/'init_files'/f'{suite}_env').mkdir(exist_ok=True)
    report=dict(protocol='extra1200_v1',upstream=json.loads((OUT/'upstream-source.json').read_text()),
                official_environment_implementation_unmodified=True,
                environment_tasks=records,environment_effective_tasks=sum(x['effective_shift'] for x in records),
                note='Official EnvironmentReplacePerturbator hardcodes living_room_table; five existing living-room tasks have no environment change. Report these controls separately.')
    (OUT/'assets-definition.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ASSETS_PREPARED',len(records),report['environment_effective_tasks'])
if __name__=='__main__':main()
