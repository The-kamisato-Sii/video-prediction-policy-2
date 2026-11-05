"""Register the additional environment suites only in this evaluator process."""
from pathlib import Path
import re
from vpp2.libero.protocols import BASE_SUITES

def register_environment_suites():
    from vpp2.libero.pro_alias import route_liberopro_as_libero
    from vpp2.libero.pro_runtime import configure_harnessvla_liberopro_runtime
    route_liberopro_as_libero()
    configure_harnessvla_liberopro_runtime()
    from libero.libero import benchmark as b, get_libero_path
    for base in BASE_SUITES:
        name=base+'_env'
        if name in b.BENCHMARK_MAPPING: continue
        tasks={}
        for key,t in b.task_maps[base].items():
            path=Path(get_libero_path('bddl_files'))/name/t.bddl_file
            match=re.search(r'\(:language\s+([^)]*)\)',path.read_text(),re.S)
            assert match, path
            tasks[key]=t._replace(problem_folder=name,language=' '.join(match.group(1).split()).strip('"'))
        b.task_maps[name]=tasks
        b.register_benchmark(b._make_perturbation_benchmark(name))
