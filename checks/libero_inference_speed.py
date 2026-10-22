"""Matched-observation latency benchmark using the production LIBERO policy path.

Use separate processes per model/GPU, and swap GPU assignments in round two.
CUDA synchronization brackets uninstrumented end-to-end timings. Component
profiling is a separate pass and is excluded from the latency statistics.

From the repository root, source scripts/libero/activate_pro_env.sh. First run:
  python checks/libero_inference_speed.py capture --output RUN --config FM_EVAL
Then run worker with --output RUN --config FM_EVAL --method flow_matching
--gpu 0 --round 1, and with MF_EVAL / meanflow / GPU 1. Repeat with GPUs swapped
and --round 2. Finally run summarize --output RUN. Use a fresh RUN directory.
Both evaluation configurations must point at the corresponding 30k checkpoints.
Each worker defaults to 10 warmup and 50 measured calls, batch size 1.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import time


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def setup(cfg_path, output, gpu):
    from omegaconf import OmegaConf
    from vpp2.libero.evaluate import runtime_environment, worker_config
    evaluation = OmegaConf.load(cfg_path)
    evaluation.runtime_config_dir = str(output / 'runtime')
    os.environ.update(runtime_environment(evaluation, gpu))
    sys.path.insert(0, str(Path(evaluation.benchmark_root).resolve()))
    from vpp2.libero.pro_alias import route_liberopro_as_libero
    from vpp2.libero.pro_runtime import configure_harnessvla_liberopro_runtime
    route_liberopro_as_libero()
    configure_harnessvla_liberopro_runtime()
    cfg = worker_config(evaluation, output, output / 'unused-manifest.txt', gpu, None)
    return evaluation, cfg


def capture(args):
    import numpy as np
    _, cfg = setup(args.config, args.output, args.gpu)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from vpp2.libero.eval_helpers import load_trusted_task_init_states
    keys = ['agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos',
            'robot0_eef_quat', 'robot0_gripper_qpos']
    tasks = ['libero_spatial_swap', 'libero_object_swap', 'libero_goal_swap',
             'libero_10_swap', 'libero_spatial_task']
    manifest = []
    for index, suite_name in enumerate(tasks):
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(0)
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        try:
            env.seed(1)
            env.reset()
            obs = env.set_init_state(load_trusted_task_init_states(suite, 0)[1])
            for _ in range(15):
                obs, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            path = args.output / f'obs_{index}.npz'
            np.savez(path, **{key: np.asarray(obs[key]) for key in keys})
            manifest.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 suite=suite_name, task_id=0, init_state=1, settle_steps=15,
                                 prompt=task.language))
        finally:
            env.close()
    write_json(args.output / 'inputs.json', manifest)
    print('CAPTURE_PASS', len(manifest), flush=True)


def worker(args):
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    evaluation, cfg = setup(args.config, args.output, args.gpu)
    from vpp2.libero.evaluate import validate_checkpoint
    from vpp2.libero.eval_helpers import _load_model_checkpoint
    from vpp2.libero.eval_libero_single import _predict_action_chunk
    from vpp2.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    metadata = validate_checkpoint(evaluation)
    assert metadata['action_sampler'] == args.method
    torch.set_num_threads(4)
    torch.manual_seed(42)
    inputs_path = args.output / 'inputs.json'
    manifest = json.loads(inputs_path.read_text())
    observations = []
    for item in manifest:
        path = args.output / item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            observations.append({k: arrays[k].copy() for k in arrays.files})
    print('LOADING', args.method, 'gpu', args.gpu, flush=True)
    started = time.perf_counter()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device='cuda')
    _load_model_checkpoint(model, cfg.ckpt, video_checkpoint_override=cfg.EVALUATION.video_checkpoint_override)
    model = model.to('cuda').eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.EVALUATION.dataset_stats_path))
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    assert model.action_sampler == args.method if hasattr(model, 'action_sampler') else args.method == 'flow_matching'
    def predict(i):
        action, _, _ = _predict_action_chunk(
            observations[i % len(observations)], manifest[i % len(manifest)]['prompt'],
            model, processor, cfg, action_horizon=32, input_w=448, input_h=224,
            model_device='cuda',
        )
        assert action.shape == (32, 7) and np.isfinite(action).all()
        return action
    started = time.perf_counter()
    predict(0)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - started) * 1000
    for i in range(args.warmup):
        predict(i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latency = []
    print('TIMING', args.iterations, flush=True)
    for i in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        predict(i)
        torch.cuda.synchronize()
        latency.append((time.perf_counter() - started) * 1000)
        if (i+1) % 10 == 0:
            print('SAMPLES', i+1, 'mean_ms', float(np.mean(latency)), flush=True)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    # Separate instrumented pass. CUDA events include GPU-side execution and
    # stream gaps within a component; no synchronization is inserted inside it.
    stages = {'vae': '_encode_input_image_latents_tensor', 'clip': '_encode_clip_fea_from_image',
              'text': 'encode_prompt', 'video_dit': '_build_frozen_video_cache',
              'action_dit': '_predict_action_noise_with_cache'}
    events = {key: [] for key in stages}
    originals = {}
    for key, name in stages.items():
        original = getattr(model, name)
        originals[name] = original
        def timed(*a, _key=key, _fn=original, **kw):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = _fn(*a, **kw)
            end.record()
            events[_key].append((start, end))
            return result
        setattr(model, name, timed)
    for i in range(5):
        predict(i)
    torch.cuda.synchronize()
    counts = {key: len(value)/5 for key, value in events.items()}
    assert counts['video_dit'] == 1
    assert counts['action_dit'] == int(evaluation.action_nfe)
    stage_ms = {key: sum(a.elapsed_time(b) for a,b in value)/5 for key,value in events.items()}
    result = dict(method=args.method, physical_gpu=args.gpu, round=args.round,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  inputs_sha256=hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
                  checkpoint=str(evaluation.action_checkpoint), checkpoint_step=30000,
                  batch_size=1, dtype='bf16', action_horizon=32, executed_actions_per_replan=10,
                  image_hw=[224,448], video_frames=17, nfe=int(evaluation.action_nfe),
                  warmup=args.warmup, iterations=args.iterations, model_load_s=load_s,
                  first_call_ms=first_ms, mean_ms=float(np.mean(latency)),
                  median_ms=float(np.median(latency)), p95_ms=float(np.percentile(latency,95)),
                  std_ms=float(np.std(latency)), replans_per_s=1000/float(np.mean(latency)),
                  peak_allocated_gib=peak_allocated, peak_reserved_gib=peak_reserved,
                  component_cuda_ms_per_replan=stage_ms, component_calls_per_replan=counts,
                  latency_ms=latency, evaluation=OmegaConf.to_container(evaluation,resolve=True))
    write_json(args.output / f'{args.method}_gpu{args.gpu}_round{args.round}.json', result)
    print('BENCHMARK_PASS', json.dumps({k:result[k] for k in ('method','physical_gpu','mean_ms','p95_ms','peak_allocated_gib','component_cuda_ms_per_replan')}), flush=True)


def summarize(args):
    import numpy as np
    rows = [json.loads(p.read_text()) for p in args.output.glob('*_gpu*_round*.json')]
    assert len(rows) == 4
    assert len({r['inputs_sha256'] for r in rows}) == 1
    report = {'scope': 'production observation-to-action-chunk path; no simulator stepping/video writing/model loading in steady latency', 'methods': {}, 'runs': rows}
    for method in ('flow_matching', 'meanflow'):
        selected = [r for r in rows if r['method']==method]
        assert {r['physical_gpu'] for r in selected} == {0,1}
        values = [v for r in selected for v in r['latency_ms']]
        report['methods'][method] = dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),samples=len(values),replans_per_s=1000/float(np.mean(values)),peak_allocated_gib=max(r['peak_allocated_gib'] for r in selected),components_ms={k:float(np.mean([r['component_cuda_ms_per_replan'][k] for r in selected])) for k in selected[0]['component_cuda_ms_per_replan']})
    fm,mf=(report['methods'][k]['mean_ms'] for k in ('flow_matching','meanflow'))
    report.update(speedup=fm/mf,latency_reduction_percent=(1-mf/fm)*100)
    write_json(args.output/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['capture','worker','summarize'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--method',choices=['flow_matching','meanflow'])
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--iterations',type=int,default=50)
    args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(exist_ok=True,parents=True)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    os.environ['MUJOCO_EGL_DEVICE_ID']=str(args.gpu)
    globals()[args.mode](args)
