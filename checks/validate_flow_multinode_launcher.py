"""Validate actual Accelerate dispatch, then exercise two node ranks on loopback.

The Gloo smoke test checks launch/rank propagation, not cross-host NCCL hardware.
Run from repository root with the isolated Flow Python environment.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import timedelta


def worker(output):
    import torch
    import torch.distributed as dist
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    value = torch.tensor(rank + 1)
    dist.all_reduce(value)
    assert world == 2 and value.item() == 3
    assert Path(sys.executable).resolve() == Path(os.environ['FLOW_EXPECTED_PYTHON']).resolve()
    result = dict(rank=rank, world_size=world, local_rank=int(os.environ['LOCAL_RANK']),
                  group_rank=int(os.environ['GROUP_RANK']), sum=value.item(), python=sys.executable)
    (output / f'rank{rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--output', type=Path, default=Path('runs/libero_flow_failure_20261026/launcher_validation'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.output)
        return
    from accelerate.commands.launch import launch_command_parser, _validate_launch_command
    from accelerate.utils.launch import prepare_deepspeed_cmd_env
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    common = ['--config_file', 'scripts/libero_flow/accelerate_zero1.yaml',
              '--main_process_ip', '127.0.0.1', '--main_process_port', str(port)]
    plans = []
    for nodes, processes in ((1, 2), (2, 16)):
        for rank in range(nodes):
            parsed = launch_command_parser().parse_args(common + ['--num_machines', str(nodes),
                '--num_processes', str(processes), '--machine_rank', str(rank), 'unused.py'])
            parsed, defaults, _ = _validate_launch_command(parsed)
            parsed.deepspeed_fields_from_accelerate_config = ','.join(defaults.deepspeed_config)
            command, _ = prepare_deepspeed_cmd_env(parsed)
            assert command is None, command  # must dispatch torchrun, never pdsh/hostfile
            assert int(parsed.nproc_per_node) == processes // nodes
            if nodes > 1:
                assert int(parsed.nnodes) == nodes and parsed.node_rank == rank
                assert (getattr(parsed, 'master_addr', None) == '127.0.0.1'
                        or parsed.rdzv_endpoint == f'127.0.0.1:{port}')
            plans.append(dict(nodes=nodes, world=processes, node_rank=rank,
                              nproc_per_node=parsed.nproc_per_node, launcher=parsed.deepspeed_multinode_launcher))
    env = dict(os.environ, FLOW_EXPECTED_PYTHON=sys.executable, OMP_NUM_THREADS='1')
    # Clear inherited worker topology, allowing torchrun to populate it.
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK'):
        env.pop(key, None)
    processes = []
    try:
        for rank in range(2):
            command = [sys.executable, '-m', 'accelerate.commands.launch', *common,
                '--num_machines', '2', '--num_processes', '2', '--machine_rank', str(rank),
                str(Path(__file__).resolve()), '--worker', '--output', str(args.output.resolve())]
            with (args.output / f'node{rank}.log').open('w') as log:
                processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait(timeout=150) for p in processes]
        assert codes == [0, 0], codes
        results = [json.loads((args.output/f'rank{r}.json').read_text()) for r in range(2)]
        report = dict(status='passed', dispatch_plans=plans, loopback_gloo=results,
                      limitation='Not a physical two-host NCCL/full-model training test.')
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()


if __name__ == '__main__':
    main()
