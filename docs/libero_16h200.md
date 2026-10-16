# LIBERO：启智 16×H200、单卡 16、离线 W&B

入口：`scripts/libero/train_16h200.sh`。需要 **2 个 GPU 节点，每节点 8 张 H200**；
两个节点都挂载同一个 `/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037`。
在每个节点各启动一次脚本。主角色也必须有 8 张 GPU，不能用无 GPU 的协调节点代替。

## 配置

继续读取官方 `configs/libero/train_action.yaml`，不复制一套容易漂移的训练配置。
单卡 batch 从官方的 32 改为 16、GPU 数从 8 改为 16；梯度累积仍为 1，global batch 仍为 256。
按用户要求，16 卡入口已改用 **ZeRO-1**（官方为 ZeRO-2），保持无 CPU offload、
`overlap_comm=false`、`contiguous_gradients=false`；配置在 `configs/libero/deepspeed_zero1.json`。
其余训练配方保持官方值：Action1B + proprio，冻结 Video，BF16，
LR 1e-4、warmup 500、cosine 下限 0.01、30,000 步，AdamW betas=(0.9,0.95)、
weight decay=0.01、clip=1、seed=42。每卡 DataLoader workers=16，
每 10 步记录训练指标、每 200 步运行固定样本动作诊断、每 5,000 步保存权重及完整训练状态。
固定样本动作诊断不是 LIBERO 模拟器成功率评测。

新任务从 `weights/libero/action_init.pt` 开始，冻结的 Video 来自
`weights/libero/video_step010000.pt`。数据沿用已经准备的四套公开 LIBERO；
与 VPP2 精确训练数据的差异和匹配证据见 [数据来源](libero_8h200.md#数据与权重来源)。
不会自动接着已有 ZeRO-2 任务续训，也不直接加载 ZeRO-2 的完整优化器分片状态。

本机 `vpp2` 环境已补装 `wandb==0.30.0`。新建环境可执行
`python -m pip install -r requirements-wandb.txt`。离线运行无需 W&B 登录；
不上传训练指标。原理见 [W&B 离线说明](https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues)。

## 两个节点都执行的环境准备

```bash
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/miniconda/etc/profile.d/conda.sh
conda activate vpp2

export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export CUDA_HOME=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/tools/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=2
export RUN_NAME=libero_action_16h200_b16_zero1
```

同一次训练两个节点的 `RUN_NAME` 必须相同；新实验使用一个新名字。
输出目录的已有配置检查会拒绝意外覆盖旧训练。

## 启智分布式训练任务

选择 PyTorch 分布式任务、2 个实例、每实例 8 张 H200。按平台 `PET_*` 变量格式，
将下面完整命令放进**每个实例**的启动命令：

```bash
bash -c '
set -euo pipefail

cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/miniconda/etc/profile.d/conda.sh
conda activate vpp2

export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export CUDA_HOME=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/tools/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=2

export NNODES="${PET_NNODES:?}"
export NPROC_PER_NODE="${PET_NPROC_PER_NODE:?}"
export NODE_RANK="${PET_NODE_RANK:?}"
export MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:?}}"
export MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"

export RUN_NAME="libero-action-16gpu-b16-zero1-${TRAIN_JOB_ID:?}-round${RUNNING_ROUND:-0}"
export WANDB_PROJECT=VPP2-LIBERO
export WANDB_NAME="$RUN_NAME"
export WANDB_MODE=offline

bash scripts/libero/train_16h200.sh \
  max_steps=30000 learning_rate=1e-4 warmup_steps=500 \
  save_every=5000 log_every=10 2>&1
'
```

脚本固定单卡 batch=16、累积=1、workers=16，并检查拓扑为 2×8。
`RUNNING_ROUND` 用于区分输出目录；此命令每轮从 Action 初始化开始，不自动查找断点续训。
脚本使用 `MASTER_ADDR` / `MASTER_PORT`，并优先使用 `NODE_RANK`；
未提供时也支持直接读取 `PET_*`，节点编号还可回退至 launcher 启动前的平台 `RANK`（必须为 0 或 1）。
若平台未注入这些变量，脚本会明确报错，此时按下面的手动方式指定。
不要把 GPU 全局进程 rank 或 world size 当作节点编号。底层 Accelerate 会为 16 个 GPU 进程设置正确的 rank/world size。

两个节点应在同一个可互通的 GPU 计算组中，主节点地址不能是 `localhost` / `127.0.0.1`。
脚本沿用现有 NCCL/RDMA 环境；若平台提供 `ARNOLD_RDMA_DEVICE`，会传给 `NCCL_IB_HCA`。
不硬编码网卡名。

## 手动启动两个节点

在两个节点分别完成上面的环境准备。将下方地址替换为 **主节点可被另一个节点访问的内网 IP 或主机名**。

主节点（rank 0）：

```bash
export MASTER_ADDR=主节点内网IP
export MASTER_PORT=29616
export NODE_RANK=0
bash scripts/libero/train_16h200.sh
```

工作节点（rank 1）：

```bash
export MASTER_ADDR=与主节点相同的内网IP
export MASTER_PORT=29616
export NODE_RANK=1
bash scripts/libero/train_16h200.sh
```

手动启动时两端都需要保持命令运行，例如各自在 tmux 中执行。只启动一端会等待另一端。
可先把最后一行换为 `bash scripts/libero/train_16h200.sh --dry-run` 检查配置，
应显示 `global_batch: 256` 和 `max_steps: 30000`；dry-run 不验证 NCCL 连通性。

## 日志、权重与离线 W&B

默认 `RUN_NAME=libero_action_16h200_b16_zero1` 时：

```text
runs/libero/libero_action_16h200_b16_zero1/
  config.yaml                 # 本次解析后的完整配置
  checkpoints/weights/        # 每 5k 步的紧凑 Action 权重
  checkpoints/state/          # 每 5k 步的完整恢复状态
  eval/                       # 固定样本动作诊断
  wandb/offline-run-*/         # 只由全局 rank 0 写入
runs/libero/libero_action_16h200_b16_zero1_logs/
  node0.log
  node1.log
```

W&B 记录完整解析后的配置、loss / loss_action / loss_video、学习率、梯度范数、
吞吐和诊断指标。没有开启逐层梯度或参数直方图采集。

终端每 10 步显示 `step=当前步/30000`、loss、梯度范数、学习率、step/s 和 ETA；
目前是逐条打印的进度日志，没有原地刷新的 tqdm 图形进度条。
上面的启智命令使用带任务 ID 的 `RUN_NAME`，对应日志和输出路径也会使用这个名字。

```bash
tail -f runs/libero/libero_action_16h200_b16_zero1_logs/node0.log
```

离线目录是供后续同步的记录文件，不会自动生成在线曲线。
需要上传时，在能访问 W&B 的机器上登录后，手动同步目标 `offline-run-*` 目录：

```bash
wandb login
WANDB_MODE=online wandb sync runs/libero/libero_action_16h200_b16_zero1/wandb/offline-run-具体目录
```

## 验证范围

已检查两节点启动参数、官方训练配置的一致性，以及真实 W&B 离线写入和指标回读。
切换 ZeRO-1 后，6 项启动测试通过；本地单 GPU 的小模型 BF16 / 累积 1 检查中，
实际梯度 hook 注册数量为 **0**，梯度与普通 PyTorch 参考结果完全一致，
可训练参数更新、冻结参数保持不变。见
[单卡数值检查](validation/libero_zero1_bf16_gradient_reference_1gpu.json)。
该测试覆盖共享参数和 non-reentrant 激活 checkpoint，不代表完整模型的多卡验证。
此前 16 卡 ZeRO-2 已完成真实训练及 RDMA/GDR 通信验证，稳态约 8.16 秒/step；
采样发现 DeepSpeed 参数梯度 hook 的重复扫描占据大量 CPU 时间。
本次只切换后续启动配置，当前正在运行的 ZeRO-2 进程不受影响。
ZeRO-1 的完整 16 卡吞吐、显存和恢复仍须实测，不能把配置检查当作端到端验证。
