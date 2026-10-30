<div align="center">

# Video Prediction Policy 2: Predict Better, Act Better

Official implementation of **["Video Prediction Policy 2: Predict Better, Act Better"](https://arxiv.org/abs/2610.10270)**.

[![arXiv: 2610.10270](assets/badges/arxiv.svg)](https://arxiv.org/abs/2610.10270)
[![Project Page](assets/badges/project-page.svg)](https://robert-gyj.github.io/video-prediction-policy-2/)
[![ModelScope — Models](assets/badges/modelscope.svg)](https://modelscope.cn/models/haodong123/VPP2)
[![Hugging Face — Models](assets/badges/huggingface.svg)](https://huggingface.co/Haodong082399/VPP2)

</div>

## News

- **[2026-10-10]** 🎬 **VPP2 video model weights are now available!** Download the **Stage-1 (49 frames)** and **Stage-2 (17 frames)** checkpoints on [Hugging Face](https://huggingface.co/Haodong082399/VPP2/tree/main/checkpoints_video) and [ModelScope](https://modelscope.cn/models/haodong123/VPP2/files), and get started with our [zero-shot video prediction guide](docs/video_prediction.md).

- **[2026-10-08]** 🥇 **VPP2 ranks #1 overall on the official [RoboDojo-Sim leaderboard](https://robodojo-benchmark.com/leaderboard)**, achieving state-of-the-art performance across agent and non-agent tracks with **39.26 Score** and **32.26% SR**.

## Overview

We introduce **Video Prediction Policy 2 (VPP2)**, a world-action model (WAM) with
strong zero-shot generalization in both video prediction and action generation.

Our pipeline has three stages: (1) **event-level video pretraining**, (2) **single-step visual planning through post-training and distillation**, and (3) **action learning with a mixture-of-transformers (MoT) architecture**.

This codebase includes:

- **[Zero-shot video prediction](docs/video_prediction.md).** Generate videos from an initial image and a text instruction using the **Stage-1 (49 frames)** or **Stage-2 (17 frames)** video model, with input preparation, prompt enhancement, and ready-to-run examples.
- **Policy post-training on [RoboDojo](docs/robodojo.md) and [LIBERO](docs/libero.md).** Training configurations and scripts for adapting VPP2 to both benchmarks, together with policy evaluation workflows. For RoboDojo, we add **history frames to the model input** to support tasks that require memory. To accommodate this architecture change, we start post-training from the **Stage-2 video model before distillation**.

<p align="center">
  <a href="assets/teaser.png"><img src="assets/teaser.png" alt="VPP2 paper overview: event-level video prediction, instruction following, and robot policy learning." width="100%"></a>
</p>

## 本机环境、训练与验证（2026-10-15）

原始 FM baseline 的单机 **8×H200** 配置、固定版本数据准备和测速命令见
**[RoboDojo 8 卡 baseline](docs/robodojo_8h200.md)**。
入口为 `bash scripts/robodojo/train_8h200.sh`；保持原配方 100k 次更新、
全局 batch 288（每卡 3 × 累积 12），从 history-10k 初始化，使用 ZeRO-2。

**冻结 Video、只训练 Action2B 的容量实验（2026-10-16）：** 8×H200、BF16、
ZeRO-2、累积 1 时，每卡 batch **28** 完成 3 次更新，**29** 前向 OOM；
batch 24 的显存峰值约 122.6 GiB，batch 28 约 136.4 GiB。这些是 MoT 训练态修复前、
激活 checkpoint 未实际生效时的历史结果，不能作为当前代码的容量上限。
这与官方 RoboDojo 联合训练配方不同。测试命令、checkpoint 来源和 ZeRO-2
数值检查见 **[冻结 Video 容量实验](docs/robodojo_frozen_video.md)**。

**LIBERO 8×H200（2026-10-16）：** 官方 Action 阶段冻结 Video，训练 Action1B +
proprio，global batch 256、LR 1e-4、warmup 500、30k 步。
已下载并校验 LIBERO Video-10k 和四套 MuJoCo 3.3.2 公开数据，生成 40 条文本缓存及新 Action 初始化。
修复了 MoT 激活 checkpoint 未生效和多卡启动的目录检查竞态。每卡 32 已通过 20 步 +
恢复 2 步，峰值 reserved **137.18 GiB**，余量不足；默认改为每卡 **16 × 累积 2**，
global 256，8 步验证峰值 reserved **87.23 GiB**。Video 完整 hash 前后不变，各卡训练参数一致。
两种配置的短跑计时均值分别为 11.49 / 15.69 秒每更新。
之后已按用户选择启动单卡 32 / 累积 1 的 8 卡正式 30k 任务，
输出为 `runs/libero/action_b32_20261016_0719`。
官方/本机配置对照、数据来源、训练命令与短跑/恢复检查见
**[LIBERO 本机训练与实测](docs/libero_8h200.md)**。

**LIBERO 单步对照：** [MeanFlow / 直接 Action MSE 的 16 卡启动指令与控制变量](docs/libero_one_step.md)。默认首帧＋uint8、每卡 4 workers；从相同 Action 初始化开始，冻结 Video-10k。

**当前实现（2026-10-20）：** [LIBERO MeanFlow 自适应 loss 加权](#libero-mf-adaptive-loss)。LIBERO fresh-init MeanFlow 默认使用按样本、停止梯度的 adaptive weighting（p=0.5、c=0.001），保留当前时间采样和 BF16 配方。缺省字段的历史配置仍解释为 uniform；验证记录见 `docs/libero_one_step.md`。

**下一轮实验规格（尚未实现）：** [iMF boundary / 辅助 v-head 两种方案](#libero-imf)，包括目标公式、40 层与 Video KV 接入、adaptive/对角采样控制变量、代码修改与验收。

**LIBERO 数据管线优化实测：** 同机 8×H200、ZeRO1、单卡 batch 32 下，只解码首帧将平均步耗时从 14.20 秒降至约 4.77 秒；workers 4/8/16/20 吞吐相近。完整对照表及运行方法见 **[数据管线性能对照](docs/libero_loader_benchmark.md)**。

**LIBERO 16×H200 启动准备：** 2 个节点，每节点 8 卡，每卡 batch 16、累积 1，
global batch 256；按用户要求切换为 ZeRO-1，其余训练配置沿用官方 LIBERO YAML，
每 5k 步保存，启用离线 W&B。
入口为 `bash scripts/libero/train_action.sh`，每个节点各执行一次。
完整 `cd` / `source` / `conda activate`、启智变量及手动主/工作节点命令见
**[LIBERO 16 卡与离线 W&B](docs/libero_16h200.md)**。此前 16 卡 ZeRO-2 已跑通；
ZeRO-1 的完整 16 卡训练吞吐尚未实测。

已新增可选的 D0 轨迹蒸馏、M0 MeanFlow、K0 可学习 K-MF、K1 中点 K-MF，
保留原 FM 默认路径。完整安装、数据/权重路径、各方法训练、恢复、导出和
RoboDojo / LIBERO 测试命令见 **[MeanFlow 使用与验证](docs/meanflow.md)**。
其中 [各方法训练命令](docs/meanflow.md#commands-for-every-method) 覆盖 B0 原始 FM、
B1 同权重一步推理、B2 FM 续训，以及 D0 / M0 / K0 / K1；B1 不需要重新训练。

```bash
source /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/miniconda/etc/profile.d/conda.sh
conda activate vpp2
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
export PYTHON_BIN="$CONDA_PREFIX/bin/python"

# 将 distill 改为 mf、kmf_learned、kmf_midpoint；两个 benchmark 均有对应配置。
bash scripts/meanflow/train.sh --benchmark robodojo \
  --config configs/meanflow/robodojo_distill.yaml --dry-run
bash scripts/meanflow/train.sh --benchmark libero \
  --config configs/meanflow/libero_mf.yaml --dry-run

# 数学、梯度、原路径兼容性与实际 1-NFE 检查
PYTHONPATH=src OMP_NUM_THREADS=2 python -m pytest -q \
  checks/test_meanflow.py checks/test_robodojo_adapter.py checks/test_clip_offload.py
# 保留 40 层、缩小宽度的 GPU 工程短跑；不是完整模型或闭环结果
PYTHONPATH=src python checks/smoke_meanflow.py --device cuda --steps 20 \
  --output runs/new_engineering_probe
```

**验证范围：** [工程短跑记录](docs/validation/meanflow_engineering_probe.json) 包含
两个 benchmark × 四种方法各 20 次更新、checkpoint 恢复与继续训练。
完整训练配方、真实数据收敛和 RoboDojo 2100-episode 闭环成绩需单独验证；
不能从这些小宽度测试推断。单张约 48 GiB 4090 无法直接承载原 ZeRO-1/2
全参数 joint 配方以及独立 teacher。正式训练仍须按显存需求配置分布式资源。
几何 heads / VGGT 保持关闭，等待 README 中的一步闭环验收。

**8×H200 验证：** NCCL 通信、BF16 反向传播，以及两个 benchmark 的小模型
ZeRO-2 保存/恢复均通过。原尺寸 M0（Video / Action 均为 40 层，约 184 亿可训练参数）
已在 8 卡上各完成 2 次优化器更新，真实 VAE/CLIP 编码、Video/Action/proprio 抽样梯度及
Action 权重变化均通过检查；PyTorch 已分配显存峰值 **78.69 GiB/卡**，预留峰值
**96.59 GiB/卡**。输入为合成数据，全局 batch 为 8，不能代替 batch 288 的正式训练。
平台随后于 2026-10-15 15:45 UTC 抢占实例：K0 在优化器初始化阶段中断，K1/D0
尚未执行原尺寸测试。[验证记录](docs/validation/h200_8gpu_training.json) 与
[复跑命令](docs/meanflow.md#eight-gpu-h200-diagnostics) 已保存。

**完整策略推理已验证：** 原 joint-100k 权重、10 步 FM、CPU UMT5，输入合成 RGB/EE16
后得到有限的 `32×16` 动作；3 次固定 seed 请求完全一致。同步测得 p50 **2.316 s**、
p95 **2.385 s**，GPU 峰值 **38.41 GiB**（包含计时 instrumentation；不是闭环成功率）。
详见 [完整策略测试记录](docs/validation/robodojo_policy_smoke_cu128.json)。

```bash
bash scripts/robodojo/smoke_policy.sh \
  --bundle weights/robodojo_release/checkpoints/joint2b_s100000 \
  --wan weights/robodojo_release/checkpoints/Wan2.1-I2V-14B-480P \
  --synthetic --steps 10 --text-encoder-device cpu \
  --output runs/new_full_policy_smoke
```

**同卡 RoboDojo 原生闭环已验证：** 本机约 48 GiB 的 RTX 4090 同时运行策略和
Isaac Sim，`stack_bowls` / layout 0 在第 **330 / 800** 步成功，单回合得分 **100**。
三路视频均为 331 帧且解码通过，客户端正常退出；整卡显存每 2 秒采样峰值
**45.60 GiB**。这是一次部署验证，不能作为 2100-episode 的基准成绩或 MeanFlow 成绩。
[完整记录](docs/validation/robodojo_native_stack_bowls_b0.json)；本地视频、评分和日志位于
`runs/robodojo_native_stack_bowls_b0/`。

```bash
# 终端 1：vpp2 环境，工作目录为本仓库。
bash scripts/robodojo/install_adapter.sh --robodojo-root ../RoboDojo --update
PYTORCH_ALLOC_CONF=expandable_segments:True bash scripts/robodojo/server.sh \
  --robodojo-root ../RoboDojo \
  --bundle "$PWD/weights/robodojo_release/checkpoints/joint2b_s100000" \
  --wan "$PWD/weights/robodojo_release/checkpoints/Wan2.1-I2V-14B-480P" \
  --gpu 0 --port 23000 --text-encoder-device cpu \
  --defer-model-load --release-cuda-cache --offload-clip

# 终端 2：同一工作目录，脚本使用独立的 robodojo 环境。
bash scripts/robodojo/smoke_client.sh --robodojo-root ../RoboDojo \
  --sim-python ../miniconda/envs/robodojo/bin/python --port 23000 --task stack_bowls
```

冷启动需要数分钟。CLIP 仍在 GPU 上计算，随后将权重移回 CPU，为视频模型留出显存；
同一输入及 seed 的完整策略对比中，卸载前后的 `24×16` 动作完全一致。
现有 22 项单元测试通过；正式训练所需的数据和分布式资源限制见上方验证范围。


## 本分支研究计划：40 层 Action DiT → 一步 MeanFlow → 几何辅助监督

> **状态：以下保留原研究规格。** 阶段 1–3 的工程实现与当前已验证范围见上面的使用指南；完整基线复现、正式训练和闭环验收尚未由工程测试替代。后续几何 / VGGT 仍是待验证研究计划。

### Material Passport

- 类型：代码实验计划；使用 academic-research-suite / experiment-agent 的实验规划流程。
- 源码核查日期：2026-10-15；目标分支：`baseline-setup`。
- 核查提交：[`4b0540937a6c8ee1d44fd1d887ec1041bae3da6e`](https://github.com/The-kamisato-Sii/video-prediction-policy-2/tree/4b0540937a6c8ee1d44fd1d887ec1041bae3da6e)。后续实现若基于新提交，先核对本节的路径、符号和配置差异。
- 计划撰写时仅完成源码与协议核查。当前工程短跑和完整 FM 策略推理记录见上方；本研究规格中的正式训练与闭环成绩仍待验证。
- 实验假设、建议默认值和验收阈值属于本项目设计，不是论文已证明的结论。

### 1. 实验目的与阶段边界

按以下顺序验证，前一阶段未通过时不要同时加入后一阶段：

1. **复现当前 VPP2。** 使用原 40 层 Action DiT、配对的 Video/Action checkpoint 和各 benchmark 原训练、测试协议，记录 10 步 FM 的真实结果。
2. **先做 MeanFlow 参数化的一步蒸馏。** 冻结已验证的 FM teacher，student 保留原 40 层主干，从相同的配对 Video / Action / proprio 权重初始化；首先只改变 action 学习目标及其所需的双时间条件、一步采样接口。
3. **比较 MeanFlow 训练方法及改进。** 分开测试原始 data-based MeanFlow、可学习分割 K-MF、中点消融、方向正则和导数实现。先单项比较，再决定是否组合。
4. **确认一步闭环有效后，再增加几何 heads。** 保留选定的一步 action 方法，增加一次主干前向即可读出的 3D point / flow / camera pose 辅助输出，验证几何监督是否改善 action。
5. **最后评估 VGGT 预训练权重的增益。** 本节只定义这个阶段的研究问题和公平对照，**不规定 VGGT 权重如何加载、映射或适配**。

本项目最终希望回答三个独立问题：

- Q1：在 VPP2 原 benchmark 协议下，40 层 Action DiT 能否用 **1 次 action 网络求值（1 NFE）**保持接近原 10 步 FM 的闭环性能，并降低实测端到端时延？
- Q2：在 action 方法和预算匹配时，显式几何辅助监督能否改善成功率、泛化或数据效率，而不只是降低几何误差？
- Q3：在架构和监督匹配时，VGGT 权重是否带来额外增益？

阶段 1–3 不增加 loop、不减少主干层数、不更换原 MoT/Video KV 接口、不加入几何 heads。本文中的“40 层”始终指 **Action DiT 的 40 个 blocks**；Video expert 也有独立的 40 层。1 NFE 指 action expert 一次完整求值，不表示整个系统只运行一个网络，也不表示训练只需要一次 forward。

### 2. 当前源码与必须保留的基线协议

当前实现已经包含一些条件路由、Video token 投影等可选接口。第一阶段不得因为这些接口存在就启用它们；使用原配置中的直接逐层 Video KV、原 mask 和原 action 表示。

| 项目 | RoboDojo | LIBERO |
|---|---|---|
| 训练配置 | `configs/robodojo/train_100k.yaml` | `configs/libero/train_action.yaml` |
| 原训练阶段 | joint Video + Action | action-only，Video expert 冻结 |
| Action DiT | 40 层，hidden 1024，FFN 4096 | 40 层，hidden 512，FFN 2048 |
| 注意力维度 | 两者均为 40 heads × 128；不能把它当成 residual hidden width | 同左 |
| Video DiT | 40 层，hidden 5120 | 同左 |
| action / proprio 维度 | 16 / 16 | 7 / 8 |
| action horizon / token chunk size | 32 / 1 | 32 / 1 |
| action 可直接看见的 Video latent frames | 5 | 3；仍保留完整 5 个 video latent frames 的 Video self-attention |
| 数据与归一化 | 原任务平衡、critical-phase 采样、history 条件、绝对 EE16、z-score | 原四个训练 suite、两视角水平拼接、min/max、原 delta/gripper 规则 |
| 参考全局 batch / 原训练步数 | 288 / 100k | 256 / 30k |
| 原学习率 | Video 5e-6，Action 1e-4 | Action 1e-4 |
| 原训练随机种子 | 100008 | 42 |
| 原 action 训练 shift | 3 | 5 |
| 原闭环 action 推理 | 10 步 Euler，shift 1，seed 1 | 10 步 Euler，shift 5；Standard/PRO policy seed 42，OOD 为 7 |
| 每次执行 action 数 | 24 | 10 |

完整协议以 [RoboDojo 文档](docs/robodojo.md)、[评测文档](docs/evaluation.md)、[LIBERO 文档](docs/libero.md) 及锁定的配置为准。保留相机组成/旋转/resize、history 更新、proprio 归一化、gripper 后处理、任务清单、reset、settle no-ops、episode horizon、失败重试与统计口径。不得为了让一步模型成功而同时改变执行 horizon 或评测 seed。

特别注意：

- RoboDojo 训练内 held-out action diagnostic 默认 **shift 3**，部署闭环是 **shift 1**。二者不是同一条件，分别记录。主蒸馏 teacher 使用目标部署协议的 shift 1；若额外计算 shift-3 diagnostic，不能把它当作同一 teacher 配对误差。
- RoboDojo 的参考闭环为 54 runnable entries / 42 official cells / 2100 episodes；五个维度等权汇总，不以 pooled episode SR 替代官方指标。
- LIBERO Standard / OOD / PRO 的原 episode 预算分别为 2000 / 4500 / 800；继续使用现有配置和 audit。训练内四个固定样本 diagnostic 不是闭环成功率。
- 不照抄其他论文的“4 步 FM”作为本仓库基线。本仓库配置和部署协议是 **10 步**，不要依据 `infer_action()` 的函数默认参数猜测实际 NFE。
- 原学习率、优化器、warmup、LR tail、global batch 等从对应配置继承。蒸馏是新增训练阶段，额外 steps 和 teacher 标签成本必须单列；不能称为与原训练完全等量。第一轮预算可沿用对应 30k / 100k 作为上限，先做短跑，再按预先固定的 checkpoints 比较。

### 3. 第一阶段“蒸馏”的精确定义

这里首先实现的是 **冻结 FM teacher 的轨迹平均速度蒸馏，student 使用 MeanFlow 的区间平均速度参数化**。它不是原始无需 teacher 的 MeanFlow 自举算法；仅从 FM checkpoint 初始化，也不能单独称为蒸馏。

源码 `schedulers/scheduler_continuous.py` 的实际约定为：

$$
z_t=(1-t)a^*+t\epsilon,\quad v=\epsilon-a^*,\quad 0\le r\le t\le1.
$$

`t=0` 为 clean action，`t=1` 为 Gaussian noise；原网络时间输入为 `1000*t`。scheduler 的 delta 为负数，更新方向是从噪声到动作。不要额外翻转时间、符号或重复应用 shift。

冻结 teacher 的所有参数，设为 eval；从其匹配的 Video / Action / proprio checkpoint 构建完整 teacher。student 也从同一套已验证的配对权重初始化，启动时分别核对 hash。RoboDojo student Video 必须是配对的 joint-100k Video，不能继承训练 YAML 后意外退回 history-10k 初始化；LIBERO 也明确记录对应冻结 Video 来源。**不能重新调用 fresh action 初始化脚本覆盖训练完成的 action/proprio**。新增第二时间分支按下文初始化。

对同一原始观测、语言、proprio、history 和指定的 video/action 随机噪声，按照 teacher 原 10 步 Euler schedule 记录轨迹：

$$
z^T_{s_{i+1}}=z^T_{s_i}+(s_{i+1}-s_i)v_T(z^T_{s_i},s_i;C_T),\quad s_0=1,\ s_{10}=0.
$$

从**同一次 rollout**取两个状态，`t > r`：

$$
u_T=\frac{z_t^T-z_r^T}{t-r},\qquad
L_{\rm distill}=\operatorname{maskedMSE}\big(u_S(z_t^T,r,t;C_S),\operatorname{sg}[u_T]\big).
$$

部署为：

$$
\hat a=\epsilon-u_S(\epsilon,0,1;C_S).
$$

实施要求：

1. 第一版采用原 teacher 的 10 步 schedule；`t,r` 使用其 **实际 shifted sigma 值**，不能用步号除以 10 代替。
2. 建议首轮一半样本固定完整区间 `(t,r)=(1,0)`，另一半从同一条 teacher 轨迹随机选严格有序的节点对。这个 50% 是本项目初始配方，不是论文保证的最优比例。保存实际采样分布。
3. 同一轨迹状态只生成一次，供区间采样复用。首版不采 `r=t`；后续若加 diagonal，使用 teacher 瞬时速度单独监督，禁止除以零；设置最小非零 gap，差分与 loss 用 FP32。
4. 第一版使用独立的、非零的 uniform distillation weight。**原非 pure-noise FM `training_weight()` 在 `t=1` 为零，直接复用会消掉完整区间蒸馏的梯度。** 旧时间权重只可用于定义清楚的独立 FM 项。
5. 这个目标本身 **不需要 JVP**。teacher 的有限步数积分误差会进入标签；若改用更多 teacher steps，必须另命名实验并记录额外成本。
6. teacher rollout 内不做反归一化、gripper 二值化、逐步物理裁剪或重新加噪；先在原归一化 action 空间完成轨迹，再沿用部署后处理。使用原 action padding/valid masks。
7. 固定 teacher weights 不等于固定每条数据的噪声。训练可重采样噪声；对照与复现必须记录 RNG。不要用独立随机噪声和任意示范 action 的 endpoint MSE 冒充 teacher 蒸馏。

### 4. Teacher / student 的条件与梯度边界

**主实验保留 student 的原训练范围**：RoboDojo 继续 joint 训练 student Video、Action、proprio，并保留原 Video FM loss；LIBERO 继续冻结 Video/VAE/CLIP/text encoder，训练 student Action 与 proprio。teacher 始终全冻结。这是为了沿用原训练 setting；将全部条件模块冻结的低成本版本可额外做，但必须标为单独实验。

两支使用相同的**原始条件和随机 video 噪声**，不意味着可以共用同一个数值 KV：

- 原 `build_inputs()` 会先把 proprio projection 追加到 context，Video prefill 也会读取它。即使 Video weights 冻结，只要 student proprio projection 更新，student KV 仍可能变化。
- teacher 使用自己的冻结 projection 和 `C_T`；student 使用自己的 `C_S`。不能把 student 的 context 喂给 teacher，使 teacher target 随训练漂移；也不能长期用 teacher cache 替代 student inference cache。
- teacher 的 10 步轨迹共享 **一次 teacher prefill**。student 的一次 action 前向使用自己的 prefill；teacher/student 的 Video noise、视角、mask、history 与原始输入保持配对。
- LIBERO 原 `_build_frozen_video_cache()` 在 `no_grad` 下运行；保持这个梯度边界。student proprio 仍可从 action context 的直接路径获得梯度，但不从 frozen Video cache 路径反传。
- RoboDojo 原 joint 路径保留 action/video loss 到 student Video/proprio 的梯度。若拆成可复用 prefill + action 解码，student prefill 必须保留图，并先验证它与原 joint MoT 的数值和梯度等价。不能调用现有冻结 helper 后声称保持 joint setting。
- 只有两支的 **全部条件参数、输入、mask 和随机状态相同且不更新**时，才可共享 KV。跨 step 缓存须有 checkpoint、样本、augmentation、noise/seed 等完整失效键；不要先保存巨大的 40 层 KV 离线缓存。

原 Video expert 不读取 action 的 mask 是条件缓存成立的前提，应在测试中验证。teacher 轨迹不允许用测试集生成训练标签，也不允许额外读取部署时不可见的未来 RGB、真实未来 pose 或几何标签作为输入。

### 5. 必须保留的实验矩阵与阶段验收

| ID | 方法 | 初始化与预算 | action NFE | 要回答的问题 |
|---|---|---|---:|---|
| B0 | 原 VPP2 FM | 原已验证 checkpoint | 10 | 当前仓库基线是否复现 |
| B1 | 同一 FM checkpoint 直接一步 Euler | 不训练 | 1 | 单纯少跑几步会损失多少 |
| B2 | FM 继续训练 | 与蒸馏 student 同起点、同额外 optimizer updates / trainability | 10；另记 1 | 排除额外训练带来的收益 |
| D0 | teacher 轨迹平均速度蒸馏 | 同 40 层；原 student trainability | 1 | 能否有效压缩到一步 |
| M0 | 原始 data-based MeanFlow | 同初始化与额外训练预算 | 1 | 是否需要 teacher 轨迹标签 |
| K0 | data-based K-MF，可学习分割 | 与 M0 同起点、预算、conditioning | 1 | 可学习分割是否改善一步结果 |
| K1 | data-based K-MF，固定中点 | 与 K0 相同 | 1 | 分割点学习的独立价值 |
| A* | DA / dispersive / 导数实现等独立消融 | 每次只改一项，记录确切 base objective | 1 | 哪种修改有效、成本如何 |

所有 D0/M0/K0/K1 都从同一原 FM checkpoint 出发，第一轮不串联 warm-start；如果后续从 D0 再 fine-tune K-MF，给它单独 ID，并配置等预算 continuation 对照。`distill + MF/KMF` 也是单独实验，不能默默叠加到 D0。

Q1 的建议预注册门槛（本项目建议，可在正式跑之前调整并固定）：

- 原 benchmark 的完整闭环评测与 audit 通过；不能用小样本 offline MSE 宣布一步成功。
- 1-NFE 方法相对 B0 的主要闭环指标差值，配对/分层 bootstrap 的 95% CI 下界不低于 `-2` 个百分点；如果 CI 跨过门槛，标为证据不足而非通过。RoboDojo 对官方等权 SR/score 按原 cell/dimension 结构重采样，不用 pooled SR。
- 同硬件、同精度、同执行模式和相同输出项，端到端 p50 至少降低 10%，并报告 p95、action-only 时延、prefill 时延及显存。10% 是工程验收目标，不是理论承诺。
- 原标准 seed 协议作为主报告。候选方法再做至少 3 个训练 seeds 的重复验证，单独报告波动；不要替换或挑选官方测试 seeds。为 OOD/PRO 单独报告，不能拿 Standard 成功代替泛化证据。
- 报告每任务/每维度退化、动作异常和接触失败。建议用同条件、同噪声比较 student 与 teacher endpoint 误差，同时记录示范 action 误差；二者都不能替代闭环指标。

若 D0 未通过，先排查配对条件、噪声、时间符号、端点权重、padding、checkpoint 和评测入口。随后比较 M0/K0 等方法；在选定的一步方案通过前不加入 3D heads。

### 6. 后续 MeanFlow 方法的实现规格

#### 6.1 原始 data-based MeanFlow（M0）

从训练示范 `a*` 与新噪声构造 `z_t`，预测完整区间平均速度 `u`；JVP 用于构造监督，不是另一个被回归的输出 head：

$$
D=\operatorname{JVP}_{(z_t,r,t)}[u_S;(\epsilon-a^*,0,1)],\quad
L_{\rm MF}=\operatorname{maskedMSE}\left(u_S,\operatorname{sg}[\epsilon-a^*-(t-r)D]\right).
$$

- wrapper 接收归一化 `(z,r,t)`，内部将时间转为 0–1000，使链式法则自然生效。不要把直接针对 0–1000 输入的时间切向量误设为 1。
- Video 条件在求生成时间导数时固定。零切向量不等于在正常主预测中切断应有的 student Video 参数梯度；RoboDojo 保留主预测分支的 joint 梯度。
- 首轮可采用显式命名 `diagonal_probability=0.5`，其余时间对从对应 benchmark 的原 shifted sigma 分布独立采样后排序。不同论文的 logit-normal/Beta 采样列为独立 sampler 消融，避免与目标改动混淆。
- `r=t` 时直接使用 FM target `v`，无需计算 JVP。K0/K1 使用相同 diagonal 比例；本项目首版 diagonal 样本只参与 policy loss，不参与 generator loss，generator loss 只在非 diagonal 样本上归约。这个选择须写入配置，不冒充论文默认；该批次没有非 diagonal 样本时跳过 generator 更新。
- target 整体 stop-gradient；使用单独有梯度的 `u_pred` 前向和 `no_grad` 下的 forward-mode JVP，避免不必要的高阶参数图。不要把含 `@torch.no_grad()` 的旧 inference helper 当作可训练预测函数。
- uniform masked loss 保留为历史对照。LIBERO fresh-init M0 的下一轮按下述规格增加 adaptive weighting，先只改变这一因素；CFG、dispersive 等仍是独立消融。不复制 MP1 未核对的 CFG 实现，也不把含义相反的 `flow_ratio` 变量直接搬来。

<a id="libero-mf-adaptive-loss"></a>

#### 6.1.1 已实现：LIBERO MeanFlow adaptive loss weighting

**状态：已实现 adaptive weighting；新的 LIBERO fresh-init MeanFlow 默认开启，验证与运行记录见 `docs/libero_one_step.md`。这不是收敛结论。** 本节覆盖旧计划中“先用 uniform”的下一轮实验建议；历史 uniform 运行仍按原配置解释。本轮验证的问题是：在相同初始化、数据、模型、采样与优化器设置下，降低异常 MF 残差的相对梯度贡献，是否改善原始残差、一步 action 误差和闭环表现。

**依据与边界。** [原始 MeanFlow §4.3 / 附录 B.2](https://arxiv.org/html/2505.13447v1#A2.SS2) 使用停止梯度的自适应权重，图像实验中 p=1 最好，p=0.5 也有竞争力。本项目先试 p=0.5、c=1e-3，再单独比较 p=1；这是适配当前按有效 action 元素取均值的起始配方，不声称复现 ImageNet 最优参数，也不保证解决发散。已有 [JVP 精度及完整训练步诊断](docs/libero_one_step.md#jvp-precision-diagnostic-2026-10-20-utc) 表明失稳 checkpoint 在 FP32 下仍有很大导数和 loss；该结果不能证明 BF16 从训练初期就没有影响。本轮保留 BF16，不同时改为 FP32。

官方 [JAX meanflow.py](https://github.com/Gsunshine/meanflow/blob/main/meanflow.py#L216-L227) 和作者 [PyTorch meanflow.py](https://github.com/Gsunshine/py-meanflow/blob/main/meanflow/models/meanflow.py#L52-L60) 都先按样本对平方残差求和，再除以停止梯度的 `(loss+c)^p`。本项目的加权公式与 detach 语义一致，但使用有效 action 元素均值、p=0.5、c=0.001；官方 JAX 默认是元素求和、p=1、c=0.01。归约尺度及超参数是这里明确记录的任务适配，不能声称与官方完整配方逐项相同。

**控制变量。** 继续使用 `configs/libero/train_meanflow.yaml` 的 fresh Action 初始化、冻结 Video-10k、40 层 hidden512 Action1B、可训练 proprio、global batch 256、AdamW、LR=1e-4、warmup=500、30k cosine schedule、clip=1、原数据与 mask。推理仍为一次 `epsilon - u(epsilon, 0, 1)`。不更换 JVP 切向量，不增加 teacher/iMF/K-MF/几何 head，也不改变 Direct Action MSE 或 FM baseline。

本节 adaptive 首轮实验的时间采样保持原实现：两次 shift=5 采样后排序，再令 50% 样本 r=t。**该轮不同时改变对角时间分布；下一项独立消融见 [6.1.2](#libero-mf-diagonal-time)。** 作者 [time_sampler.py](https://github.com/Gsunshine/py-meanflow/blob/main/meanflow/models/time_sampler.py) 的 v0 明确标为论文版本，采用相同的“先排序、再设对角”顺序；作者该实现的基础分布是 logit-normal，当前项目是 shifted uniform。两者基础分布不同，但排序顺序本身不是数学错误。

**精确 loss 定义。** 保留当前 z、v、JVP、target 的定义，仅改变最终 policy loss 的按样本权重：

$$
z_t=(1-t)a+t\epsilon,\quad v=\epsilon-a,\quad
y=\operatorname{sg}\left[v-(t-r)\operatorname{JVP}(u;(v,0,1))\right].
$$

令 m 为已有的有效 action 元素 mask，N_i 为样本 i 的有效元素数：

$$
e_i=\frac{\sum_{h,d}m_{i,h,d}(u_{i,h,d}-y_{i,h,d})^2}{\max(N_i,1)},\quad
w_i=\operatorname{sg}\left[(e_i+c)^{-p}\right].
$$

把原来的每样本误差 e_i 替换成 w_i e_i，再使用原来的 batch/distributed reduction。具体要求：

- 按样本加权；不是每个 action 维度分别加权，也不是用整个 batch 的平均 loss 生成一个公共权重。有效元素均值的尺度必须固定，不能静默换成 sum 后仍沿用同一个 c。
- residual、平方、有效元素归约、权重和最终 loss 运算使用 FP32；这不意味着把模型/JVP 前向改为 FP32。
- target 和权重均停止梯度，主预测保留参数梯度；不新增高阶梯度。对角与非对角样本都使用相同加权规则。
- 不把权重重新归一化到均值 1，不新增 weight clipping，不乘回 FM scheduler weight。这些都会产生额外配方变化。
- 复用 `action_loss_is_pad`（保留 terminal-token 规则），不要替换成原始 `action_is_pad`。
- 保留已有 reduction 语义：fresh LIBERO 的 `baseline_reduction=True` 对整个本地 batch 取均值，全 padding 样本贡献零；其他既有路径使用 `reduce_valid` 的全局有效样本数归约。不要在同一个实验中改变分母。全局没有有效元素时返回与预测图连接的零 loss。
- p=0 时权重为 1，前向 loss 和反向梯度应退化成同一路径的 uniform 行为；旧配置缺省仍解释为 uniform。

实现公式（复用当前 helper）：

```python
per_sample, has_valid = masked_per_sample(pred, target.detach(), is_pad)
if loss_weighting == "adaptive":
    weights = (per_sample.detach() + adaptive_c).pow(-adaptive_power)
else:
    weights = torch.ones_like(per_sample)
weighted = torch.where(has_valid, weights * per_sample, 0.0)
policy_loss = (
    weighted.mean()
    if baseline_reduction
    else reduce_valid(weighted, has_valid)
)
```

这里优化的梯度为 sum_i w_i * grad(e_i)（含对应归约因子），而不是对未 detach 的 e_i/(e_i+c)^p 求导。p=1 时记录的加权 loss 数值趋于饱和，但因为权重 detach，梯度并不等于该饱和函数的完整导数。**不能以加权 loss 变小或有界作为稳定性的证据。**

**实现位置与约束。**

| 文件 | 所需修改 |
|---|---|
| `src/vpp2/meanflow/objectives.py` | 为 `data_loss()` 增加兼容旧调用的 weighting 参数；在已构造并 detach 的 MF target 后、batch reduction 前应用上述按样本权重。增加 raw/weighted 诊断。uniform 路径保持等价；首轮只开放 `data_meanflow`。 |
| `src/vpp2/meanflow/model.py` | 将 `meanflow.loss_weighting` 与 `meanflow.adaptive_weighting.{power,c}` 传给 `data_loss()`；不改变条件构建、Video cache、训练 mask、JVP 或一步推理。沿用现有 config/checkpoint metadata 存储完整 loss 设置。 |
| `src/vpp2/meanflow/cli.py` | 将“仅允许 uniform”的校验改成显式允许 uniform/adaptive；adaptive 首轮仅接受 LIBERO + data_meanflow + baseline_init。检查 power/c 有限、0<=power<=1、c>0，拒绝拼写错误和未支持组合。`--dry-run` 打印模式、power/c 和初始化来源。 |
| `configs/libero/train_meanflow.yaml` | 按用户最新要求，将新训练默认设为 `loss_weighting: adaptive`，power=0.5、c=0.001；旧配置不含这些字段时仍按 uniform 加载。显式设置 uniform 可复现历史配方。 |
| `checks/test_meanflow.py`、`checks/test_libero_one_step.py`（或新的专门测试文件） | 增加权重/梯度、mask、配置和 metadata 的针对性测试，标准见下。 |
| `docs/libero_one_step.md` | 实现完成后补充新实验命令与实际验证结果；原有 uniform 记录标为历史配方，不能改写成 adaptive 已运行。完成后同步更新本节状态。 |

新增配置字段的约定：

```yaml
meanflow:
  loss_weighting: adaptive  # 新训练默认；旧配置缺省仍按 uniform 解释
  adaptive_weighting:
    power: 0.5
    c: 0.001
```

**日志要求。** `loss_policy` / `loss_action` 继续记录实际参与优化的加权 loss；同时新增 `mf/raw_mse`、`mf/weighted_mse`、`mf/raw_mse_diagonal`、`mf/raw_mse_offdiagonal`、`mf/weight_mean`、`mf/weight_max`、有效样本数和两组样本数。所有统计排除无有效元素的样本，并注明与训练 reduction 分母的区别。分布式均值用总和/计数归约；空组记录 count=0，不把它解读成零误差。各 rank 无论有没有非对角样本，都必须以相同顺序进入 collectives，避免挂起。保留 action L1、原 JVP/correction norm、preclip grad norm、时间和显存日志；建议增加有效元素上的 correction RMS，避免仅凭未归一化 norm 比较不同 batch。

本轮无需每步保存大张量或运行额外 action 前向。为判定性能，使用现有一步 action 诊断，并在固定的小规模 LIBERO-PRO episode 子集上比较 checkpoint；预先固定任务、初始化、seed、评测预算，最终再运行完整协议。固定训练样本 L1 只能作拟合诊断，不能替代闭环成功率。

**启动示例（adaptive 现为默认；下面显式写出参数便于审计）。** 使用新 RUN_NAME/output_dir，按现有生产入口在每个分配节点执行。首先完成下述 dry-run 和短跑，再启动正式预算：

```bash
# 从仓库根目录；环境与多节点分配方式沿用现有 LIBERO 指南。
source scripts/libero/activate_env.sh
export RUN_NAME="libero-mf-adaptive-p05-${TRAIN_JOB_ID:?}-round${RUNNING_ROUND:-0}"
bash scripts/libero/train_meanflow.sh --dry-run \
  meanflow.loss_weighting=adaptive \
  meanflow.adaptive_weighting.power=0.5 \
  meanflow.adaptive_weighting.c=0.001

# 实现与短跑验收通过后，去掉 --dry-run：
bash scripts/libero/train_meanflow.sh \
  meanflow.loss_weighting=adaptive \
  meanflow.adaptive_weighting.power=0.5 \
  meanflow.adaptive_weighting.c=0.001
```

**实验顺序与验收。**

1. 单元测试：p=0 与旧 uniform 的值/梯度一致；简单张量的梯度符合 detach 权重公式且非零；改变 padding 位置的数据不改变有效 loss/梯度；混合/全 padding batch 和 r=t 情况正常；非法配置被拒绝。
2. 用至少两 rank 的小测试覆盖某 rank 无有效样本、某 rank 无非对角样本的情况，检查 loss/梯度归约与固定 reduction 的单进程参考一致，不挂起。验证保存配置、compact checkpoint metadata、训练恢复能保留 weighting/p/c。旧 checkpoint/config 仍以 uniform 解释；恢复时校验 loss 设置，拒绝静默更换。
3. 实际 LIBERO 数据短跑，确认冻结 Video、Action/proprio 梯度、一步采样、保存/恢复和日志都正常。这只是工程验收，不宣称收敛。
4. 从与 uniform 对照相同的 fresh Action/proprio 初始状态启动 A1：adaptive p=0.5、c=1e-3、LR=1e-4；保持 seed 和原 30k schedule。先观察 1–2k 更新；用独立停止条件/诊断 wrapper 提前结束，不能仅把 max_steps 改成 2000 而缩短 cosine schedule。需要短跑末 checkpoint 时使用独立诊断保存频率，记录该差异。
5. 比较 raw 对角/非对角误差、correction RMS、裁剪前梯度及固定 seed 一步 action L1。加权 loss 下降但 raw loss/JVP 持续膨胀、action 变差，不能判为修复成功。确认稳定趋势后再扩展训练和闭环评测。
6. A2 单独比较 p=1（其他同 A1）；如仍不稳定，A3 仅将 A1 的 LR 降至 3e-5。每组用新的输出目录和相同初始状态，不从已失稳 checkpoint/optimizer 继续后声称与 fresh 对照等价。旧失稳 checkpoint 仅用于固定输入诊断。

用户已授权在验证后启动新的 16-H200 adaptive MeanFlow 训练；使用独立输出目录，不自动停止其他任务。代码、测试、resolved config 和短跑记录一并保存。正式训练沿用 30k 预算，不增加自动早停；1–2k 更新用于观察趋势，不能凭短跑宣称收敛。

<a id="libero-mf-diagonal-time"></a>

#### 6.1.2 独立消融：对角 FM 样本单独采时间（diagonal_fix）

**状态（2026-10-23）：已实现独立的 `train_meanflow_diagonal_fix.sh` / `.yaml`；原 `train_meanflow.sh` 仍保留 sorted_max。启动与验证记录见 [one-step 实验说明](docs/libero_one_step.md#oct23-ablations)。这不是新实验的收敛或闭环成功率结论。** 保留当前稳定 adaptive 配方，下一轮只改变对角样本的时间分布。旧的 p=1 / 降 LR 建议降为后续可选消融，不作为必须立即执行的修改。

**为什么改。** 当前 `src/vpp2/meanflow/objectives.py::sample_intervals()` 先对两个独立时间做 shift，再取 `t=max(s1,s2), r=min(s1,s2)`，最后把对角行设成 `r=t`。对角处 `t-r=0`，MF target 的确退化为 FM target `epsilon-action`，但其时间分布仍然是两个 shifted 时间的最大值。这里没有漏掉 shift，也没有改变 FM target 的正确性；它偏离的是“对角时间服从单次 shifted uniform”的实验设计意图。新方案不是声称修复原始 MeanFlow 的数学错误，也不保证效果更好。

采用当前 `clean=0, noise=1` 约定，令 `s(u)=k*u/(1+(k-1)*u)`，其中 `u~Uniform(0,1)`。单次 shifted 时间的 CDF 是 `F(x)=x/(k-(k-1)*x)`，历史对角时间的 CDF 是 `F(x)^2`。在 `k=5` 时，两者中位数分别约为 0.8333 和 0.9235。因此，当前对角 FM 监督额外偏向高噪声；新方案只撤销对角分支中的 max 偏置，非对角区间仍保留排序。adaptive weighting 继续作用于两类样本，所以这里只规定采样分布，不声称两类样本的有效梯度占比各为 50%。

**配置契约（已实现）。**

```yaml
meanflow:
  interval:
    source: shifted_uniform
    diagonal_probability: 0.5
    diagonal_time_sampling: independent  # 新实验；历史模式为 sorted_max
  loss_weighting: adaptive
  adaptive_weighting:
    power: 0.5
    c: 0.001
```

- `sorted_max`：保留历史算法；旧配置或旧 checkpoint 缺少新字段时解释为此模式。不得把旧实验自动重标为 independent。
- `independent`：对角 mask 独立于采样时间；每个被选中的样本用一次新的 shifted uniform 时间，同时赋给 r 和 t。仍使用 `self.train_action_scheduler.shift`（当前 LIBERO 为 5），不新增一个可与训练 scheduler 冲突的 shift 参数。
- 首轮仅在 LIBERO `data_meanflow` 路径启用新模式；teacher trajectory、Direct Action、K-MF、RoboDojo 保留现有行为。未支持组合应明确报错，不能静默忽略新选项。
- 对照配置显式设 `sorted_max`，新实验配置显式设 `independent`，使用不同输出目录。新实验参数必须写入 resolved config、dry-run 输出和 checkpoint metadata。

**采样参考伪代码。** 旧模式保留原随机数调用顺序；新模式在生成相同的初始时间对和 mask 后才额外采时间。不要再次对 `td` 与原 `t` 取 max，也不要只覆盖 r 而留下旧 t。

```python
u = torch.rand(2, batch, device=device)
s = shift * u / (1 + (shift - 1) * u)
t, r = s.max(0).values, s.min(0).values
diagonal = torch.rand(batch, device=device) < diagonal_probability

if diagonal_time_sampling == "sorted_max":
    return t, torch.where(diagonal, t, r)
if diagonal_time_sampling != "independent":
    raise ValueError("Unknown diagonal_time_sampling")

ud = torch.rand(batch, device=device)
td = shift * ud / (1 + (shift - 1) * ud)
t = torch.where(diagonal, td, t)
r = torch.where(diagonal, td, r)
return t, r
```

新模式多消耗一次随机数采样，不能宣称相同 seed 会使后续每个 batch 的所有噪声逐位相同。单元测试可通过受控随机输入验证只覆盖对角行；公平训练比较保持 seed、初始化、数据划分、预算和评测噪声协议相同，并对最终候选重复 seeds。`diagonal_probability` 是逐样本 Bernoulli 概率，不要求每个 batch 恰好有一半对角样本。

**修改位置与边界。**

| 文件 | 实现契约 |
|---|---|
| `src/vpp2/meanflow/objectives.py` | 给 `sample_intervals()` 增加缺省为 `sorted_max` 的兼容参数，校验模式、有限正 shift 和 [0,1] 概率，按上述规则实现。保留 `data_loss()`、target detach、JVP 切向量、padding 和 adaptive 公式；现有对角行跳过 JVP 的逻辑不变。 |
| `src/vpp2/meanflow/model.py` | 从 interval 配置取模式并传入 sampler。保存完整配置；扩展 compact checkpoint 加载校验，避免静默改变 sampler。 |
| `src/vpp2/meanflow/cli.py` | 规范化历史缺省值，拒绝拼写错误/未支持组合，dry-run 显示 source、模式、比例、实际 train shift。不要仅接受 YAML 字段却未传到 sampler。 |
| 配置校验 helper 与 `src/vpp2/trainer.py` | 复用同一 sampler 设置解析/比较逻辑，在完整 optimizer/scheduler resume 和 compact checkpoint 加载两条路径验证 sampler 契约。旧 metadata 缺省解释为 `sorted_max`，保留现有 loss 设置校验。 |
| `configs/libero/train_meanflow.yaml` 与独立实验配置 | 原稳定配方显式保留 `sorted_max`，另建如 `configs/libero/train_meanflow_diagonal_fix.yaml` 继承该配置，仅覆盖模式和输出目录。两份都保留 adaptive。 |
| `checks/test_meanflow.py` 或专门 sampler 测试 | 验证旧模式等价、新模式分布、分支边界、JVP 跳过、配置传递、两种 checkpoint 恢复路径。 |
| `docs/libero_one_step.md` | 实现后补充可运行命令、实际测试与短跑记录，再更新本节状态；不能预先宣称已实现或性能提升。 |

**恢复与继续训练。** 普通 resume 必须保持 sampler 模式、比例、source 与实际 shift 一致；缺失的历史信息应按可核实的旧配置解析，不能猜测。切换模式是新实验：主对照从相同 fresh Action/proprio 初始化开始。若为节省迭代成本，从已有稳定 adaptive checkpoint 分出两支，必须同时做旧模式 continuation 对照，匹配恢复的 optimizer/scheduler/训练预算，并明确记录父 checkpoint 和切换 step；这是 continuation 实验，不能当成从头训练的同等证据。不得借普通 resume 静默更改模式。

**验收与观测。**

1. 受控随机数测试：`sorted_max` 与原实现逐项相同；`independent` 的非对角行仍为原 max/min，对角行的 r、t 都等于独立 td。覆盖 probability=0/1/0.5、shift=1/5，确保有限且 `0<=r<=t<=1`；非法模式和非有限参数报错。
2. 固定种子的大样本统计检查：probability=1 时，新旧经验 CDF 分别接近 F 与 F²，用足够样本和预设统计容差，不能要求有限样本严格相等。probability=0 时，两种模式的区间分布一致；不使用“无 max”等价于“均匀 t”这种错误断言。
3. Toy loss 测试：全部对角时 derivative 回调不被调用；对给定 td，两模式的 loss 都等于同一 adaptive masked FM 公式，预测分支梯度非零。两 rank smoke 覆盖某 rank 无对角/非对角行，日志归约不挂起。
4. 增加按对角/非对角分别记录的 t 分桶计数、均值和 gap 分桶；分布式使用全局 sum/count 或直方图，空桶保留 count=0。保留 raw/weighted MSE、有效样本数、权重、preclip grad norm。现有 correction RMS 混入对角零值；另记非对角有效元素 correction RMS，避免以后改变对角比例时产生假改善。
5. 在相同固定验证集、相同噪声 seed 下比较一步 `epsilon-u(epsilon,0,1)` 的 action L1/MSE，并分 action 维度报告。采样改变后，训练 loss 的平均对象也变了，不能直接用训练 loss 更低判断获益；可增加一套两模式共用的固定 (r,t) 诊断集。训练样本上的固定 L1 不能替代 held-out 验证。
6. 按原 30k schedule 做 1–2k 更新诊断，使用独立停止机制保留原 schedule；匹配 checkpoint step 做同一 LIBERO 闭环子集的早期筛选。最终报告完整 LIBERO 与 LIBERO-PRO、每任务结果、统一评测 seeds 与时延；小子集的好结果不能替代完整评测。

**在 adaptive 已稳定之后，后续优先级。**

| 顺序 | 单独比较什么 | 什么时候值得做 |
|---|---|---|
| D0 / D1（现在） | adaptive + sorted_max 对比 adaptive + independent；其余完全相同 | 验证去掉对角 max 偏置是否改善一步动作和闭环，而非继续证明 loss 能下降。保持当前稳定的 LR 与 p/c。 |
| D2（条件性） | 在选定模式下，diagonal_probability=0.25/0.5/0.75 | 如果对角局部 FM 学得好、但大 gap/一步输出弱，再检查局部与区间监督的分配。比例也改变 JVP 成本，同时报告相同更新预算及训练耗时；这些比例是项目消融，不冒充论文默认。 |
| D3（条件性） | 在选定 sampler 下，单独比较 shift=5 与较小 shift（如 1） | 只在时间分桶/固定验证显示噪声区域覆盖失衡时做；shift=1 不等于复现原论文的 logit-normal 配方。不要与 D1 同时改变非对角时间分布。 |
| D4（下一轮） | 固定 sampler 和 adaptive，比较原 MF、iMF boundary、iMF 辅助 v-head | 先按[两种 iMF 实现规格](#libero-imf)检验预测速度切向量与额外容量的收益；尚需实现，不可只改 YAML 启动。 |
| D5（之后） | 单独测试 K-MF 或 teacher 蒸馏 | 一步闭环质量仍不足时再考虑；adaptive 迁移到 K-MF 需单独设计与校验，不能直接打开配置。 |

低学习率、p=1、FP32 JVP 都不是当前必须修改的项目；若 raw 残差、JVP、梯度再次持续异常，再依据诊断单独测试。不要用“加了 adaptive 后训练正常”推断一步性能已等于 10-step FM。先完成稳定 MF 的动作与闭环对照，再进入几何 heads。

<a id="libero-imf"></a>

#### 6.1.3 下一步：iMF boundary 与辅助 v-head 两种方案（实现规格，尚未实现）

**状态与目的（2026-10-29）。** 已实现 `data_improved_meanflow` 的 boundary / auxiliary 两个版本；数学、梯度、checkpoint 与双进程同步测试通过。本次 79 项测试通过，另通过双进程不均匀 diagonal/padding 同步，以及两个版本的小型 40 层模型 BF16 + ZeRO-1 训练/完整状态续训。真实规模 H200 验证和收敛收益仍待排队任务验证，不能把实现通过测试等同于效果提升。用户反馈 adaptive 后训练恢复正常；下一步要验证的是：在相同数据、初始化和一步推理预算下，改用模型预测的瞬时速度作为 JVP 方向，是否改善 held-out action 和 LIBERO / LIBERO-PRO 闭环表现。先完成这两种 iMF 对照，再考虑几何 heads。

**当前启动方法。** 在启智每个节点执行同一条命令；自动读取 `PET_*` 拓扑。16 GPU 时每卡 batch=16、global=256、累积=1；保留 ZeRO-1、BF16、LR=1e-4、warmup=500、cosine/30k、每 5k 保存、每 200 step action diagnostic、4 workers、首帧解码及 uint8 路径、离线 W&B。初始化仍为 fresh Action + 冻结 Video-10k，**不续训旧 MF-30k**。

```bash
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero/activate_env.sh  # 激活 ../miniconda/envs/vpp2
export WANDB_PROJECT=VPP2-LIBERO
# 复用边界速度：
bash scripts/libero/train_imf_boundary.sh
# 辅助 v-head（作为另一个独立任务执行）：
bash scripts/libero/train_imf_auxiliary.sh
```

排队任务会先在所获分布式资源上执行 `scripts/libero_debug/smoke_imf.sh imf_boundary` 或 `imf_auxiliary`：10 个真实更新、每卡 batch 跟随正式配置且最多为 16（32 卡时为 8）、30k scheduler，检查分支梯度、compact reload、一次 action forward 推理及 rank-0 evaluator；全部通过才启动对应正式脚本。短跑使用独立 `*-smoke` 目录，不把短跑参数或 optimizer 带入正式实验。真实规模 smoke 未通过时脚本立即失败，不自动降 batch、改精度或改 loss。该 10-step 检查不代表已完成下文建议的 1–2k 收敛诊断，后者须在训练进展后检查。

```bash
# 不占 GPU 的公式、模型、恢复和两节点启动参数检查
source scripts/libero/activate_env.sh
OMP_NUM_THREADS=2 python -m pytest -q checks/test_improved_meanflow.py
OMP_NUM_THREADS=2 python -m torch.distributed.run --standalone --nproc_per_node=2 checks/ddp_improved_meanflow.py
# 仅解析实际节点拓扑/超参数，不加载权重
bash scripts/libero/train_imf_auxiliary.sh --dry-run
```

训练同时记录主目标原始/加权误差 `mf/*`、辅助目标 `imf/v/*`、`imf/vhat_mse` / `imf/vhat_rms` 以及 `imf/local_pre_reduce_grad_norm_{shared,u_tail,v_tail}`。最后一组是各 rank 本地、归约前的梯度范数，再对 ranks 求平均，不能当作全局 clip norm。辅助版总 loss 包含两个目标，跨方法仍比较相同 action diagnostic/L1 和闭环 SR。辅助版增加训练参数和计算，属于相同主干/推理预算的容量实验，不是参数量完全相同的实验。

**来源与范围。** [iMF 论文 §4.1、Algorithm 1、Table 1(a)、Appendix A](https://arxiv.org/html/2512.02012) 提出了两种瞬时速度估计：复用边界预测 `u(z,t,t)`，或添加训练时辅助 v-head；两者都在论文中做了比较。论文表中无 CFG 的 boundary / auxiliary FID 分别为 29.42 / 30.76，固定 CFG 下为 5.97 / 5.68，不能据此认定辅助 head 必然更好，更不能直接推出 action policy 的收益。Figure 3 的稳定性曲线没有 adaptive weighting，不能当成对本仓库“adaptive 已稳定 MF”的直接比较。实现参考：[官方目标函数](https://github.com/Lyy-iiis/imeanflow/blob/main/imf.py)、[官方网络](https://github.com/Lyy-iiis/imeanflow/blob/main/models/imfDiT.py)。下面的 40 层、32+8 分支、padding、Video KV 接入和训练配方是本项目适配，不是论文的原样复现。

**A. 共同数学契约：改的是 JVP 方向，推理仍预测平均速度。**

时间沿用 clean=0、noise=1。令条件 C 包括固定的 Video KV、文本和 proprio，a 为归一化干净动作，epsilon 为噪声；每个 batch 的所有 action 分支共用同一份条件：

$$
v_c=\epsilon-a,\qquad z_t=(1-t)a+t\epsilon,\qquad 0\le r\le t\le1.
$$

先得到下面两种方案之一的瞬时速度估计 $\hat v_\theta(z_t,t,C)$，再计算主 u 分支的导数：

$$
D_\theta=\operatorname{JVP}_{(z,r,t)}
 u_\theta(z_t,r,t,C)
 [\operatorname{sg}(\hat v_\theta),0,1],\qquad
V_\theta=u_\theta(z_t,r,t,C)+(t-r)\operatorname{sg}(D_\theta).
$$

主损失是 V 对样本速度 vc 的回归。等价实现可以保留当前 target 形式：

$$
y_u=v_c-(t-r)\operatorname{sg}(D_\theta),\qquad
E_{u,i}=\frac{\sum_j m_{ij}(u_{ij}-y_{u,ij})^2}{\max(1,\sum_j m_{ij})},\qquad
w_{u,i}=(\operatorname{sg}(E_{u,i})+c_w)^{-p},\qquad
L_u=\mathcal R\big(\mathbf1_{\mathrm{valid},i}\,w_{u,i}E_{u,i}\big).
$$

m 只覆盖有效 action 元素；每样本先做 masked mean，再求权重；本轮保持 **p=0.5、c_w=0.001**。这里的 $\mathcal R$ 必须沿用基线当前 reduction：fresh `baseline_init` 的 `baseline_reduction=True` 为全 batch mean（全 padding 行置零），其余路径沿用现有 distributed valid reduction。不要在迁移 iMF 时顺便修改归约分母。上式写成 target 或 V-loss 给出的残差相同；**仅把 JVP 移到等号另一侧仍是原 MF，必须把切向量 vc 换成预测的 vhat 才是本轮 iMF 改动。**

JVP 中固定 r 和 C，只沿 `(vhat,0,1)` 对 z、r、t 求方向导数；vhat 是已求值的切向量，不能把其计算图嵌入 JVP 再求导。对 D 的结果 stop-gradient，主 u 输出保留梯度，不引入高阶参数导数。继续使用归一化时间；现有 `predict_action_velocity` 内部把 r/t 乘 1000，JVP 应对这个完整 wrapper 求导，不能再手动乘/除 1000。只对非对角行计算 JVP，对角行直接令 D=0。

**B. I1：boundary，复用模型的对角预测（优先实现）。**

$$
\hat v_\theta(z_t,t,C)=u_\theta(z_t,t,t,C),\qquad L_{I1}=L_u.
$$

1. 保持原 40 层 Action DiT、输出 u-head、end-time embedding 和 Video KV 交互结构。主预测使用 `(z,r,t,C)` 并保留梯度。
2. 对非对角行，在固定相同条件下额外计算 `u(z,t,t,C)`，以 no-grad 得到 vhat，再用于主 u 函数的 JVP。这次对角求值**不是**重新采样一个对角训练样本，不改变原来的 r/t 或 batch 对角比例。
3. 训练 batch 中原本抽到的 r=t 行，其主损失自然退化为 adaptive FM，训练同一个边界预测器。本版本不再为额外对角求值添加一份 FM loss，避免改变监督量。非对角行的 boundary 求值只提供停止梯度的方向。
4. 无新增可训练参数；“两次调用”指额外边界求值与主分支求值的概念划分，不是精确训练 FLOPs。当前实现把带梯度主预测与无梯度 JVP 求值分开，因此还有 JVP 的 primal 计算。报告实测 step time、显存，不把 I1 称为两次普通 forward 的固定成本。

**C. I2：辅助 v-head，为瞬时速度分配额外容量。**

本项目第一版采用 **共享参数的前 32 blocks + 两条各 8 blocks 的尾部**，原有 u 路径仍为 40 层：

```text
u 路径：(z,r,t,C) → 原 blocks[0:32] → 原 blocks[32:40] → 原 head → u
v 路径：(z,t,t,C) → 同一 blocks[0:32] → 独立 aux_v_blocks[0:8] → aux_v_head → vhat
```

- 新增独立注册的 `aux_v_blocks`（8 层）与 `aux_v_head`，不要把它们 append 到 `action_expert.blocks`；后者始终长 40，避免破坏 MoT 的 video/action 层数匹配。首版在 baseline 权重正确加载后，deep-copy 原尾部 8 blocks 和输出 Linear 初始化辅助尾部；不得共享尾部 Parameter 对象。额外初始化不得改变公共主干、proprio 或后续训练 RNG 状态。
- 共享 action/text/time 输入模块与前 32 层的**参数**，但首版不复用两个条件不同的 prefix **激活**。现有 `pre_dit` 把 r 的 embedding 注入所有 blocks，因此 `(z,r,t)` 的 prefix 不能冒充 `(z,t,t)` 的 prefix。v 路径显式使用 r=t，以保证 vhat 不依赖此次抽取的 r。这是保守、易验证的项目设计；如果以后要一次共享 prefix forward，须另行设计仅依赖 t 的 prefix，并作为结构消融，不能悄悄修改 I2。
- 两条路径复用 `prepare_condition()` 生成的同一份冻结 Video cache，不重新跑 Video。辅助 block j 使用原 action 层 `32+j` 对应的 Video KV；保留 `_video_source_layer` 的映射和已有 attention mask、RoPE、context 规则，不新增 Video 层。启用 `use_video_tokens` 时，首版共享原层 `32+j` 的 token/KV projection 参数并正确复用其逻辑；主对照保持该开关和其他 conditioning 开关与基线一致。
- v 路径对全 batch 计算一次，保留梯度以接受独立 FM 监督；将同一输出的 detach 版本作为非对角 JVP 的方向。JVP 只作用于主 u 路径，不同时对辅助输出求 JVP。两分支共享参数的梯度正常相加。

$$
E_{v,i}=\operatorname{maskedMean}_{j}\!\left[(\hat v_{ij}-v_{c,ij})^2\right],\qquad
w_{v,i}=(\operatorname{sg}(E_{v,i})+c_w)^{-p},\qquad
L_v=\mathcal R(\mathbf1_{\mathrm{valid},i}\,w_{v,i}E_{v,i}),\qquad
L_{I2}=L_u+\lambda_vL_v,\quad\lambda_v=1.
$$

两项**分别**计算自适应权重，再相加；不能用 `(E_u+E_v)` 计算共同权重，也不能用主分支的权重给辅助项加权。Lv 监督所有有效样本，包括对角和非对角，不仅是 r=t 行。不能省略 Lv：停止梯度的 JVP 本身不会给辅助尾部提供学习信号。这里继续使用本仓库 masked mean 和 p/c，避免同时改变 loss 尺度。新增尾部使训练参数增加，且本方案需重新执行不同时间条件的 prefix，**不预设训练比 I1 更快**。

**D. 两者的一步推理完全相同。**

$$
\hat a=\epsilon-u_\theta(\epsilon,0,1,C).
$$

推理只执行原 40 层 u 路径，`action_sampler: meanflow`、`action_nfe: 1`；不执行边界求值、JVP、辅助 v 分支或 FM solver。这是 action 的 1-NFE，Video prefill 仍存在，须同时报告 action-only 与端到端时延。保持原动作归一化/反归一化、chunk、执行步数和控制频率。

**E. 配置与代码修改契约（以下字段和文件名均为待实现）。**

建议新增 `configs/libero/train_imf_boundary.yaml`，继承现有 `train_meanflow.yaml`；YAML 必须按现有 loader 深合并，保留初始化、interval 和 adaptive 配置：

```yaml
_base_: train_meanflow.yaml
output_dir: runs/libero/imf_boundary_fresh
meanflow:
  objective: data_improved_meanflow
  imf:
    variant: boundary
```

建议新增 `train_imf_auxiliary.yaml`，同样继承 `train_meanflow.yaml`，覆盖独立输出目录及：

```yaml
_base_: train_meanflow.yaml
output_dir: runs/libero/imf_auxiliary_fresh
meanflow:
  objective: data_improved_meanflow
  imf:
    variant: auxiliary
    auxiliary_depth: 8
    auxiliary_loss_weight: 1.0
```

本轮明确以已完成的 **MF + adaptive + diagonal-fix（D1）** 为对照，MF / I1 / I2 全部使用 `shifted_uniform`、`independent`、diagonal_probability=0.5、实际 action train_shift=5；I1/I2 配置直接继承 `train_meanflow_diagonal_fix.yaml`。不把早期 MF/sorted_max 与本轮 iMF/independent 的差异归因于 iMF。维持现有 LR=1e-4、warmup=500、30k schedule、global batch=256、BF16、JVP backend 和相同 Action/proprio 初始化及冻结 Video-10k；不同时引入 CFG、K-MF、几何、JVP clipping、FP32 JVP 或改 LR。硬件不足时可调整 microbatch/累积但匹配 global batch，并记录差异。

| 修改位置 | 必须实现和检查的行为 |
|---|---|
| `src/vpp2/meanflow/objectives.py` | 新增独立 iMF loss 或明确 objective 分支，接收 predicted-v 回调；沿用 masking、adaptive 和 reduction。I1/I2 共享主目标，I2 增加独立加权 Lv。旧 MF/K-MF 数学路径不变，不能只放开枚举却仍传 vc 给 JVP。 |
| `src/vpp2/meanflow/model.py` | 配置 variant；每 batch 一次 prepare_condition；沿用 condition slicing 与 derivative_mode。边界求值和 JVP 使用正确时间条件；辅助预测保留训练图，JVP 内仅用 detach 方向。derivative_mode 退出后必须恢复 checkpointing 等开关。 |
| `src/vpp2/models/wan21_14b/action_dit.py`、`mot.py` | I1 不改网络。I2 单独注册辅助尾部、输出 head，抽取/复用层范围执行逻辑，保留原 40 层主路径及 KV 层索引。构建辅助模块必须早于 optimizer / DDP / ZeRO 包装，并纳入参数组；检查共享模块不被重复计入 optimizer。 |
| `src/vpp2/meanflow/cli.py`、`experiment_config.py`、`loss_config.py` | 显式接入新 objective，允许 LIBERO 对应初始化下的 adaptive 和两种已支持的 interval 模式；拒绝未知 variant、非法 depth/权重和未支持组合。首版 auxiliary_depth 固定为 8。dry-run 展示解析后的 objective、variant、p/c、sampler、实际 shift、主/辅助层数、训练参数量及推理 NFE。不能误放开其他未验证 objectives。 |
| `src/vpp2/trainer.py`、`meanflow/model.py` checkpoint 路径、`src/vpp2/libero/action_model.py` | 完整与 compact checkpoint 都记录 objective、variant、辅助结构/初始化、lambda_v、loss、sampler、父初始化及代码版本；加载时先按 metadata 建模块再 strict load。普通 resume 禁止 MF↔iMF、I1↔I2 或 lambda_v/结构静默切换；历史 checkpoint 无 iMF 字段仍是原 MF。若另做去辅助参数的部署导出，须显式 artifact 类型和允许键列表，不可用全局 strict=False 掩盖缺失。 |
| `configs/libero/`、`scripts/libero/`、`docs/libero_one_step.md` | 增加独立 I1/I2 配置和启动入口，使用独立目录。实现并验证后再补可执行命令和记录、更新本节状态；本次文档提交不提供声称已可运行的命令。 |
| `checks/test_meanflow.py` 及对应模型/恢复测试 | 覆盖下面验收项，不需要为纯文档提交运行 GPU 训练。 |

**F. 验收、诊断与实验顺序。**

1. **数学与梯度。** 用解析 toy 函数 `u(z,r,t)=Az+br+dt` 验证 JVP 为 `A*vhat+d`，r 切向量为 0，并覆盖内部时间缩放。替换 vhat=vc 时主残差应回到旧 MF；r=t 时严格退化为 adaptive masked FM，完全跳过 JVP。确认 u 有梯度，D/权重不反传高阶梯度；I2 的辅助尾部由 Lv 获得非零梯度，关闭 Lv 后不应通过 detached correction 获得梯度。
2. **结构与兼容性。** 启用辅助头前后，初始主 u 输出在固定输入下相同；aux 权重是数值拷贝、独立存储。检查两条路径都按原 40 个层索引取条件，Video 只 prefill 一次；改变 sampled r 不应改变同一 `(z,t,C)` 的 vhat。实际推理 hook 确认只调用 40 层 u 路径且辅助路径调用数为零。
3. **padding、分布式与恢复。** 覆盖全部对角/全部非对角、局部 rank 无非对角、全 padding、不同有效 horizon，确保无 NaN 或 collective 挂起。所有 rank 的归约调用顺序一致；rank-zero eval 不进入训练 all_reduce。完整 optimizer/scheduler 恢复与 compact save/load 都 round-trip 校验，错误 variant/结构/sampler 必须报错；旧配置和旧 MF 输出仍兼容。
4. **分开记录损失。** 保留主目标 raw/weighted MSE、权重分布、对角/非对角分组、非对角 correction RMS、t/gap 分桶、preclip grad norm。I2 额外记录 Lv raw/weighted MSE、vhat RMS、共享 prefix / u-tail / v-tail 梯度；I1 可记录 detached boundary FM 误差用于诊断但不加入训练 loss。I2 总 loss 多一项，不可直接与 MF/I1 的总 loss 数字比较。
5. **先小验证，再匹配训练。** 先 math/CPU 测试和两 rank 短跑、恢复、一步推理 smoke；再按原 30k schedule 运行 1–2k 更新诊断（通过停止机制结束，不把 scheduler 总步数缩成 2k）。MF、I1、I2 从相同初始化开始，保存相同步数 checkpoint。现有 MF run 若 sampler/预算不匹配，需补匹配对照，不能直接拼表。
6. **质量与成本分别报告。** 固定 held-out 样本和评测噪声 seed，报告一步 action L1/MSE、各动作维度和统一闭环子集；候选按相同协议完成 LIBERO、LIBERO-PRO、每任务成功率及多 seeds。记录训练更新数、有效 batch、耗时、显存、参数量、action-only/端到端推理延迟。I2 增加训练容量，必要时补等 GPU-hours 对照；一步 action 误差和训练 loss 均不能替代闭环成功率。

**最终要回答的三个问题：** I1 是否在现有稳定 adaptive MF 上带来一步闭环收益；I2 的额外瞬时速度建模容量是否在相同推理成本下进一步获益；这种收益是否值得其训练成本，并能在 LIBERO-PRO 泛化中保持。若两种 iMF 都无收益，应保留已稳定 MF，并据诊断决定下一项实验，不预设 iMF 必然胜出。

#### 6.2 K-MF（K0 主方案，K1 中点对照）

保持 M0 的数据、主干和 sampler，改变导数 target：

$$
\lambda=G_\phi(t,r),\quad c=r+\lambda(t-r),\quad z_c=(1-c)a^*+c\epsilon,
$$

$$
\bar D_1=\operatorname{sg}[D(z_t,c,t)],\quad
\bar D_2=\operatorname{sg}[D(z_c,r,c)],
$$

$$
R_\phi=v-(t-r)\left((1-\lambda)^2\bar D_1+\lambda^2\bar D_2\right),\quad
L_{\rm policy}=\operatorname{maskedMSE}(u_{\rm pred},\operatorname{sg}[R_\phi]).
$$

两状态使用同一 `(a*,epsilon)`；两个局部 JVP 各自固定下端点，切向量均为 `(v,0,1)`。中点版本固定 `lambda=0.5`。可学习版本按论文使用仅输入 `(t,r)` 的 `2→128→128→1` MLP、SiLU 和末端 sigmoid；它不读取场景，不代表物理动作阶段选择。

可学习版另实现 generator loss：令 `d_full=stopgrad(D(z_t,r,t))`、`d_split=(1-lambda)^2*D1_bar+lambda^2*D2_bar`，按论文的两种导数估计之差构造 `omega=1+gamma*exp(-1/(||d_split-d_full||+1e-6))`，再计算：

$$
L_{\rm gen}=\operatorname{mean}_{\rm non\text{-}diagonal}\!\left[
\omega_\phi\,\operatorname{maskedMSE}_{\rm sample}
(\operatorname{sg}[u_{\rm pred}],R_\phi)\right].
$$

这里 `R_phi` 在 generator loss 中**不能整体 detach**，否则会切断 residual 对 lambda 的梯度。起始配方 `L_policy + alpha*L_gen` 中 `alpha=0.1`，`gamma` 从 -0.5 线性到 0.5；作为本项目 padding 适配，残差与导数差 norm 均只统计最终有效 action 维度，先逐样本再做 batch 归约。首版残差为有效维度均值，omega 中使用有效维度 L2 norm；改变 norm 归一化属于独立消融，不能隐式更改尺度。

**梯度必须分开**：policy loss detach 整个 target、只更新 student；generator loss detach policy prediction 和各 JVP 数值，只让 lambda 的显式权重及相关 generator weighting 路径更新 phi，不对 policy/JVP 做高阶反传。主预测全区间导数在中点版可省略，但可学习版构造 generator weighting 时需要。推理不加载 teacher、不调用 generator，依然 1 action NFE。

论文方法的同 conditional 直线中间状态是一种经验训练设计，不在本 README 中承诺与 marginal ODE identity 严格无偏等价。K0/K1 的价值以本仓库闭环实验为准。

#### 6.3 可选改进，每次只开一项

| 改进 | 实施与注意事项 |
|---|---|
| OMP Directional Alignment | 对齐 `u` 与生成空间位移 `epsilon-a*`，不是机械臂物理速度或 3D flow。独立权重，norm/log 数值保护；在 data-based 目标上定义，不能未经推导混入任意 teacher 子轨迹标签。 |
| 有限差分替代 JVP | 固定条件及下端点，使用 `z±h*v` 和 `t±h` 的中央差分；靠近时间边界明确用合法的单边式/缩小步长，禁止越界。固定随机掩码，差分用 FP32，记录误差和训练时延；不能声称必然比正确的 no-grad JVP 省时。 |
| Dispersive loss | 它是隐层样本排斥，不是 3D teacher 对齐。先独立消融，明确层、归一化、温度、mask 和权重。 |
| CFG / guidance | 第一版关闭；若开启，训练定义和 conditional dropout 必须完整。部署再做两次 conditional/unconditional 求值不能计为 1 NFE。 |
| Riemannian MeanFlow | 涉及动作表示、噪声分布和 flow-map 目标变化，单独研究。不能把它算作“保持原 action setting，只换 loss”的主表对照。 |

先核验普通 attention 数学实现上的 JVP，再测试实际 SDPA/FlashAttention、gradient checkpointing、bf16、DDP/DeepSpeed 组合；以实际安装版本为准。短跑要验证各 rank 的参数/梯度同步，不允许用 `model.module` 绕过 wrapper 后忽略分布式同步。主预测和导数分支如含随机算子，应使用一致随机实现；不支持时显式关闭并报告差异。

### 7. 代码改动地图与后续扩展

下表保留设计与验收要求。MF/KMF 工程实现已加入；当前验证范围见顶部与 `docs/meanflow.md`，几何扩展及完整闭环验收仍待完成。实现使用独立 objective/model 层，保留 legacy FM 默认行为和原 40 层 MoT。

| 当前位置 / 拟新增位置 | 要做的修改 |
|---|---|
| `src/vpp2/models/wan21_14b/action_dit.py`：`ActionDiT.pre_dit()` | 增加可选第二端点 `timestep_end`；保留原 `time_embedding(1000*t)`，增加独立 end-time embedding，末层零初始化后加到时间表示，再走原 `time_projection`。未提供端点时完全保留 legacy 行为。原 action encoder、40 blocks、输出线性 head 不重置。 |
| 同文件 `post_dit()` | 继续输出 `[B,H,A]` 的速度；MF 模式解释为平均速度。不要改成直接 clean-action head 后仍套 `eps-u`。几何阶段才增加可选 hidden-state readout。 |
| `src/vpp2/models/wan21_14b/mot.py`：`prefill_video_cache()` / `forward_action_with_video_cache()` | 保留 40→40 的原 KV 对应、RoPE、QK norm、mask、context 和 mixed attention。封装可复用调用，不把原混合 attention 擅自改成另一个 cross-attention 架构。joint student prefill 不能 detach；teacher 和 LIBERO frozen prefill 按各自约定 no-grad。 |
| `src/vpp2/models/wan21_14b/vpp2.py`：`_predict_action_noise_with_cache()` | 当前带 `@torch.no_grad()`。抽出可微的底层 `predict_action_velocity(z,r,t,condition)` 供训练/JVP；旧 inference wrapper 继续 no-grad，保持兼容。它只求值一次 action 网络，不创建/重采样 Video cache。 |
| 同文件 `infer_action()` | 抽取公共条件构建和 prefill；增加显式 `action_sampler=flow_matching/meanflow` 分派。MF 分支只执行 `eps-u(eps,0,1)`；训练为 FM 的 checkpoint 不能因 `steps=1` 自动变成 MF。返回的 action shape、CPU/dtype 和后处理契约保持原样。 |
| 同文件 `training_loss()`、`_build_frozen_video_cache()` | 当前 joint 路径有硬编码 stage 与保护。不能只改 YAML 的 `train_mode`。保留旧路径；新增蒸馏训练路径，RoboDojo student 沿用 joint Video loss/梯度，teacher 独立 no-grad prefill。先验证拆分前后 joint 等价。 |
| `src/vpp2/libero/action_model.py`：`LiberoActionModel`、`_training_loss_action_only()` | 单独接入共享 MF objective。复用其 first-frame 编码和 frozen Video cache 路径、action mask；不要误以为改基类 `VPP2.training_loss()` 就覆盖了 LIBERO。 |
| 拟新增 `src/vpp2/meanflow/objectives.py` | 纯函数实现 masked losses、teacher-pair target、MF/KMF target、端点/diagonal sampler；不在 loss helper 内运行 VAE/Video prefill或做反归一化。 |
| 拟新增 `src/vpp2/meanflow/teacher.py` | 独立加载冻结 teacher；生成/复用同一次 rollout 的 11 个状态；保存时间网格、seed、mask。显式区分 teacher 与 student condition builders。 |
| 拟新增 `src/vpp2/meanflow/model.py` | 共用 MeanFlow mixin 或明确的 RoboDojo/LIBERO 子类，提供 `prepare_condition()`、`predict_action_velocity()`、`training_loss()`、`sample_action()`。不要复制两份逐渐分叉的数学目标。 |
| `src/vpp2/trainer.py` / `src/vpp2/libero/trainer.py` | 现有 `JointTrainer` 与 `LiberoTrainer` 的训练范围不同。新增对应分派/子类，保留原 student 参数组、LR、resume、accumulation；teacher 永不进入 optimizer；K-MF generator 单独参数组和 state。 |
| `src/vpp2/runtime.py` / `src/vpp2/libero/runtime.py`、两个 CLI | 注册新的模型/训练入口和配置校验；不得只添加 YAML 键而运行仍走 legacy loss。新配置启动时打印 objective、trainability、teacher/student hashes、sampler。 |
| `VPP2.save_checkpoint()` / `load_checkpoint()`、`src/vpp2/weights.py` / `src/vpp2/checkpoint_compat.py`、LIBERO checkpoint methods | 支持新时间分支、generator、后续 geometry heads、objective/sampler 版本、teacher 来源、student update step、原始 source step、paired Video 来源。原完整 checkpoint 只保存 `mot + proprio`，顶层新模块和 metadata 需显式加入。旧 strict loader 与导出器有固定格式/step 约束，必须显式升级；禁止整体 `strict=False` 吞掉主干缺失权重。 |
| `src/vpp2/policy.py`、`policy/VPP2/*`、`src/vpp2/libero/eval_helpers.py` / worker | 让部署按 checkpoint metadata 构造 MF 模型并选 sampler。保留 history、seed、normalization、gripper、replan；只将 MF action NFE 设置为 1。teacher 不进入部署 bundle。 |
| `src/vpp2/evaluation.py` / `src/vpp2/libero/diagnostics.py` / benchmark audit | 分开记录原 FM diagnostic 与 MF 1-NFE diagnostic、同噪声 teacher endpoint 差值；加入实际 action forward 计数、prefill 计数与 timing。audit 校验 objective/checkpoint/run ID，旧10步协议仍可运行。 |

训练和推理都必须走 `pre_dit → MoT.forward_action_with_video_cache → post_dit` 的完整条件路径；直接调用 `action_expert.forward()` 会丢失 Video KV 条件，不能拿它实现 MF/JVP 的网络函数。

不要用原 `initialize_action()` 将 Video 权重再次转换成 action 初始权重来实现蒸馏；那个脚本用于原基线的 fresh action 初始化。本计划的 student 使用已训练的 FM 配对权重。初始化格式显式区分 `full_training` 与 `deployment_pair`：`VPP2.load_checkpoint()` 接收含 `mot` 的完整训练 checkpoint；release `action.pt` 应按 `policy.py` 的部署加载路径完整恢复 `action_expert + proprio_encoder`，并加载配对 Video。原 `action_dit_pretrained_path` 要求 `backbone_state_dict` 格式的 fresh initializer，不能把 release `action.pt` 直接塞进去。新增 end-time 分支只允许明确列出的 missing keys，其余严格核验。

### 8. 拟新增配置与命令契约

保留原 YAML 不动，新增以下配置（对应文件现已新增，使用方法以 docs/meanflow.md 为准）：

```text
configs/meanflow/robodojo_distill.yaml
configs/meanflow/libero_distill.yaml
configs/meanflow/{robodojo,libero}_mf.yaml
configs/meanflow/{robodojo,libero}_kmf_learned.yaml
configs/meanflow/{robodojo,libero}_kmf_midpoint.yaml
```

每个配置从对应 benchmark 配方生成并保存完整 resolved config。至少新增以下字段；尖括号是必须填写的路径，不是可运行参数：

```yaml
# 已实现配置 schema；explicit_hybrid 与几何扩展暂不支持。
meanflow:
  objective: teacher_trajectory  # data_meanflow / kmf / explicit_hybrid
  time_convention: clean0_noise1
  student_init:
    format: deployment_pair     # 或full_training；两种格式显式分派
    action_checkpoint: <verified_fm_action_and_proprio_checkpoint>
    video_checkpoint: <paired_verified_fm_video_checkpoint>
    # full_training格式改用full_checkpoint，不能依赖旧paths.video_init回退
  preserve_baseline_trainability: true
  teacher:
    action_checkpoint: <paired_teacher_action>
    video_checkpoint: <paired_teacher_video>
    proprio_source: teacher_checkpoint
    steps: 10
    sigma_shift: 1.0            # RoboDojo闭环；LIBERO改为5.0
    frozen: true
  interval:
    source: teacher_grid        # M0/K0/K1另定义data-based sampler
    full_interval_probability: 0.5
    diagonal_probability: 0.0  # teacher-pair首版；不是MF论文默认
    min_gap: 0.0001
  loss_weighting: uniform
  derivative_backend: none      # 蒸馏none；MF/KMF才需要jvp或finite_difference
  auxiliary_geometry: false
inference:
  action_sampler: meanflow
  action_nfe: 1
```

已新增 `scripts/meanflow/train.sh --benchmark ... --config ...`，内部复用 `scripts/train.sh` 和对应 benchmark 的环境、Accelerate/DeepSpeed 配置；使用 `vpp2.meanflow.cli` 显式入口。禁止让 RoboDojo 的 12-node 默认值意外用于单机短跑。新脚本应支持现有 `--dry-run` 与 `key=value` override，并在没有必要文件/硬件配置时清楚失败。

调用形态如下（先补齐真实数据、配对权重和足够的训练资源）：

```bash
bash scripts/meanflow/train.sh --benchmark libero \
  --config configs/meanflow/libero_distill.yaml --dry-run

bash scripts/meanflow/train.sh --benchmark libero \
  --config configs/meanflow/libero_distill.yaml \
  output_dir=runs/libero/mf_distill_probe max_steps=20 \
  eval_every=0 save_every=20
```

正式训练仍使用原 train split / sampler / normalization / batch 协议；只为新实验换独立 output_dir、objective、必要的双时间条件和部署 sampler。使用新的评测配置副本指向新 checkpoint，不覆盖原配置、原 checkpoint 或原 run 的结果。不要把新的 student update step 伪装成原 `action_step=30000/100000` 以通过旧校验；扩展 metadata 和对应 audit。

### 9. 实施验收清单与交付顺序

每个子阶段完成后提交可复核的代码 diff、resolved config、短跑日志和结果位置；不要在同一个不可拆分修改中实现全部后续研究。

1. **兼容性**：新增 MF 参数默认关闭时，原模型加载和固定输入推理保持一致；40 层/heads/宽度、mask、token 长度没有改变。新时间分支零初始化时，给相同当前时间，输出与原 FM 相符。
2. **数学**：用常速度 toy field 检查 `t/r` 顺序、负步长、teacher-pair平均速度、MF identity、diagonal 与一步 endpoint；验证 `1000*t` 链式法则。以有限差分对小网络 JVP 做数值对照。
3. **梯度**：teacher/冻结模块没有梯度；student 按原 benchmark 的范围和边界获得梯度；K-MF generator 与 policy loss 的参数更新互不越界。LIBERO frozen-cache 和 RoboDojo joint 两条路径分别检查。
4. **配对**：同 seed 的 teacher cached rollout 与原 `infer_action()` 输出一致；若重构 joint→cache，比较前向及有效参数梯度；teacher rollout 状态的 `(condition,noise,mask)` 未被重采样。测试改动未来不可见 RGB 不会改变部署条件 cache。
5. **Mask / 精度**：沿用 `build_inputs()["action_loss_is_pad"]`；原实现为 `action_is_pad & ~action_terminal_mask`，部分 terminal action 仍参与监督，不能直接按原始 padding 标志删掉。最终有效监督为空的样本明确跳过/计数；不同 action 维度不广播错误。`t=1` 的蒸馏梯度非零。teacher Euler 积分保持原 dtype 和随机数抽取顺序，保存节点后转 FP32 计算 `(z_t-z_r)/(t-r)` 与 loss；若改成 FP32 积分，单列数值配置并重新验证。配对检查优先直接共享已生成的 video/action 噪声张量。
6. **20-step 短跑**：无 NaN/Inf；损失和 teacher endpoint 差距可观察；记录显存/迭代时间；有参数更新、checkpoint round-trip、resume 与多卡同步检查。短跑通过不等于算法成功。
7. **推理**：instrument 实际 action expert 完整调用数为 1；所有40层各运行一次；标准 action-only 请求保持原 prefill 次数；不加载teacher、不调用split generator、不暗中做CFG双分支/多步视频可视化。记录可选输出引入的额外调用。
8. **闭环**：使用原 episode清单完成 B0/B1/B2/D0 等；运行原审计；结果未达到第5节门槛前，不启用几何阶段。

训练日志至少包含 `objective`、各 loss、有效 action 数、时间/gap 分布、端点比例、teacher endpoint 差距、teacher NFE、student NFE、更新步数、实际全局 batch、学习率和峰值显存。MF/KMF 额外记录 JVP 范数、修正项范数及其分时间区间分布；K-MF 记录 lambda 均值/分位数、generator loss 和梯度范数。

时延用固定 warmup 后的 CUDA 同步计时，报告 p50/p95；分别统计预处理/视觉prefill/action/后处理/可选heads及端到端。每次请求生成一个32步chunk，执行24或10个动作；不能把chunk生成频率直接当成机器人控制频率。

### 10. 一步验证通过后：几何 heads 的实验与修改方式

固定前阶段选定的 action objective 和训练预算，首先只增加 readout heads。**保持40层主干和原Video KV交互，不在第一版把 geometry tokens 插入主干，也不改变层数或引入在线VGGT预处理。**

建议拟新增 `src/vpp2/geometry/heads.py` 与 `losses.py`：

- 在 `ActionDiT.post_dit()` 前返回最终 action hidden states；可选记录少量指定层，作为后续独立多层readout消融。
- point/flow head 需要明确空间位置。使用带 view ID、像素/patch坐标和预测跨度的查询，在 head 内读取原Video特征以及action hidden states；不能仅将32个action tokens reshape成稠密point map。
- pose head 输出指定参考系下的相对旋转/平移。若只读取冻结Video KV，几何loss不会训练action主干；至少保留一条未detach的共享hidden路径，并做detach对照来验证辅助梯度的作用。
- action主干每次请求只运行一次，head随后读取这次hidden states；不为3D再运行一次完整policy。全部heads开启后的实际延迟单独报告。

新增 heads 必须显式纳入两个 trainer 的解冻规则、optimizer 参数组/LR、完整 checkpoint 和部署导出。原 trainer 先冻结整个模型，再仅解冻指定模块；仅把 head 挂到模型顶层会导致它静默不训练或不保存。K-MF generator 也遵守同一要求。验收时核对每个新增可训参数在 optimizer 中恰好出现一次，分别测试完整 resume 与部署 weights round-trip；恢复后 head 输出一致，generator 的参数及 optimizer state 能继续训练。

**数据先验只确认了多视角 RGB、相机标定与 action。** 不假定已有可靠 depth、metric scene flow 或 robot-base hand–eye：

| 输出 | 标签与损失要求 |
|---|---|
| 当前3D点 | 可从同步多视角对应与标定三角化，或使用离线重建teacher伪标签；过滤重投影误差、负深度、遮挡和低置信度。采用同一参考相机系，只有可靠物理尺度时才声明metric。valid/confidence加权Huber等鲁棒损失。 |
| Camera pose | 明确当前跨视角相对pose还是未来移动相机pose；旋转用合法表示与SO(3)误差，平移按统一尺度。固定rig的已知外参近似常量，先比较常量预测基线，不能预设会提升action。 |
| 3D flow | 同一物理点在同一参考坐标系下的跨物理时间位移；需要跨帧对应、可见性和时间跨度。不能相减两张独立point maps；相机运动要变换到共同参考系后再算。没有可信标签则保持该head关闭。 |

几何标签只在训练时作为监督；数据处理保留 view/timestamp、resize/crop 后的intrinsics、extrinsics方向、坐标单位、置信度和teacher版本。训练/验证划分按episode隔离，不能因同场景多视角把测试信息混入训练。

优先从**当前点/当前几何**开始，再加入动态flow。部署输入是纯action噪声，因此当前几何监督优先在同样的full-interval主预测上计算，避免依靠低噪声action中的示范信息。若选定MF配方没有端点样本，给有/无几何的配对实验同时加入相同端点前向与训练预算，单列此修改。固定观测、改变action noise或时间，检查当前几何输出是否异常漂移。

未来flow与action存在行为模式配对问题：随机生成action不一定对应该条示范的未来，因此不要直接把任意生成action和这条示范future flow做MSE，就宣称二者一致。第一版可用小型action-conditioned head `D(C,H,a_GT) -> F_GT`，训练使用真实配对，推理接本次生成的action；记录teacher-forcing分布差异并检查head是否真的依赖action。这里的 `H` 同样来自纯 action 噪声的 full-interval 前向，训练 `a_GT` 与推理 `a_hat` 使用完全相同的 action 表示、归一化和时间跨度。该设计先验证辅助动态预测，不证明联合生成一致性；联合 `[action,flow]` 生成留作独立研究。

总loss以已选 action loss 为基础，各几何项按有效点数、单位和跨度归一化，权重单独配置。保持action-only对照，依次做 `+当前点`、`+flow`、`+非平凡pose`，再做相同heads但几何loss对action主干detach的对照。需要证明的是**闭环action收益**；仅有更低重建误差不够。

### 11. 最后阶段：VGGT 权重是否带来额外增益

前置条件是一步 action 与几何heads分别已有稳定对照。只验证：在相同的下游架构、输入、监督、数据和预算下，VGGT预训练信息能否进一步提高action成功率、几何质量或数据效率。

至少保留不使用VGGT权重的配对对照；若该阶段必须改变架构，还需要相同新架构但不使用VGGT权重的对照，避免把拓扑变化的收益归给预训练。记录总参数量、训练参数量、训练成本和端到端延迟。

**本 README 暂不描述 VGGT 权重的加载、层映射、宽度适配或初始化细节。** 后续单独设计，不提前改动40层基线。

### 12. 结果记录模板与参考资料

每个run输出独立 `manifest.json`、resolved config、checkpoint来源/hash、数据split/hash、normalization版本、teacher标签成本、随机seed、硬件/精度/attention实现、训练更新数与时间。最终表至少包含：

| Run | Objective | Teacher steps/shift | Student action NFE | 训练预算/seed | 几何heads | 原协议闭环指标与CI | action p50/p95 | E2E p50/p95 | 峰值显存 | audit |
|---|---|---|---|---|---|---|---|---|---|---|
| 待运行 | — | — | — | — | — | 未验证 | — | — | — | 未运行 |

不预填论文成绩作为本仓库结果；固定验证/选模规则，不按测试集最高checkpoint挑选。没有完整评测只报告已完成范围。若结论涉及优于原FM，必须同时展示等额继续训练的B2；若涉及几何或VGGT增益，展示对应匹配对照。

方法来源（实验规格与实现建议由本项目制定）：

- [VPP2](https://arxiv.org/abs/2610.10270)：原架构及benchmark背景；本节具体默认值以锁定源码和仓库文档为准。
- [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)、[作者PyTorch实现](https://github.com/Gsunshine/py-meanflow)：平均速度、JVP、stop-gradient与分布式实现注意事项。
- [Kinematic MeanFlow](https://arxiv.org/abs/2610.00864)：分段导数目标和可学习分割。正式实现时核对作者后续代码；不要推定论文伪代码已解决全部梯度/边界细节。
- [MP1](https://arxiv.org/abs/2507.10543)、[OMP](https://arxiv.org/abs/2512.19347)：机器人MeanFlow配方与正则的候选消融，不能直接外推到本40层模型。
- [Riemannian MeanFlow Policy](https://arxiv.org/abs/2609.30127)：动作流形方向的独立研究参考。

---


## Installation

```bash
git clone https://github.com/roboterax/video-prediction-policy-2.git
cd video-prediction-policy-2
conda create -n vpp2 python=3.10 -y
conda activate vpp2
```

Install the CUDA/PyTorch dependencies using the [installation guide](docs/robodojo.md#policy-environment), then install the dependencies from the repository root:

```bash
python -m pip install -r requirements-train.txt -c environment-reference.txt
```

The scripts load VPP2 directly from this checkout; installing the `vpp2` package
is unnecessary. For inference and data processing only, use `requirements.txt`.

## Get started

### Zero-shot video prediction

| Task | What is included | Guide |
|---|---|---|
| Video prediction | Input image processing, prompt enhancement, and single-image or batch video prediction with [sample inputs](assets/random_samples/). | [Video prediction guide](docs/video_prediction.md) |

### Policy learning

| Benchmark | What is included | Guide |
|---|---|---|
| RoboDojo | Joint Video + Action2B training, checkpoint export, policy serving, and closed-loop evaluation | [RoboDojo guide](docs/robodojo.md) |
| LIBERO | Video-stage training, action training, and evaluation on LIBERO, LIBERO-OOD, and LIBERO-PRO | [LIBERO guide](docs/libero.md) |

## Model Weights

Weights are available on [Hugging Face](https://huggingface.co/Haodong082399/VPP2) and
[ModelScope](https://modelscope.cn/models/haodong123/VPP2).
The Hugging Face repository is public; ModelScope requires an account with access.

| Checkpoint | Use | Hugging Face | ModelScope |
|---|---|---|---|
| VPP2 Stage-1 Video (49 frames) | Zero-shot video prediction(Event level) | [Download](https://huggingface.co/Haodong082399/VPP2#stage-1-video) | [Download](https://www.modelscope.cn/api/v1/models/haodong123/VPP2/repo?Revision=master&FilePath=checkpoints_video/vpp2-video-stage1-49f.pth) |
| VPP2 Stage-2 Video (17 frames) | Zero-shot video prediction (Fixed time horizon) | [Download](https://huggingface.co/Haodong082399/VPP2#stage-2-video) | [Download](https://www.modelscope.cn/api/v1/models/haodong123/VPP2/repo?Revision=master&FilePath=checkpoints_video/vpp2-video-stage2-17f.pth) |
| RoboDojo history-conditioned Video-10k | Starting Video model for joint + Action2B training | [Download](https://huggingface.co/Haodong082399/VPP2#robodojo-training) | [Download](https://www.modelscope.cn/api/v1/models/haodong123/VPP2/repo?Revision=master&FilePath=checkpoints/initialization/robodojo_his10k.pt) |
| RoboDojo joint Video-100k | Video model paired with Action2B-100k for evaluation | [Download](https://huggingface.co/Haodong082399/VPP2#robodojo-evaluation) | [Download](https://www.modelscope.cn/api/v1/models/haodong123/VPP2/repo?Revision=master&FilePath=checkpoints/joint2b_s100000/video.pt) |
| RoboDojo Action2B-100k | RoboDojo policy evaluation | [Download](https://huggingface.co/Haodong082399/VPP2#robodojo-evaluation) | [Download](https://www.modelscope.cn/api/v1/models/haodong123/VPP2/repo?Revision=master&FilePath=checkpoints/joint2b_s100000/action.pt) |
| LIBERO Video-10k | Video model for Action training and evaluation | [Download](https://huggingface.co/Haodong082399/VPP2#libero) | [Download](https://www.modelscope.cn/api/v1/models/haodong123/VPP2/repo?Revision=master&FilePath=checkpoints/libero/video_step010000.pt) |
| LIBERO Action-30k | LIBERO, LIBERO-OOD and LIBERO-PRO evaluation | [Download](https://huggingface.co/Haodong082399/VPP2#libero) | [Download](https://www.modelscope.cn/api/v1/models/haodong123/VPP2/repo?Revision=master&FilePath=checkpoints/libero/action_step030000.pt) |

The Hugging Face download instructions include the root `config.json` with each checkpoint.

The Stage-1 and Stage-2 video checkpoints are available on the Hugging Face main branch
and mirrored on ModelScope.
See the [video prediction guide](docs/video_prediction.md) for input preparation and inference.

Policy evaluation uses the matching Video and Action checkpoints together. Shared Wan2.1
VAE, CLIP, UMT5 and tokenizer assets are also available in the model repositories
under `checkpoints/Wan2.1-I2V-14B-480P/`. Follow the benchmark guides to download the
complete assets and normalization files:

- [RoboDojo downloads and evaluation](docs/robodojo.md#checkpoint-downloads)
- [LIBERO / OOD / PRO downloads and evaluation](docs/libero.md#3-data-and-required-model-files)

<!-- ## TODO

- [ ] Release large-scale, event-level video pretraining code and configurations. -->

## Links

| Resource | Link |
|---|---|
| Paper | [arXiv:2610.10270](https://arxiv.org/abs/2610.10270) |
| Project page | [VPP2 Project Page](https://robert-gyj.github.io/video-prediction-policy-2/) |
| XPolicyLab adapter | [VPP2 RoboDojo adapter PR](https://github.com/XPolicyLab/XPolicyLab/pull/150) |
| ModelScope | [haodong123/VPP2](https://modelscope.cn/models/haodong123/VPP2) |
| Hugging Face | [Haodong082399/VPP2](https://huggingface.co/Haodong082399/VPP2) |

## Documentation

- [Zero-shot video prediction](docs/video_prediction.md)
- [RoboDojo: training and testing](docs/robodojo.md)
- [LIBERO, LIBERO-OOD and LIBERO-PRO](docs/libero.md)

## Citation

If you use VPP2 in your research, please cite our [paper](https://arxiv.org/abs/2610.10270)
([BibTeX](CITATION.bib)):

```bibtex
@misc{guo2026videopredictionpolicy2,
  title = {Video Prediction Policy 2: Predict Better, Act Better},
  author = {Yanjiang Guo and Haodong Yan and Zhide Zhong and Zhongru Zhang and
            Qingyuan Yang and Qingzhou Lu and Xiaoyu Chen and Yen-Jen Wang and
            Shuying Deng and Chenghan Yang and Puzhen Yuan and Chenxin Liu and
            Tun Ban and Xiang Zhu and Yichen Liu and Kun Feng and Haoang Li and
            Jianyu Chen},
  year = {2026},
  eprint = {2610.10270},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2610.10270}
}
```

## License

The code is released under the [MIT License](LICENSE). Third-party components
retain their original licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
Model weights, datasets, and simulator assets have their own terms.

### LIBERO: controlled one-step training

New LIBERO Action defaults: first-frame decoding, uint8 image path, 4 workers/GPU.
See [MeanFlow and Direct Action MSE](docs/libero_one_step.md) for the shared controls,
16-H200 Qizhi commands, environment activation, logs and validation.
Both start from baseline Action initialization, not the trained Action30k release.

### LIBERO baseline: training-data fractions

The manifests in `configs/libero/subsets/` select whole episodes from all 40 tasks,
using seed 42, task-stratified quotas and nested prefixes (1/8 ⊂ 1/4 ⊂ 1/2).
They reference the existing LeRobot data; videos and parquet files are not copied.
Within each task, episodes are ordered by SHA256 of the compact JSON array
`[42, dataset_basename, task_index, episode_index]`. Quotas use largest remainders;
ties use SHA256 of `[42, dataset_basename, task_index, "quota"]`.

| Fraction of episodes | Episodes | Frames |
| --- | ---: | ---: |
| 1/8 | 214 | 34,900 |
| 1/4 | 428 | 69,942 |
| 1/2 | 856 | 139,789 |
| Full | 1,712 | 277,713 |

Submit **three separate one-node, 8-H200 jobs** with the following command; set
`FRACTION` to `1of8`, `1of4` or `1of2`. The launcher derives batch 32/GPU,
global batch 256 and gradient accumulation 1 from the allocation.

```bash
bash -c '
set -euo pipefail
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
source scripts/libero/activate_env.sh
FRACTION=1of8
MANIFEST="configs/libero/subsets/episodes_${FRACTION}_seed42.json"
MANIFEST_SHA256="$(sha256sum "$MANIFEST" | cut -d " " -f 1)"
export WANDB_PROJECT=VPP2-LIBERO
export RUN_NAME="libero-baseline-${FRACTION}-${TRAIN_JOB_ID:?}-round${RUNNING_ROUND:-0}"
bash scripts/libero/train_action.sh \
  "data.train.episode_manifest=$MANIFEST" \
  "data.train.episode_manifest_sha256=$MANIFEST_SHA256"
'
```

The queued jobs pin the manifest hash at submission. Each run saves
`episode_manifest.json` and `episode_selection.json` alongside `config.yaml`.
Selection happens inside the LeRobot loader, including its random error retries.
Dataset paths in manifests resolve from the repository root, as in the training
configuration. An explicit manifest requires `val_set_proportion=0`.

All runs start from the same Action initialization, use the same frozen full-data
Video10k checkpoint and official normalization, and retain the baseline recipe:
30,000 steps, LR 1e-4, cosine schedule, 500 warmup steps, BF16, ZeRO-1,
first-frame uint8 decoding, 4 workers/GPU, checkpoints every 5,000 steps and
offline W&B. This measures Action-stage data efficiency at fixed training steps;
smaller subsets are revisited more often. It is not an end-to-end low-data Video
training experiment. The four training-sample diagnostics are not held-out
evaluation and refer to different samples across subsets; compare final policies
with the same closed-loop LIBERO-PRO evaluation protocol.

To validate episode membership, nesting, retry bounds and exact equality of
boundary samples against the full-data loader (including action padding):

```bash
source scripts/libero/activate_env.sh
python checks/libero_subset_data.py
```
