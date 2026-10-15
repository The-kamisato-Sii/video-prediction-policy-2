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
- 先使用 uniform masked loss。adaptive weighting、CFG、dispersive 等各自作为单独开关与消融；不复制 MP1 未核对的 CFG 实现，也不把含义相反的 `flow_ratio` 变量直接搬来。

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
