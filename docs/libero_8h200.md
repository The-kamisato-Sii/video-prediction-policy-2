# LIBERO 官方配方与本机 8×H200 调试

配置依据：[VPP2 官方 LIBERO 文档](https://github.com/roboterax/video-prediction-policy-2/blob/72c17acf022494ff63617be86079bb92e92f5bf8/docs/libero.md)。
本次使用原始 FM Action 训练，不启用 MeanFlow。官方源配置保留在
`configs/libero/train_video.yaml` 和 `configs/libero/train_action.yaml`。

## 训练配置对照

| 项目 | Stage 1 Video | Stage 2 Action（官方） |
|---|---|---|
| 训练范围 | Video | Action 1B + proprio 投影；Video/VAE/CLIP/text 冻结 |
| 初始化 | robot-video step 28k，作者尚未发布 | 已发布 LIBERO Video step 10k；从它插值初始化新 Action |
| 单卡 micro batch / 卡数 / 梯度累积 | 8 / 8 / 1 | 32 / 8 / 1 |
| Global batch | 64 | 256 |
| 最大 optimizer steps | 10,000 | 30,000 |
| LR / warmup | 5e-6 / 200 | 1e-4 / 500 |
| AdamW betas / weight decay | (0.9, 0.999) / 0.01 | (0.9, 0.95) / 0.01 |
| LR 调度 / 最低比例 | cosine / 0.01 | cosine / 0.01 |
| 精度 / clipping / seed | BF16 / 1.0 / 42 | BF16 / 1.0 / 42 |
| checkpoint / diagnostic 间隔 | 2,000 / 1,000 | 5,000 / 200 |
| 时间步 | Video Beta(7,1) | frozen Video t=1000；Action shift=5 |
| 分布式 | ZeRO-2，无 CPU offload | ZeRO-2，无 CPU offload |

实测可训练参数为 **929,436,167**（含 proprio），其中插值初始化的骨干为 929,391,616。
Action hidden=512、FFN=2048、40 层、40 heads × 128；与 RoboDojo Action2B 配方不同。
`train_wan: true` 不表示这里更新 Video：`LiberoTrainer._apply_train_mode()` 先冻结全部模块，
然后只开放 Action 和 proprio。ZeRO-2 分片优化器状态及梯度；冻结 Video 的参数仍每卡保留一份。

输入：两路相机各 resize 到 224×224，横向拼接成 224×448；65 个观测时间点取
`t0,t4,...,t64` 共 17 帧；Action `[32,7]`，proprio `[1,8]`。
官方首帧快路径只将第一帧送入 VAE/CLIP，仍构造五帧 Video latent 和双向 Video attention；
Action 直接看前三帧 Video K/V。文本缓存 context length=128。

## 数据与权重来源

VPP2 文档注明其精确训练数据尚待发布。本次调试使用
[FastWAM 的公开 MuJoCo 3.3.2 / LeRobot v2.1 数据](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)，
固定 revision `ee018b997c430bb12b5bf3c892d744798c5a2f91`，四个压缩包均校验 SHA256。
数量匹配不能证明与作者数据逐字节一致，因此不能把短跑称为论文成绩复现。

| Suite | Episodes | Frames |
|---|---:|---:|
| Spatial | 434 | 53,229 |
| Object | 457 | 67,309 |
| Goal | 433 | 52,895 |
| LIBERO-10 | 388 | 104,280 |
| 合计 | 1,712 | 277,713 |

额外聚合 1,712 条 episode 的元数据统计后，state/action 的全局 min/max 与 VPP2
自带统计逐项完全相等；这增加了匹配证据，但仍不是全体图像/数据行的逐字节校验。

使用全部 demonstrations、自然帧数混合、无验证集划分。
归一化使用 VPP2 自带 `configs/libero/dataset_stats.json` 的 min/max（包括 gripper）；
前六个 action 坐标使用 delta 标记，gripper 不使用 delta。
统计文件 SHA256 `4717a109760fcd1d504c7d301ff95106b14446a89b019a026427ddbbcc866d1b`。

Video 权重来自 [作者 VPP2 release](https://huggingface.co/Haodong082399/VPP2/tree/04824ddb3aae11f13db50984ed7d540c3b77dd9f)，
文件 `checkpoints/libero/video_step010000.pt`，32,789,968,889 bytes，SHA256：
`7f7a08073facfd08037801a83ba60ee45cf386599a47679fc91b11ccbdeb15ba`。
使用 ModelScope 镜像传输，并用上述固定版本的 HF LFS digest 校验。
共享 Wan VAE/CLIP/UMT5/tokenizer 复用已有文件，不重复下载。

本机目录（相对于仓库）：

- `data/libero/`：四套数据、官方归一化统计、文本缓存、下载来源记录。
- `weights/libero/video_step010000.pt`：LIBERO 专用 Video，不能替换成 RoboDojo Video。
- `weights/libero/action_init.pt`：从上述 Video 初始化的新 Action，不是已训练 Action-30k。
- `weights/Wan2.1-I2V-14B-480P/`：共享编码器。
- `runs/libero_8h200/`：下载日志、真实训练日志、完整 resolved config、checkpoint 与 profiler。

## 准备与训练命令

```bash
cd /inspire/hdd3/project/embodied-model/linbokai-CZXS24250037/video-prediction-policy-2
export VPP2_ROOT=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037
export PATH="$VPP2_ROOT/miniconda/envs/vpp2/bin:$VPP2_ROOT/tools/cuda-12.8/bin:$PATH"
export PYTHON_BIN="$VPP2_ROOT/miniconda/envs/vpp2/bin/python"
export CUDA_HOME="$VPP2_ROOT/tools/cuda-12.8"
export OMP_NUM_THREADS=2
```

Video 文件的固定版本下载命令（本机已完成；共享编码器已复用）：

```bash
hf download Haodong082399/VPP2 checkpoints/libero/video_step010000.pt \
  --revision 04824ddb3aae11f13db50984ed7d540c3b77dd9f \
  --local-dir weights/libero_release
printf '%s  %s\n' \
  7f7a08073facfd08037801a83ba60ee45cf386599a47679fc91b11ccbdeb15ba \
  weights/libero_release/checkpoints/libero/video_step010000.pt | sha256sum -c -
mkdir -p weights/libero
test -e weights/libero/video_step010000.pt || \
  ln -s ../libero_release/checkpoints/libero/video_step010000.pt weights/libero/video_step010000.pt
```

数据下载和准备（Action 初始化只执行一次，脚本拒绝覆盖已有权重）：

```bash
python scripts/libero/download_public_data.py
bash scripts/libero/prepare.sh --config configs/libero/train_action.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/libero/text_cache.sh --device cuda
bash scripts/libero/init_action.sh \
  --video weights/libero/video_step010000.pt \
  --config configs/libero/train_action.yaml --output weights/libero/action_init.pt
```

8 卡全局 batch 256 的两种方式；launcher 只处理本机路径及 batch，完整学习超参数来自官方 YAML。官方每 rank 的 DataLoader workers=16；本机默认 4，
因为节点 `/dev/shm` 为 128 GiB，避免 batch 32 的大量视频预取队列耗尽共享内存。
可用 `NUM_WORKERS=16` 恢复官方 worker 数；它不改变样本、batch 或优化器配置：

```bash
# 默认：单卡 16，累积 2，8×16×2=256
bash scripts/libero/train_8h200.sh --dry-run
bash scripts/libero/train_8h200.sh output_dir=runs/libero/action_b16_acc2

# 单卡 32，累积 1，8×32×1=256；已跑通，但显存余量很小
BATCH_SIZE=32 GRAD_ACC=1 bash scripts/libero/train_8h200.sh \
  output_dir=runs/libero/action_b32
```

本机正式参数的完整快照为 [`docs/validation/libero_8h200_formal.yaml`](validation/libero_8h200_formal.yaml)，
可直接与官方 `configs/libero/train_action.yaml` 对比；其中 max_steps=30000、warmup_steps=500。
这是当时单卡 16 / 累积 2 的参数预览。之后按用户选择启动了单卡 32 / 累积 1 的正式任务，
实际配置在 `runs/libero/action_b32_20261016_0719/config.yaml`。
两节点、单卡 16 的启动方法见 [16 卡与离线 W&B](libero_16h200.md)。

单卡 16、累积 1 在 8 卡上是 global batch 128，仅作明确标注的诊断；
若以后扩到 16 卡，16×16×1 才是 256。本次不申请 16 卡。

## 短跑与恢复检查

短跑保留官方配置中的 base LR / warmup 数值、数据、模型、优化器和 batch，缩短步数并提前保存/诊断。
原 trainer 会按 `max_steps` 自动压缩实际 warmup 和 cosine 周期；因此这属于容量/链路诊断，
并非正式 30k 训练的精确前 20 步。正式任务从 `action_init.pt` 重新启动，
使用默认 30,000 步 / warmup 500；短跑 state 仅用于恢复检查。
检查器验证每卡冻结 Video 的完整 tensor hash、Action/proprio 更新、有限非零梯度、8 卡参数一致；
官方 preflight 还检查输入形状、峰值显存和 checkpoint 加载。

```bash
BATCH_SIZE=32 GRAD_ACC=1 VPP2_TRAIN_MODULE=checks.libero_training_audit \
  bash scripts/libero/train_8h200.sh \
  output_dir=runs/libero/probe_b32 max_steps=20 \
  preflight.enabled=true preflight.expected_steps=20 \
  preflight.memory_headroom_fraction=0.05 \
  log_every=1 save_every=20 eval_every=20 \
  profiling.enabled=true profiling.skip_steps=2 profiling.collect_steps=18

# STATE 使用 probe.json 中的 state_path（完整优化器/调度器/RNG/数据进度状态目录）。
BATCH_SIZE=32 GRAD_ACC=1 VPP2_TRAIN_MODULE=checks.libero_training_audit \
  bash scripts/libero/train_8h200.sh \
  output_dir=runs/libero/probe_b32_resume resume="$STATE" max_steps=22 \
  preflight.enabled=true preflight.expected_steps=22 \
  log_every=1 save_every=22 eval_every=0
```

每次选择新输出目录。`probe.json`、`training_audit.json`、`profiling/summary.json`
分别记录 preflight、参数/梯度检查和耗时。正式训练使用默认入口即可，不执行额外完整 hash。
训练内固定四个样本的 10 步 FM action diagnostic 不等于 simulator 闭环成功率；
标准 LIBERO/OOD/PRO 闭环评测沿用 [官方评测步骤](libero.md#7-evaluate-the-same-action-30k-checkpoint)。

## 激活 checkpoint 修复

干净官方 commit 的 `LiberoTrainer._apply_train_mode()` 在全模型 `eval()` 后只切换
Action 子模块，MoT 容器仍为 `training=False`。MoT 的 mixed-attention 和 post-attention
checkpoint 都要求 `self.training=True`，导致 YAML 中的 checkpoint 开关实际未生效。
本次修复为先将 MoT 切到训练态，再明确将冻结 Video 切回 eval；可训练参数集合不变。
回归测试 `checks/test_libero_checkpointing.py` 检查实际 checkpoint 调用、输出/全部训练参数梯度
与关闭 checkpoint 的参考实现一致，并保证 Video 无梯度。

未修复版本的真实数据 batch 32 已在 Action cross-attention 前向 OOM：设备 139.81 GiB，
PyTorch allocated 137.66 GiB，申请额外 42 MiB 失败。日志为
`runs/libero_8h200/b32_a1.log`。这一结果不应被当成修复后的容量上限。

多卡启动还修复了一处输出目录检查竞态：只有写配置的 rank 0 检查已有输出，
避免它写入本次 `config.yaml` 后其他 rank 误报重复任务；已有任务的防覆盖检查保留。
对应回归测试为 `checks/test_libero_launch.py`。

## 本次实测

修复后 batch 32 / 累积 1 / global 256 已完成 20 次真实数据更新。

| 配置 | 更新数 | 峰值 allocated | 峰值 reserved | 5% 余量检查 |
|---|---:|---:|---:|---|
| 修复前，32 × 8 × 1 | 0 | 137.66 GiB（OOM 时） | — | OOM |
| 修复后，32 × 8 × 1 | 20 + 恢复 2 | 136.31 GiB | 137.18 GiB | 未通过 |
| 修复后，16 × 8 × 2 | 8 | 86.15 GiB | 87.23 GiB | 通过 |

8 个 rank 的完整 Video tensor hash 前后不变；Action/proprio 参数更新，梯度有限且非零；
各卡最终 Action/proprio/Video hash 一致；DeepSpeed 与 trainer 的 optimizer-step 计数一致。
compact Action checkpoint 加载通过，4 个固定样本的 10 步 FM diagnostic 完成
（action L1=0.25462、MSE=0.0709594，仅训练样本诊断）。

跳过前两步，18 次更新按每步最慢 rank 统计：均值 **11.49 秒**、中位数 **9.67 秒**；
包含数据等待和同步 profiling，不含加载、完整 hash、评测与保存。
仅训练步外推 30k 约 **95.79 小时 / 3.99 天**，长期耗时仍取决于 I/O 与周期任务。
完整 DeepSpeed 状态保存约 3 分钟；与每步速度分别记录。

batch 32 已从第 20 步恢复至第 22 步，模型/优化器/调度器/RNG 和数据位置均成功恢复，
8 卡参数一致性与 Video 冻结检查再次通过。

batch 16 / 累积 2 / global 256 已完成 8 次更新，完整 checkpoint 加载、8 卡一致性、
Video 完整 hash 不变、Action/proprio 有限非零梯度和更新均通过。
跳过前两步后，6 次更新的最慢 rank 均值 **15.69 秒**、中位数 **15.06 秒**；
短跑外推 30k 训练步约 **130.78 小时 / 5.45 天**，不包括周期评测/保存及长期波动。
因此本机默认使用 **batch 16 / 累积 2**。未来若采用 16 卡，改为每卡 16 / 累积 1 即保持
global 256；跨节点吞吐仍须另测。这些调试结束时没有申请 16 卡或启动正式任务；
后来启动的 8 卡正式任务使用单卡 32，路径见上文。

全部结构化证据见 [`libero_8h200.json`](validation/libero_8h200.json)。
20 项相关 CPU 回归测试通过；两种 batch 的 launcher dry-run 均显示 global 256 / 30k steps。
实际运行目录分别为：

- `runs/libero_8h200/b32_a1_checkpoint_fixed/`：20 步、推理诊断、完整状态保存。
- `runs/libero_8h200/b32_a1_checkpoint_fixed_resume/`：从 20 恢复到 22 步。
- `runs/libero_8h200/b16_a2_headroom/`：单卡 16 / 累积 2 的 8 步验证。

重跑 batch 16 检查（选择新目录）：

```bash
VPP2_TRAIN_MODULE=checks.libero_training_audit \
  bash scripts/libero/train_8h200.sh \
  output_dir=runs/libero/probe_b16 max_steps=8 \
  preflight.enabled=true preflight.expected_steps=8 \
  preflight.memory_headroom_fraction=0.05 \
  log_every=1 save_every=8 eval_every=0 \
  profiling.enabled=true profiling.skip_steps=2 profiling.collect_steps=6

PYTHONPATH=src OMP_NUM_THREADS=2 python -m pytest -q \
  checks/test_libero_checkpointing.py checks/test_libero_launch.py checks/test_meanflow.py
```
