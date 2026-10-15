# RoboDojo 原始 FM baseline：单机 8×H200

本配置复现原始 joint Video + Action2B 训练目标。Video 从公开的
history-conditioned Video-10k 开始，Action2B 从同一 Video 初始化，
训练 100,000 次优化器更新。保持全局 batch **288 = 8 卡 × 每卡 3 × 累积 12**，
保留 40 层主干、原学习率和 80k–100k 尾段，使用 ZeRO-2。
这与加载 joint-100k 权重进行推理、或从该权重继续训练是不同的实验。

## 本机环境

```bash
root=/inspire/hdd3/project/embodied-model/linbokai-CZXS24250037
source "$root/miniconda/etc/profile.d/conda.sh"
conda activate vpp2
cd "$root/video-prediction-policy-2"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export CUDA_HOME="$root/tools/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

本机 `vpp2/bin/ffmpeg` 已连接到环境中 `imageio-ffmpeg` 提供的 FFmpeg 7.0.2，
并在 H200 实例验证了 `libx264`。转换数据不依赖容器预装的系统 FFmpeg。

## 数据和初始化

共享 Wan VAE/CLIP/UMT5 位于
`weights/robodojo_release/checkpoints/Wan2.1-I2V-14B-480P`。
下载真实 EE16 数据及 history-10k 初始化权重：

```bash
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --revision cefcfbbf2497103fe46b99039dbd381976fe4a42 \
  --local-dir data/robodojo_download \
  --include 'data/RoboDojo_ee_lerobot_v30_video/**'
hf download Haodong082399/VPP2 \
  --revision 04824ddb3aae11f13db50984ed7d540c3b77dd9f \
  --local-dir weights/robodojo_baseline \
  --include 'checkpoints/initialization/robodojo_his10k.pt'
```

数据清单为 326 个文件、65,510,063,351 字节，包含 3500 个 episode。
初始化文件为 32,789,968,889 字节，SHA256：
`2d2490bd60272ec37423dabafaed4a889b2b2bf3760ac96d63d551d723ad2852`。
下载地址分别为 [RoboDojo 数据集](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)
和 [VPP2 权重](https://huggingface.co/Haodong082399/VPP2)。

在文件下载完整后执行以下步骤。转换和准备命令拒绝覆盖已有输出目录。

```bash
bash scripts/robodojo/convert.sh \
  --source data/robodojo_download/data/RoboDojo_ee_lerobot_v30_video \
  --output data/robodojo_native --workers 16
bash scripts/robodojo/prepare.sh \
  --metadata data/robodojo_native/full_episode_metadata.csv \
  --media-root data/robodojo_native --output data/robodojo_prepared
CUDA_VISIBLE_DEVICES=0 bash scripts/robodojo/text_cache.sh \
  --data data/robodojo_prepared \
  --wan weights/robodojo_release/checkpoints/Wan2.1-I2V-14B-480P
bash scripts/robodojo/init_action.sh \
  --video weights/robodojo_baseline/checkpoints/initialization/robodojo_his10k.pt \
  --output weights/robodojo_baseline/init/action2b_from_his10k.pt
```

`prepare` 检查原始帧对齐与 EE16 数据，生成固定 3466/34 划分、归一化统计和
critical sampling index。文本缓存使用一张 GPU，勿与占满显存的训练同时运行。

## 训练与恢复

```bash
# 检查配置；不会分配 GPU，也不代表数据已经准备完整。
bash scripts/robodojo/train_8h200.sh --dry-run

# 在可见 8 张 H200 的实例中启动完整原配方。
bash scripts/robodojo/train_8h200.sh

# 从完整训练状态恢复；替换为实际已保存的目录。
bash scripts/robodojo/train_8h200.sh \
  resume=runs/robodojo_baseline_8h200_100k/checkpoints/state/step_005000
```

恢复路径应使用实际生成的 Accelerate/DeepSpeed 状态目录；部署权重不包含
优化器和调度器状态。正式输出目录默认是 `runs/robodojo_baseline_8h200_100k`。
短程诊断须指定独立 `output_dir`，不能当作 100k baseline 的收敛结果。

先检查真实数据的前 20 次更新、保存完整状态，再恢复到 100k：

```bash
bash scripts/robodojo/train_8h200.sh \
  output_dir=runs/robodojo_baseline_8h200_first20 max_steps=20 \
  eval_at_start=false eval_every=0 log_every=1 save_every=20
bash scripts/robodojo/train_8h200.sh \
  output_dir=runs/robodojo_baseline_8h200_first20 \
  resume=runs/robodojo_baseline_8h200_first20/checkpoints/state/step_000020
```

保留 `lr_tail` 时，学习率按绝对步数和固定 80k/100k 边界计算；上述 `max_steps=20`
不会把 100k 学习率曲线压缩到 20 步。恢复命令回到原配置的 100k 上限、周期评估和
保存频率。首次 20 步关闭评估以集中验证训练链路，恢复后再执行 held-out 评估。

完成训练后的成对权重导出、54-entry / 2100-episode 测试命令见
[RoboDojo 训练与测试](robodojo.md#4-export-the-paired-100k-bundle)。

## 全尺寸吞吐诊断

以下使用已发布 joint-100k 权重作为测速输入，以合成 RGB 驱动真实 VAE、CLIP、
原始 FM loss、梯度和 ZeRO-2 优化器。它不训练正式 baseline，不保存部署权重，
不测真实数据解码、周期评估或 checkpoint 写盘开销。

```bash
python -m accelerate.commands.launch \
  --config_file configs/robodojo/accelerate_zero2.yaml \
  checks/profile_robodojo_baseline.py \
  --output runs/new_fm_b3_a12_profile \
  --batch-size 3 --accumulation 12 --steps 2 --warmup-steps 1
```

每个优化器更新的计时覆盖全部累积微批次。报告位于输出目录的
`profiling/summary.json`，训练时间估算应使用完整累积更新的耗时，
不能将 batch 24 的单次更新直接当作 batch 288 的更新。

2026-10-15 在本机 8×H200 上的[实测记录](validation/robodojo_baseline_throughput.json)：

| 配置 | 稳定更新耗时 | 已分配显存峰值 / 卡 |
|---|---:|---:|
| 每卡 3、累积 1，全局 batch 24，4 个计时更新 | 29.11 秒 | 98.12 GiB |
| 每卡 3、累积 12，全局 batch 288，1 个计时更新 | 349.33 秒 | 102.62 GiB |

完整累积测试实际执行 2 次更新，排除第一个预热更新。按当前配置的实测速度，
100k 步纯计算约 **404 天**，前 20 步约 **1.94 小时**；上述额外开销尚未计入。
这是短程合成输入测量的外推，不是实际完成训练的时间，也不是 H200 的性能上限。
独立 400 MB all-reduce 约 259 GB/s；启用连续梯度和通信重叠的对照测试仍约
29.24 秒 / batch-24 更新，没有明显提速，因此保留原 ZeRO-2 配置。
真实数据准备、首次训练和端到端吞吐仍需单独验证。

交互式实例按用户要求设置高优先级、关闭定时自动停止；这些脚本不会停止实例。
平台自身的抢占和资源回收策略仍由平台管理。
