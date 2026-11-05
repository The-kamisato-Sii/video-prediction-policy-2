# LIBERO：4RC pseudo-flow、MotionVAE 与共享 Action DiT

本文对应 `src/vpp2/libero_pseudo_flow/` 和 `src/vpp2/libero_flow/` 的实现。
所有命令从本仓库根目录运行。生成结果只写入 `pseudo_flow/4rc_v1/`；
原始仿真 GT 的 `motions/`、`motions_fast/` 不会被替换。

## 1. 哪些代码在仓库里

| 代码 | 职责 |
| --- | --- |
| [`model.py`](../src/vpp2/libero_pseudo_flow/model.py) | 加载官方 4RC；编码窗口；查询 source/target 对；与官方全目标 decoder 对比 |
| [`extract.py`](../src/vpp2/libero_pseudo_flow/extract.py) | 读取 LeRobot 主相机 RGB；分窗口提取；坐标/尺度对齐；episode 锁与断点续跑 |
| [`geometry.py`](../src/vpp2/libero_pseudo_flow/geometry.py) | 像素采样、投影、LZ4 读写、8 帧 clip 归一化 |
| [`training.py`](../src/vpp2/libero_pseudo_flow/training.py) | pseudo-flow reader/dataset；沿用 action 清单与固定 flow 样本集合 |
| [`visualize.py`](../src/vpp2/libero_pseudo_flow/visualize.py)、[`tracks.py`](../src/vpp2/libero_pseudo_flow/tracks.py) | 投回 2D 的箭头视频与固定源点轨迹视频 |
| [`compare_gt.py`](../src/vpp2/libero_pseudo_flow/compare_gt.py) | RGB / GT / 4RC 并排可视化；只在这里读取 GT 进行比较 |
| [`codec.py`](../src/vpp2/libero_flow/codec.py) | 冻结的官方 MotionVAE 权重，geometry/flow 编解码 |
| [`model.py`](../src/vpp2/libero_flow/model.py) | motion patch、共享 Action DiT、indicator、action head 与 motion unpatch |
| [`reconstruction_codec.py`](../src/vpp2/libero_flow/reconstruction_codec.py) | 相同 codec 结构的随机初始化重建 decoder，不加载预训练权重 |
| [`inference.py`](../src/vpp2/libero_flow/inference.py)、[`evaluate.py`](../src/vpp2/libero_flow/evaluate.py) | 从训练配置恢复共享模型并直接输出 action |

外部依赖不复制进本仓库：

- [4RC 源码](https://github.com/Luo-Yihang/4RC)，本次使用 commit
  `a08ca0a5394c7c5f6a2a3423223cb98f85f84e6a`。
- [4RC 权重](https://huggingface.co/Luo-Yihang/4RC)，本次使用 revision
  `3634b324abb16634a527987c8efa212e392a1449`。
- [MotionCrafter 源码](https://github.com/TencentARC/MotionCrafter)，codec 实现参考 commit
  `9cb4e9679f5f34e249945544052464ef46324bc2`；
  [MotionCrafter 权重](https://huggingface.co/TencentARC/MotionCrafter)使用 `geometry_motion_vae/`。

`weights/`、`data/`、`runs/` 和生成的视频都被 Git 忽略。

## 2. Flow 的精确定义

只使用外部/主相机 `observation.images.image`，不从 wrist 相机生成标签。
训练 RGB 条件仍是主相机和 wrist 两个视角。

令 `P_t(u)` 为源帧像素 `u` 对应的三维物质点；标签是同一物质点的位移：

```text
F_(t,dt)(u) = P_(t→t+dt)(u) - P_t(u)
```

这里不是两个 point map 在相同像素位置相减。`FourRC.pairs()` 调用官方 motion
decoder / track head，指定源帧 query 和目标帧，输出该源点到目标时刻的位移。
目标时刻被遮挡不自动表示物质点运动无效。

实现细节：

1. RGB 按现有 LIBERO 存储方向做 `flip_x`，缩放到 `504×504` 输入 4RC。
2. 每窗口处理 16 个源帧，额外包含后续最多 4 帧；每个窗口都包含 episode 第 0 帧。
3. 点变换到预测的初始主相机 OpenCV 坐标系；向量只旋转、不平移。
4. 用共享第 0 帧的中位深度比对齐窗口尺度，点和位移乘同一个尺度。
   这是预测的相对尺度，**不是米制 GT**；也不是完整的窗口几何配准。
5. 按 GT 相同的原图像素中心约定采样到 `224×224`。保留 `dt=1` 和 `dt=4`。
6. 非有限值、非正深度或非正置信度标记无效；每帧有效覆盖不足 95% 时提取失败。
   没有利用 GT/action 做运动校正，也没有把静态背景强制置零。

模型估计的相机运动通过坐标变换剥离；实际仍可能有重建误差和背景残余运动。

## 3. 数据布局与时间对应

```text
data/libero/<suite>_no_noops_lerobot/
  videos/chunk-000/observation.images.image/episode_000000.mp4
  meta/episodes.jsonl
  pseudo_flow/4rc_v1/episode_000000/
    meta.json                 # complete_episode、坐标系、模型版本、窗口尺度等
    progress.json
    decoder_parity.json       # 开启 --parity 时，对每个 worker 首个成功 episode 检查
    frame_000000.npz.lz4
    frame_000001.npz.lz4
    ...
```

单帧 LZ4 包含未归一化 float32 `point_map[224,224,3]`、
`scene_flow_dt1/scene_flow_dt4[224,224,3]`、geometry/flow valid masks、
置信度、源/目标预测相机参数。内部是 `np.savez`，外层标准 LZ4；无 FP16 几何量化。
超出 episode 末尾的目标以零 flow、false mask 保存。

| 参数 | 当前规则 |
| --- | --- |
| action chunk | 32 个动作，`[B,32,7]` |
| motion 源帧 | `start + [0,4,8,12,16,20,24,28]`，固定 8 帧 |
| dt=1 目标帧 | `start + [1,5,9,13,17,21,25,29]` |
| dt=4 目标帧 | `start + [4,8,12,16,20,24,28,32]` |
| 两种 dt 的起点范围 | 都要求 `start+32 < episode_length`，保持相同 clip 集合 |

时间索引指 **no-op 筛选后的 LeRobot 帧**，不是筛选前固定仿真时间间隔。
GT 两视角的 `.mrec` LZ4 容器与这里逐帧 `.npz.lz4` 不同；训练选择对应 reader，
不把 pseudo 文件送入 GT 转换器，也不回退到 HDF5。

## 4. 主视角归一化、×10、latent scale 与 shape

`load_clip()` 读取 8 个源帧，只用这 8 帧**主视角有效点**计算：

```text
mu = mean(P_valid)
S  = mean(norm(P_valid - mu)) + epsilon
P_normalized = (P - mu) / S
F_normalized = F / S
```

不使用 wrist 的统计量，不给 flow 减均值，不单独计算 flow 方差，不裁剪到 `[-1,1]`。
4RC 使用自身预测点的 mu/S，不借用仿真 GT 统计量。无效点/位移在归一化后清零。
磁盘保存的是原始相对尺度数值，**没有提前乘 10 或 latent scale**。

`MotionCodec.encode()`：

```text
point map → geometry encoder → posterior mean (4 channels)
scene flow × 10 → motion encoder → posterior mean (4 channels)
concat(geometry_mean, motion_mean) × scaling_factor
```

`scaling_factor` 从权重的 `config.json` 读取，当前是 **0.18215**；
使用 posterior mean，不随机采样 posterior。decode 先除该 scale，分别解码，
motion decoder 同时接收 geometry latent，最后将输出 flow 除以 10。

| 阶段 | shape / kernel |
| --- | --- |
| reader 的 P、F | 各 `[8,224,224,3]` |
| 打包后的 motion | `[B,1,6,8,224,224]`；6 = geometry XYZ + flow XYZ |
| MotionVAE latent | `[B,8,8,28,28]`，顺序 `B,C,T,H,W` |
| 32 tokens | patch kernel=stride=`(4,7,7)`，grid=`(2,4,4)` |
| 64 tokens | patch kernel=stride=`(2,7,7)`，grid=`(4,4,4)` |
| 128 tokens | patch kernel=stride=`(1,7,7)`，grid=`(8,4,4)` |
| action 输出 | Conv1d 按 1/2/4 倍降采样 + SiLU + projection → `[B,32,7]` |

MotionVAE **空间下采样 8 倍、时间不下采样**。patch 参数按 PyTorch 的 `(T,H,W)` 排列，
无 padding。indicator `0=action, 1=flow`；两种任务共享 Action DiT。
FM 模式冻结整个 MotionVAE，loss 在缩放后的 latent 上计算；flow decode 只用于输出或
显式启用的辅助 3D loss。action 推理不需要先生成 flow。

一步重建消融使用 learnable queries → DiT → ConvTranspose3d → 随机初始化、可训练的
VAE decoder。只读 codec config，不读预训练 safetensors，也不编码 GT；未使用的 encoder
和 motion 输入 patch 保留但冻结。实验需显式设置 `flow.reconstruction_loss=mse`、
`flow.scene_flow_loss_weight=100`；后者表示 **100 × scene-flow MSE**，不是把位移乘 100。
详见[重建说明](../scripts/libero_flow/reconstruction/README.txt)。

## 5. 环境与权重

### 当前共享机器

```bash
source scripts/libero_pseudo_flow/activate.sh
export PSEUDO_PY="$PYTHON_BIN"
"$PYTHON_BIN" -c 'from arc.models.arc.arc import Arc; import av, lz4.frame; print("imports OK")'
```

现有 overlay 位于 `runs/libero_pseudo_flow_4rc_20261102/venv/`，源码通过
`../tools/4RC` 加入 PYTHONPATH。该目录不会随 Git clone 一起出现。

### 新机器/新 checkout

先按照主项目准备兼容 CUDA 的 PyTorch/torchvision 和 VPP2 环境。以下命令创建独立
overlay，不升级正在训练的 baseline 环境。`python` 应指向已准备好的基础环境：

```bash
# 若 ../tools/4RC 已存在，不要重复 clone 或覆盖正在使用的 checkout。
git clone https://github.com/Luo-Yihang/4RC.git ../tools/4RC
git -C ../tools/4RC checkout a08ca0a5394c7c5f6a2a3423223cb98f85f84e6a
python -m venv --system-site-packages runs/libero_pseudo_flow_runtime/venv
export PSEUDO_PY="$PWD/runs/libero_pseudo_flow_runtime/venv/bin/python"
"$PSEUDO_PY" -m pip install --no-deps \
  addict==2.4.0 roma==1.8.0 lz4==4.4.5 av==16.0.1 \
  opencv-python-headless==4.11.0.86 einops==0.8.1 safetensors==0.5.3
export FOUR_RC_ROOT="$(cd ../tools/4RC && pwd)"
export PYTHONPATH="$PWD/src:$FOUR_RC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PSEUDO_PY" -c 'from arc.models.arc.arc import Arc; import av, lz4.frame; print("imports OK")'
```

`--no-deps` 依赖基础环境已具备 NumPy、Pillow、huggingface_hub 等；缺失时只在独立环境
补齐。实际提取环境使用 PyTorch `2.11.0+cu128`、NumPy `1.26.4`。MotionVAE 训练 overlay
由 `bash scripts/libero_flow/setup.sh` 创建，固定 diffusers `0.31.0`。

在可联网环境下载权重（如之前 source 了离线环境，需临时关闭离线标志）：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$PSEUDO_PY" - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
repo = "Luo-Yihang/4RC"
revision = "3634b324abb16634a527987c8efa212e392a1449"
hf_hub_download(repo, "model.safetensors", revision=revision, local_dir="weights/4RC")
Path("weights/4RC/source.json").write_text(json.dumps(dict(repo=repo, revision=revision), indent=2))
snapshot_download("TencentARC/MotionCrafter", allow_patterns=["geometry_motion_vae/*"],
                  local_dir="weights/MotionCrafter")
PY
```

MotionVAE 预期文件是 `weights/MotionCrafter/geometry_motion_vae/config.json` 和
`diffusion_pytorch_model.safetensors`。提取器还会读取 `weights/4RC/source.json` 保存 provenance。
4RC 代码版本和权重版本是两个不同 revision。

## 6. 先预览，再批量提取

本节直接调用模块，不依赖历史机器 hostname。当前机器先运行：

```bash
source scripts/libero_pseudo_flow/activate.sh
export PSEUDO_PY="$PYTHON_BIN"
export FOUR_RC_ROOT="$(cd ../tools/4RC && pwd)"
```

新环境则保留上一节设置的 `PSEUDO_PY`、`FOUR_RC_ROOT`、`PYTHONPATH`。
先选择一个至少 33 帧的 episode，例如 object/0，输出到独立 preview 目录：

```bash
CUDA_VISIBLE_DEVICES=0 "$PSEUDO_PY" -m vpp2.libero_pseudo_flow.extract \
  --suite object --episode 0 --max-frames 64 \
  --checkpoint weights/4RC/model.safetensors \
  --resolution 504 --block 16 --pair-batch 4 \
  --preview-root runs/4rc_preview/object --status runs/4rc_preview/object/status.json \
  --code-revision "$(git -C "$FOUR_RC_ROOT" rev-parse HEAD)" --parity --preview-tracks

"$PSEUDO_PY" -m vpp2.libero_pseudo_flow.visualize \
  --episode-dir runs/4rc_preview/object/episode_000000 \
  --dt 4 --output runs/4rc_preview/object/flow_dt4.mp4
"$PSEUDO_PY" -m vpp2.libero_pseudo_flow.tracks \
  --episode-dir runs/4rc_preview/object/episode_000000 \
  --output runs/4rc_preview/object/tracks.mp4
```

箭头视频将 `P` 和 `P+F` 用预测相机投回 RGB，使用真实投影位移，20 fps；
显示过滤只改变画出的箭头，不改变训练标签。轨迹视频查询固定源帧像素到后续帧的位置，
不是累积二维 optical flow。`compare_gt` 模块可选，但需要已生成的仿真 GT。
部分 episode 的 preview 不允许作为完整训练 episode。

每台机器两个 GPU 处理一个 suite，用两个终端分别运行（第二个终端改为 GPU 1）：

```bash
suite=object  # 四台分别使用 spatial、object、goal、long
gpu=0        # 第二个终端设为 1；共享磁盘 episode 锁自动分配任务
CUDA_VISIBLE_DEVICES="$gpu" "$PSEUDO_PY" -m vpp2.libero_pseudo_flow.extract \
  --suite "$suite" --checkpoint weights/4RC/model.safetensors \
  --resolution 504 --block 16 --pair-batch 4 \
  --status "runs/4rc_production/$suite/gpu$gpu.json" \
  --code-revision "$(git -C "$FOUR_RC_ROOT" rev-parse HEAD)" --parity
```

命令可放进各自 tmux。每个 worker 按 episode 加锁，配置相同的已完成 episode 自动跳过；
中断后的未完成 episode 重新提取，配置不同的已有成品会报错。
检查 status 的 `failed` 和 episode `error.json`；`COMPLETE_WITH_ERRORS` 不表示全量成功。
历史 `launch_suite.sh` / `preview.sh` 绑定了旧机器或旧目录，仅用于当时的部署，不是通用入口。

## 7. 只换 flow 来源的受控训练

GT 与 4RC 必须使用同一个 `flow.available_manifest`。虽然 4RC 可对没有成功 replay 的
episode 生成标签，为公平比较，当前训练 adapter 要求实际 flow episode 集合与该清单完全一致。
缺失 pseudo-flow 会明确报覆盖不一致，不静默缩小成另一组样本。

```bash
export RUN_NAME=libero-4rc-flow32-a1of8-new
# 替换为对应 GT 实验使用的固定清单；不要随 action 比例改变该清单。
export FLOW_MANIFEST=/absolute/path/to/gt_run_flow_manifest.json
bash scripts/libero_flow/train_32tokens_1of8.sh \
  flow.motion_source=4rc_v1 \
  "flow.available_manifest=$FLOW_MANIFEST"
```

本次历史对照使用 `runs/libero_flow_submit_20261025/flow_manifest.json`。
1/4、1/2 仅替换成 `train_32tokens_1of4.sh`、`train_32tokens_1of2.sh`。
先在带 GPU 的机器给脚本加 `--dry-run` 可检查启动参数；它不加载权重或启动训练。
多节点时每节点执行一次，并按平台设置 `NNODES/NPROC_PER_NODE/NODE_RANK/MASTER_ADDR`。

控制项：32 tokens、dt4、主视角 mu/S、相同 Video-10k/Action 初始化、30,000 optimizer steps、
global batch 256、BF16/ZeRO1；16 H200 时每卡 16、梯度累积 1。GPU 数变少时自动增大累积。
每个 global batch 只有一个任务，任务按各自 batch 数量交织，**不是固定 flow/action 1:1 轮流**。
action 标签比例是 episode 覆盖率，不是 batch 混合概率；详见[采样说明](../scripts/libero_flow/SAMPLING.md)。
baseline action-only 与共享训练在相同总 step 下 action 更新次数并不相同。

## 8. 验证与推理

```bash
# CPU：投影约定、尺度/平移不变的归一化、LZ4 无损、dt1/dt4 clip 时间范围。
"$PSEUDO_PY" scripts/libero_pseudo_flow/check_geometry.py

# 独立训练环境：全尺寸 dataloader + model 检查（需要空闲且显存足够的 GPU）。
bash scripts/libero_flow/validate_single.sh --help
```

`--parity` 比较优化的 pair decoder 与相同 encoder features 上的官方 full-target decoder，
允许 BF16 数值误差；不能把该检查解释为物理 flow 精度验证。
归一化后有效点中心应接近零、平均半径接近 1；flow 数值不要求落在固定区间。

部署通过 `vpp2.libero_flow.inference` / `evaluate` 读取 checkpoint 的训练配置并恢复 task
embedding、共享 DiT、action head；固定 action chunk 32、action NFE10、逻辑 video 长度 17。
action 路径不读 flow 文件、不运行 motion decoder；不得套用原 baseline 的 action head。
已排好的具体 LIBERO-Pro 评测入口在 `scripts/libero_4rc*_pro_eval/`，其中训练路径和队列是
历史实验专用的，新 checkpoint 需重新准备配置。
