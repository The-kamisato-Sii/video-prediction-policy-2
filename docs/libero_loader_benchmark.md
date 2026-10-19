# LIBERO 8 H200 数据管线实测（2026-10-19）

在 `vpp2-freeze-8h200-1016-0240` 的 `qb-prod-gpu910` 节点顺序运行六组真实 8 卡 ZeRO1 训练。
单卡 batch 32、累积 1、全局 batch 256，BF16。每组独立初始化并运行 56 步，统计第 9–56 步。
学习率 1e-4、warmup 500、30k 步 cosine 计划保留；`BenchmarkTrainer` 在优化器和 scheduler 初始化后把测试停止点设为 56。
关闭评估及 checkpoint 写入，保留离线 W&B。每秒采样全部 GPU；性能 profiler 不插入额外 CUDA 同步。

| 数据处理 | workers/卡 | 秒/步 | P95 秒/步 | GPU 平均利用率 | 平均功耗 W/卡 | 数据等待 ms/步 | CPU 使用核数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始整段图像处理 | 16 | 14.204 | 117.070 | 55.0% | 311 | 8894.59 | 140.4 |
| 仅首帧，保留类型往返 | 16 | 4.768 | 4.947 | 98.6% | 660 | 2.19 | 20.3 |
| 首帧＋减少类型转换 | 4 | 4.756 | 4.933 | 98.8% | 661 | 2.17 | 19.8 |
| 首帧＋减少类型转换 | 8 | 4.761 | 4.929 | 98.3% | 660 | 2.28 | 20.1 |
| 首帧＋减少类型转换 | 16 | 4.769 | 4.915 | 98.8% | 660 | 2.26 | 20.5 |
| 首帧＋减少类型转换 | 20 | 4.772 | 4.941 | 98.4% | 659 | 2.36 | 21.0 |

首帧处理是主要提速来源；进一步减少类型转换没有测出额外的训练吞吐收益。workers 4/8/16/20 速度接近，推荐每卡 4 个，减少数据进程数量。小于 1% 的差别不应视作确定的性能优势。
原始流程和两个优化的数据输出已对四个子集、边界样本共 9 个位置做逐元素比较，首帧像素、action、proprio、context 等一致。这里只验证短程性能及有限 loss，不代表长程收敛或闭环成功率结论。
数据等待是每步各 rank 的最大等待时间均值，包含 Accelerate DataLoader 及其传输路径；GPU 利用率为统计区间全部卡的一秒采样平均。
完整聚合结果：[JSON](validation/libero_loader_benchmark_8h200.json)。原始日志、逐 rank profile、GPU 采样、CSV 和 W&B 保存在 `runs/libero_loader_benchmark/`。

## 使用首帧与 uint8 优化

按 [环境设置](libero_16h200.md) 激活 vpp2，在单机 8 H200 上运行：

```bash
export RUN_NAME=libero-action-8h200-firstframe-w4
export VPP2_VIDEO_DECODE_UINT8=1
bash scripts/libero/train_8h200_zero1_first_frame.sh num_workers=4
```

`uint8` 优化是显式开关，仅支持当前 PyAV 数据后端。它保留解码后的 uint8，省掉中间 float→uint8 往返，仍在原有 ToTensor 位置转 float；resize 与归一化顺序不变。
验证：`PYTHONPATH=src python checks/libero_first_frame_equivalence.py --uint8 --output /tmp/libero-uint8.json`。

## 重跑有界性能测试

在同一 conda/CUDA 环境设置如下变量；测试入口只运行 56 步，不保存最终 checkpoint：

```bash
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export NNODES=1 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export VPP2_TRAIN_MODULE=checks.libero_loader_benchmark
export VPP2_ACCELERATE_CONFIG=configs/libero/accelerate_zero1.yaml
export BATCH_SIZE=32 GRAD_ACC=1 NUM_WORKERS=4 OMP_NUM_THREADS=2
export VPP2_VIDEO_DECODE_UINT8=1
bash scripts/libero/train_8h200.sh \
  output_dir=runs/libero/benchmark_uint8_w4_repeat overwrite=false \
  max_steps=30000 warmup_steps=500 log_every=8 eval_every=0 \
  save_every=0 save_final_checkpoint=false \
  data.train.decode_only_first_video_frame=true data.train.processor.num_image_obs_steps=1 \
  profiling.enabled=true profiling.skip_steps=8 profiling.collect_steps=48 \
  profiling.synchronize_cuda=false profiling.log_every=8
```

原始流程（当前默认已优化）：设置 `VPP2_VIDEO_DECODE_UINT8=0`，显式覆盖 `data.train.decode_only_first_video_frame=false data.train.processor.num_image_obs_steps=65`，workers=16。
仅首帧：保留首帧覆盖项但设置 `VPP2_VIDEO_DECODE_UINT8=0`，workers=16。
worker 扫描：保留两种优化，分别设置 `NUM_WORKERS=4/8/16/20`。
