Motion view policy updated2026-10-25: default flow.motion_cameras=[image].
RGB remains dual-view; latent channels=8, decoded GT=[B,1,6,8,224,224].
The historical validation results below predate this change unless stated otherwise.

2026-10-27 production MSE ablation:
  32tokens x action1/8,1/4,1/2; explicitly pass flow.reconstruction_loss=mse.
  MotionVAE config only, random trainable decoders; learnable queries; one-pass
  point-map/scene-flow masked MSE. Action still uses FM and10-step inference.
  Head-only mu/S normalization, ALL valid flow, LZ4 records, homogeneous global
  task batches match the corrected Flow32 FM runs. 2x8H200, micro16, GAS1,
  global256,30000steps,priority10. Existing default L1 remains for old commands.
  Resolved-config comparisons and submission records:
    runs/libero_recon32_mse_submit_20261027/

2026-10-27 scene-flow weighting update:
  The three original MSE jobs were stopped and replaced by fresh mse100 runs.
  Explicit override: flow.scene_flow_loss_weight=100 (default1 preserves old runs).
  L_F = 0.5 * (masked_MSE(point) + 100 * masked_MSE(scene_flow)).
  This multiplies the MSE, not the flow vector, by100. Normalization/codec scales,
  validity masks and action FM remain unchanged. Model size is unchanged.
  loss_scene_flow remains RAW; loss_scene_flow_weighted logs the weighted term.
  Both enter loss_flow with the existing overall factor0.5.
  Checkpoints store the weight and reject a mismatched weight on weights reload.
  Commands, config diff, actual-GT gradient/mask check and replacement jobs:
    runs/libero_recon32_mse100_submit_20261027/

一步重建消融（2026-10-23）

目的
  保持已有共享 ActionDiT 的模块尺寸不变，将 flow 任务改为 queries -> 一步3D重建。
  action 任务继续原来的 flow matching 和10步推理。原 latent-FM 默认入口不变。

模型
  共用原40层、hidden_dim=512的 ActionDiT、冻结的视频条件模型、语言和 proprio 条件。
  indicator=0: noisy actions[B,32,7] -> action encoder ->32/64/128 tokens
               ->共享DiT ->原Conv1d+SiLU+Conv1d action head ->[B,32,7] velocity。
  indicator=1: learnable queries[1,N,512]扩展到batch，加入原position和indicator
               ->共享DiT（固定t=0）->原ConvTranspose3d ->[B,8,8,28,28]
               ->随机初始化、可训练的原VAE解码结构 ->[B,1,6,8,224,224]。
  最后6通道为point map XYZ + scene flow XYZ，仍为原loader归一化坐标。
  flow不加噪声、不采样sigma、不做速度预测、不做10步迭代、不编码GT。
  视频条件分支仍保留原来的冻结video noise机制；这是对motion输入的消融。
  输出反卷积THW kernel=stride为(4,7,7)/(2,7,7)/(1,7,7)，padding=0。
  N=32/64/128；action head序列下采样倍率=1/2/4。

参数控制
  唯一新参数为motion_queries：16,384 / 32,768 / 65,536。
  MotionCodec原模块的键名和形状全部保留，编码器保留但冻结且不调用。
  原motion_patch输入卷积保留但冻结且不调用，因为输入已是queries。
  重新从config构造VAE，完全不读取MotionVAE safetensors预训练权重；两个decoder
  及post_quant_conv随机初始化并训练。没有VAE KL、重参数采样或latent重建目标。
  保留原固定scaling_factor和flow /10输出尺度，以免额外改变坐标约定。
  因decoder由冻结改为可训练，可训练参数数量及优化器显存会增加；总模型参数
  数量只增加queries。decoder使用activation checkpointing，不改变架构和参数。
  原codec共167,326,158个参数，其中98,998,830个decoder/post_quant参数在新实验
  中参与训练；还需扣除不再训练的motion_patch参数，才能算净可训练参数变化。

监督
  默认flow.reconstruction_loss=l1，可设mse。
  每个flow样本分别对有效point像素和有效flow像素计算通道均值：
    L_F = 0.5 * (masked_L1(point_pred, point_GT) + masked_L1(flow_pred, flow_GT))。
  每项按自身有效像素数归一化；全无效时为0。GT仅在预测完成后进入loss。
  action损失公式、噪声调度、padding mask与原模型相同。
  总loss=(sum(action FM损失)+sum(flow重建损失))/实际batch样本数。
  未额外人为放大flow损失；不同任务的数值尺度并不因此保证相同。
  这组对照同时改变motion输入、目标和decoder初始化/训练状态，不能将其结果
  解释成“仅更换loss函数”的独立因果效果。

数据与训练控制
  2026-10-24起：保留seed42 action1/8、1/4、1/2名单，flow_episode_scope=all。
  flow使用全部有完整flow数据的episode，包括action名单内的episode。被选中episode分别贡献
  action任务样本与flow任务样本，每个样本仍只有对应任务的标签。
  三个比例的flow集合和可用start相同；均匀抽样方式不变，action标签覆盖率
  不等于batch任务占比。实际采样占比记录在dataset_audit.json。
  如需复现旧分区，显式设置flow.flow_episode_scope=complementary。
  新旧数据分区不能续用同一sampler/optimizer状态，应新建实验目录。
  motion源帧0,4,...,28，dt=1和dt=4仍各8帧，数据归一化/有效区域/尾部规则不变。
  复用原dataset、collator、trainer、AdamW、LR1e-4、warmup500、seed42、BF16、ZeRO1。
  30000 optimizer steps、global batch256；按GPU数量分配microbatch及累积步数。
  workers默认每卡12。2026-10-25起，生产默认flow.missing_motion=skip：
  缺少/未完成flow的episode仅跳过flow任务；action仍严格使用原筛选名单，
  即使该episode的replay失败也保留其action训练样本，不按flow成功名单过滤action。
  跳过发生在dataset构建阶段，总batch仍256。正式启动会冻结可用flow清单，
  保证各rank及resume索引一致；dataset_audit.json记录跳过名单和数量。
  可显式设flow.missing_motion=error恢复严格检查。损坏文件的读取异常不静默忽略。
  验证记录：runs/libero_flow_implementation/skip_missing_flow_20261025/。
  同一fraction不是每个batch严格同一比例；仍按原可用start均匀采样。

运行（仓库根目录）
  bash scripts/libero_flow/reconstruction/train_32tokens_1of8.sh --dry-run
  FLOW_DT=4 bash scripts/libero_flow/reconstruction/train_64tokens_1of4.sh
  FLOW_DT=1 bash scripts/libero_flow/reconstruction/train_128tokens_1of2.sh
  bash scripts/libero_flow/reconstruction/train_32tokens_1of8.sh flow.reconstruction_loss=mse
  所有32/64/128 × 1of8/1of4/1of2都有脚本；每次仅启动指定的一组实验。
  输出默认runs/libero_flow/recon_allflow_n{N}_a1of{denominator}_dt{dt}，不会混用原FM目录。
  对同一组合换loss或重复种子时，请设置独立RUN_NAME；默认拒绝覆盖已有输出。
  --micro-batch只改变microbatch和累积，global batch仍为256。

验证与checkpoint
  source scripts/libero_flow/activate_env.sh
  "$PYTHON_BIN" -m vpp2.libero_flow.validate_reconstruction
  bash scripts/libero_flow/reconstruction/validate_distributed.sh
  新checkpoint格式vpp2.libero_flow.reconstruction.v1，包含queries、随机codec权重、
  action expert、proprio encoder和adapters。与原latent-FM checkpoint严格区分。
  诊断evaluator自动选择flow一步/action十步，保留相同3D RMSE指标。
  实测记录：runs/libero_flow_implementation/reconstruction/。

实测结果（2026-10-23，完整模型、实际LIBERO数据）
  以下GPU验证使用当时的互补标签分区。2026-10-24的数据分区修改另见
  runs/libero_flow_implementation/all_flow/validation.json；模型和loss未改。
  48GiB4090，batch1：32/64/128 tokens均通过两分支前向、反向、推理及参数核对；
  峰值allocated显存42.51/44.13/44.39GiB（无优化器状态，不代表训练容量上限）。
  checkpoint成功恢复随机decoder和queries；旧视频骨干/保留编码器没有梯度。
  2H200，128tokens：每卡micro16、累积8次、global256，完整optimizer step通过；
  每卡峰值allocated95.84GiB。两个rank五组监测参数梯度均有限且非零；包含
  action缺席的microbatch，分布式梯度同步正常。未启动30000步正式训练。
  首步约175.75秒，其中取数等待约8.46秒；这是含预热的诊断，不是稳定吞吐。
  L1已做完整GPU验证；MSE已做数值/mask单元检查，未单独开展完整训练。

此前NPZ/workers测试结论
  2026-10-25起默认flow.motion_backend=records，严格要求有效LZ4缓存，
  不会静默回退HDF5。缺少GT仍只跳过flow样本；有GT但缓存缺失/损坏时
  明确报错，需要转换或修复缓存。可用flow.prefetch_factor=1/2/4调预取深度，
  含义为每个worker预取的batch数；num_workers为每张卡的worker数。
  保留现有LZ4 records缓存；本次NPZ未超过LZ4，已记录读取与GPU实测及其限制。
  详见runs/libero_flow_implementation/npz_workers/REPORT.txt。
  原baseline/MeanFlow目录及运行任务不在本次修改范围。

2026-10-25 LZ4/prefetch实测
  记录：runs/libero_flow_implementation/prefetch_20261025/REPORT.txt。
  当前一步重建模型128tokens、2H200、global batch256，prefetch1/4对比：
  去掉首步后88.587/88.810秒每step，等待数据仅0.071/0.065%。
  因而默认维持12workers/prefetch1；2可增加读取余量，4未带来训练加速。
  这是16个episode的短诊断；不等于全量长期吞吐保证。
