# 4RC pseudo-flow 脚本

完整的提取定义、依赖/权重、归一化、MotionVAE、预览、双卡生成和训练接入命令见
**[4RC → MotionVAE → 共享 Action DiT 指南](../../docs/libero_pseudo_flow.md)**。

- `activate.sh`：激活本机现有的隔离环境，不安装依赖。
- `check_geometry.py`：CPU 几何、归一化、LZ4、clip 时间对齐检查。
- `preview.sh` / `worker.sh`：历史部署包装脚本，含本机绝对路径。
- `launch_suite.sh`：历史四台实例的 hostname 检查及两 GPU tmux 启动，不能直接用于新实例。
- `validate_previews.py` / `status.py`：历史运行目录的预览验收和状态统计。

新机器请使用指南中的模块命令，避免历史 hostname、下载临时文件或报告路径依赖。
`check_checkpoint_header.py` 是当时分块下载阶段的 header 检查，不是通用安装步骤；
实际提取通过 `FourRC` 的 strict state-dict 加载和 `--parity` 验证模型。
