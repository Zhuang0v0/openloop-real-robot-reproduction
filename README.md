# Efficient-WAM-RT 与 Fast-WAM 真机微调

本仓库只保留两个模型的可复现实验代码：

- **Efficient-WAM-RT**：RoboTwin checkpoint 的 14D -> 16D 适配、数据打包、Stage 2/3 微调。
- **Fast-WAM**：RoboTwin checkpoint 的 14D -> 16D 适配、统计量生成、真机数据微调。

完整步骤见 [`README_MODEL_TRAINING.md`](README_MODEL_TRAINING.md)。文档假设代码会在一台有 CUDA 的 Linux GPU 机器上运行；所有数据、权重和运行目录都通过环境变量指定，不依赖某个用户的本机或服务器路径。

## 快速开始

```bash
git clone https://github.com/Zhuang0v0/openloop-real-robot-reproduction.git
cd openloop-real-robot-reproduction

export PROJECT_ROOT="$PWD"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
```

然后按主训练说明的顺序执行：

1. 安装 Efficient-WAM 和 Fast-WAM 的 Python/CUDA 环境。
2. 下载 `XinyaoZhuang/nero_building_block_tower` 数据集，并验证 100 个 episode。
3. 下载对应的 RoboTwin checkpoint 和 WAN-2.2/T5/VAE 依赖。
4. 适配 16D checkpoint，生成 Efficient-WAM packed dataset 和 Fast-WAM normalization stats。
5. 先做短 smoke，再提交正式微调。
6. 训练完成后只在真机上做独立评测，不从训练 episode 划分离线 eval 集。

## 目录

```text
adapters/efficient_wam/   Efficient-WAM 数据转换、16D 配置模板
scripts/                  下载、校验、适配、预处理和 smoke 脚本
vendor/Efficient-WAM/     上游 Efficient-WAM 源码副本（含许可证）
README_MODEL_TRAINING.md  从下载到训练和评测的完整操作说明
```

权重、视频、parquet、packed shards、optimizer state、缓存和运行日志均被 `.gitignore` 排除，不应提交到 Git。上游代码和模型的版权、许可证以各自仓库为准。
