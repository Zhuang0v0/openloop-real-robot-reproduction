# Classify Objects: FastWAM 与 Efficient-WAM-RT 微调

本仓库只实现 **Classify Objects** 一个真机任务。代码从 RoboTwin 预训练
checkpoint 开始，在 ModelScope 的 `XinyaoZhuang/nero_classify_objects`
数据上做微调，然后用独立的真机执行流程评测。数据、模型权重、缓存和
运行结果都在仓库外保存；本 README 不包含任何集群连接或凭据信息。

## 1. 实验合同

唯一使用的 Excel 工作表名称必须是 `classify objects`（大小写和空格均须
一致）。该表应包含 106 条记录，对应 episode `0..105`，场景组数量为
`C1/C2/C3/C4 = 27/26/27/26`。不得读取或混入其他工作表：
`build block tower`、`pack & pour fruit`、`实物实验任务`。

所有 episode 使用同一个 task label，不按 C1--C4 改写 prompt：

```text
Put all fruits into the large basket first, then put all blocks into the small basket.
```

C1--C4 只作为场景 metadata，用于分层划分和报告。状态和动作均为 16D，
通道顺序由数据审计报告确定；不要把 14D RoboTwin checkpoint 直接加载到
16D 数据。

默认采用 **episode-disjoint 80/20 train/validation split**，两个模型读取
同一份 split，避免一个 episode 的重叠窗口泄漏到验证集。需要把 106 条都
用于最终训练时，可设置 `CLASSIFY_VAL_FRACTION=0`；此时没有 offline
validation，训练 loss 不能称为 validation loss。

## 2. 目录和环境变量

所有路径都可按机器调整：

```bash
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/classify_objects}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export EXCEL_PATH="${EXCEL_PATH:-$PROJECT_ROOT/task.xlsx}"
export DATASET_ROOT="${DATASET_ROOT:-$DATA_ROOT/nero_classify_objects}"
export EFFICIENT_WAM_ROOT="${EFFICIENT_WAM_ROOT:-$PROJECT_ROOT/vendor/Efficient-WAM}"
export FASTWAM_ROOT="${FASTWAM_ROOT:-$PROJECT_ROOT/vendor/FastWAM}"
mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$RUN_ROOT"
```

`DATASET_ROOT` 应是解压后直接包含 `data/` 与 `videos/` 的目录。模型权重
和 WAN/T5/VAE 缓存只放在 `MODEL_ROOT` 或其他数据盘，不要放进 Git。

## 3. 获取代码、数据和模型

```bash
git clone --branch codex/classify-objects-finetuning --single-branch \
  https://github.com/Zhuang0v0/openloop-real-robot-reproduction.git
cd openloop-real-robot-reproduction
```

安装与上游版本匹配的 CUDA/PyTorch 环境。Efficient-WAM 的依赖模板在
`vendor/Efficient-WAM/envs/train.yaml`，本仓库额外需要 `pyarrow`、
`safetensors`、`opencv`、`ffmpeg/decord` 和 `requests`。FastWAM 使用其
自己的环境，路径通过 `FASTWAM_ROOT`、`FASTWAM_PYTHON` 指定：

```bash
conda env create -f vendor/Efficient-WAM/envs/train.yaml
conda activate EfficientWAM
"$PYTHON_BIN" -m pip install -r adapters/efficient_wam/requirements.txt
bash scripts/clone_fastwam.sh
"$PYTHON_BIN" -m venv "$FASTWAM_ROOT/.venv"
"$FASTWAM_ROOT/.venv/bin/python" -m pip install -e "$FASTWAM_ROOT"
export FASTWAM_PYTHON="$FASTWAM_ROOT/.venv/bin/python"
```

Efficient-WAM 和 FastWAM 必须使用各自独立的 Python 环境；不要把一个环境
的 `site-packages` 通过 `PYTHONPATH` 注入另一个环境。Efficient-WAM 的
`torch` 与 `flash-attn` 必须是同一 CUDA/PyTorch ABI，并应在已分配 GPU 的
节点上预检：

```bash
python -c "import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__); assert torch.cuda.is_available()"
```

如果出现 `flash_attn_2_cuda...undefined symbol`，说明环境不匹配；请重新
激活由 `vendor/Efficient-WAM/envs/train.yaml` 创建的环境并安装匹配的
`flash-attn`，不要用旧环境中的 wheel 继续训练。`easydict`、`deepspeed`、
`pyarrow` 和 `decord` 也必须按依赖文件安装。登录节点没有 GPU 时可以做
配置/语法检查，但不能把导入探针或 CPU 检查称为 GPU smoke。

下载 ModelScope 的 `per_episode` 导出（SDK token 只通过环境变量或交互式
输入提供）：

```bash
"$PYTHON_BIN" -m pip install modelscope
"$PYTHON_BIN" scripts/download_modelscope_dataset.py \
  --repo XinyaoZhuang/nero_classify_objects \
  --output-dir "$DATA_ROOT/modelscope"
```

将生成的 `*_per_episode.tar` 解压到数据盘，并把 `DATASET_ROOT` 指向其中
包含 `data/`、`videos/` 的目录。公开 checkpoint 的下载地址和文件校验由
仓库中的 `download_*_checkpoint.py` 脚本处理；例如：

```bash
"$PYTHON_BIN" scripts/download_fastwam_checkpoint.py \
  "$MODEL_ROOT/robotwin_uncond_3cam_384.pt"
"$PYTHON_BIN" scripts/download_efficientwam_checkpoint.py \
  "$MODEL_ROOT/Efficient-WAM-RT_stage3.pt" \
  --stats-output "$MODEL_ROOT/Efficient-WAM-RT_action_stats.json"
```

FastWAM 还需要官方 `dataset_stats.json`；Efficient-WAM-RT 需要 WAN-2.2
TI2V-5B、VAE 和 UMT5 文件。按各自上游模型卡下载并设置缓存目录，脚本
不会把这些大文件上传到 GitHub。

## 4. Excel 映射与数据审计

先用 Excel 的“对应编号”匹配 episode，不能按文件名排序猜标签：

```bash
"$PYTHON_BIN" tools/match_classify_excel.py \
  --excel "$EXCEL_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$RUN_ROOT/mapping" \
  --sheet-name "classify objects" \
  --episodes 106
```

确认 `classify_episode_manifest.csv`、`classify_unmatched_episodes.csv` 和
`classify_dataset_audit.md` 后，再执行结构审计：

```bash
"$PYTHON_BIN" tools/audit_classify_dataset.py \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$RUN_ROOT/mapping/classify_episode_manifest.csv" \
  --output-dir "$RUN_ROOT/audit" --probe-videos
```

审计必须核对 episode 完整性、parquet schema、16D finite state/action、
时间戳、三路相机和实际视频 FPS。缺失、损坏或未匹配记录先停下处理；不
要让训练脚本静默跳过。`？` 质量分保留为 `unknown`，不能当成 0 或 1。

## 5. 动作质量与四种筛选策略

原始轨迹永远不覆盖。动作分析使用稳健的逐维速度、加速度、jerk、尖峰、
停顿、夹爪转换以及 state/action 对齐检查：

```bash
"$PYTHON_BIN" tools/analyze_classify_actions.py \
  "$DATASET_ROOT" \
  --output "$RUN_ROOT/action_analysis/report.json" \
  --quality-csv "$RUN_ROOT/action_analysis/classify_action_quality.csv" \
  --anomaly-csv "$RUN_ROOT/action_analysis/classify_anomaly_segments.csv" \
  --report "$RUN_ROOT/action_analysis/classify_quality_report.md" \
  --plot-dir "$RUN_ROOT/action_analysis/plots"
```

训练入口支持四个可比较策略：

| 策略 | 含义 |
|---|---|
| `baseline` | 结构有效且映射可靠的全部窗口 |
| `quality` | 按 Excel 质量分筛选，可比较 0.7/0.8/0.9 |
| `action` | 删除严重动作异常的完整窗口 |
| `combined` | 同时满足人工质量和动作条件 |

窗口筛选不会删除轨迹中间帧再拼接前后动作；每个窗口保留原始连续帧、
source row、padding/mask 和 episode 边界。`filter_classify_windows.py`
会输出窗口 JSON/CSV、`classify_action_quality.csv`、异常段和 Markdown
报告，并按 episode（可按 C1--C4 分层）生成统一 split。

## 6. 14D checkpoint 到 16D

两个 RoboTwin checkpoint 都需要先做显式通道适配：

```bash
MODEL_ROOT="$MODEL_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/adapt_fastwam_checkpoint.sh
MODEL_ROOT="$MODEL_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/adapt_efficient_checkpoint.sh
```

适配脚本生成带校验报告的 16D 权重；零初始化的新增夹爪通道只在真机
微调中学习。先检查报告和 checkpoint shape，再启动训练。

## 7. FastWAM 微调

FastWAM 使用上游 `dit_only` 训练模式：冻结 WAN/VAE 等非 DiT 组件，但同时
训练 video DiT、action DiT 和 proprio encoder；默认 batch 1、gradient
accumulation 8、起始学习率 `5e-5`、最多 2000 steps。可通过
环境变量覆盖，但两个模型应保留各自的配置：

```bash
export CLASSIFY_MANIFEST="$RUN_ROOT/mapping/classify_episode_manifest.csv"
export FASTWAM_CHECKPOINT="$MODEL_ROOT/robotwin_uncond_3cam_384_16d.pt"
export CLASSIFY_FILTER_STRATEGY=baseline   # baseline|quality|action|combined
export CLASSIFY_VAL_FRACTION=0.20          # 设为 0 表示全部 episode 训练
bash scripts/train_fastwam_classify.sh
```

脚本会先生成 action statistics、ActionDiT backbone（缺失时调用上游
预处理脚本），再通过 `adapters/fastwam/run_with_manifest.py` 启动上游
Hydra 训练。训练和验证 manifest 明确按 episode 过滤；不要依赖上游随机
`val_set_proportion`。

## 8. Efficient-WAM-RT 微调

默认先进行 Stage 2：冻结 WAN/video expert，只更新 action expert；在 Stage
2 checkpoint 能前向、反向并保存/加载后，才考虑可选的短 Stage 3 joint
continuation。默认 action learning rate 为 `5e-6`，batch 1、累积 8。

```bash
export CLASSIFY_MANIFEST="$RUN_ROOT/mapping/classify_episode_manifest.csv"
export EFFICIENT_WAM_ROOT="$PROJECT_ROOT/vendor/Efficient-WAM"
export WAN_MODEL_ROOT="$MODEL_ROOT/Wan2.2-TI2V-5B"
export CLASSIFY_FILTER_STRATEGY=baseline
export CLASSIFY_VAL_FRACTION=0.20
bash scripts/train_efficientwam_classify.sh
```

Efficient-WAM-RT 的步数必须区分：本项目微调配置显式使用 **10 个训练
flow-matching bins**；RT 推理仍是 **10 个 action inference steps**，但
video branch 只在 `[0, 1]` 两个 step refresh。不要把“训练十步”改写成
“视频推理十步”，也不要把 RT 的两次视频 refresh 当作训练 schedule。
该 10-bin 设置是本项目工程配置，必须在目标 GPU 上 smoke 验证，不声称是
论文统一结论。

## 9. Smoke、评估和报告

正式训练前至少提交一个小规模 GPU smoke，验证前向、loss、反向、checkpoint
保存/导出以及 16D shape。没有 GPU 或缺少模型缓存时，明确记录为未执行，
不能把脚本启动成功称为训练成功。

```bash
# Uses the first matched episode and one optimizer step; this is a smoke test,
# not a completed fine-tuning run.
export CLASSIFY_MANIFEST="$RUN_ROOT/mapping/classify_episode_manifest.csv"
bash scripts/smoke_efficientwam_classify.sh

export DATASET_ROOT="$DATASET_ROOT"
export CLASSIFY_EVAL_CHECKPOINT="$RUN_ROOT/.../checkpoint.pt"
bash scripts/evaluate_classify.sh
```

离线预测误差、训练 loss 和 validation loss 都不能称为真机任务成功率。
真机评测应使用未参与微调的执行回放，并分别报告 C1--C4、碰撞、超时和
完成率。记录有效 episode/window 数、GPU 显存、训练步数、最优 checkpoint
及所有未解决的单位、FPS、文本条件和动作空间假设。

## 10. 目录约定与 Git 边界

```text
configs/                  Classify Objects 合同和筛选配置
tools/                    Excel 映射、审计、动作分析、窗口准备
adapters/fastwam/         FastWAM manifest 适配器
adapters/efficient_wam/   Efficient-WAM 转换器与 16D 配置
scripts/                  下载、适配、训练和评估入口
vendor/Efficient-WAM/     上游 Efficient-WAM 源码副本（含训练所需 models/*.py）
reports/                  调研、审计、训练计划和质量报告模板
tests/                    不依赖大数据/权重的单元测试
```

不提交数据集、视频、parquet、模型权重、optimizer state、缓存、日志、
token、密码、TOTP、个人机器路径或集群连接文档。上游项目的许可证和模型
使用条款仍然适用。
