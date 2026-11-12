# `classify_objects` 微调流程说明

本文解释本仓库如何把 Classify Objects 真机数据送入 FastWAM 和
Efficient-WAM-RT 微调。它是代码和数据流的操作说明，不是训练结果报告。
只有在目标 GPU 节点上完成数据审计、smoke、训练和 checkpoint 校验后，才
能把一次运行记为成功。数据、权重、缓存和运行产物均应放在仓库之外。

## 1. Quick Start

下面是从一个新工作区到正式训练的最短顺序。先定义一次 Excel 路径；后续
命令都引用 `$EXCEL_PATH`，不会把某台机器的文件名写死：

```bash
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export EXCEL_PATH="${EXCEL_PATH:-$PWD/task.xlsx}"
test -f "$EXCEL_PATH"
```

如果工作簿不叫 `task.xlsx`，只修改 `EXCEL_PATH`，不要改脚本源码。然后按
第 3、4、5 节完成硬件/环境、目录、数据和模型准备，再执行：

```bash
export CLASSIFY_MANIFEST="$RUN_ROOT/mapping/classify_episode_manifest.csv"

"$PYTHON_BIN" tools/match_classify_excel.py \
  --excel "$EXCEL_PATH" --dataset-root "$DATASET_ROOT" \
  --output-dir "$RUN_ROOT/mapping" --sheet-name "classify objects" \
  --episodes 106
"$PYTHON_BIN" tools/audit_classify_dataset.py \
  --dataset-root "$DATASET_ROOT" --manifest "$CLASSIFY_MANIFEST" \
  --output-dir "$RUN_ROOT/audit" --probe-videos
"$PYTHON_BIN" tools/analyze_classify_actions.py "$DATASET_ROOT" \
  --output "$RUN_ROOT/action_analysis/report.json" \
  --quality-csv "$RUN_ROOT/action_analysis/classify_action_quality.csv" \
  --anomaly-csv "$RUN_ROOT/action_analysis/classify_anomaly_segments.csv" \
  --report "$RUN_ROOT/action_analysis/classify_quality_report.md"

MODEL_ROOT="$MODEL_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/adapt_fastwam_checkpoint.sh
MODEL_ROOT="$MODEL_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/adapt_efficient_checkpoint.sh

# First run the one-episode Efficient-WAM GPU smoke (see section 11).
bash scripts/smoke_efficientwam_classify.sh

# After the smoke is recorded as passing, launch each model separately.
bash scripts/train_fastwam_classify.sh
bash scripts/train_efficientwam_classify.sh
```

`train_*` 会再次生成窗口和 split，这是有意设计的：它们从当前 manifest 和
环境变量重建可追溯的中间产物。若只想重跑审计或 smoke，可单独执行相应
命令，不要把失败的半成品 checkpoint 当成正式结果。

## 2. 任务合同

代码只读取 Excel 的 `classify objects` 工作表，默认期望 106 个 episode，
episode ID 为 `0..105`，场景组数量为 `C1/C2/C3/C4 = 27/26/27/26`。
四个组描述物体和篮子的空间布局，不会改变语言条件。所有样本都使用同一
条指令：

```text
Put all fruits into the large basket first, then put all blocks into the small basket.
```

数据合同如下：

| 项目 | 约定 |
|---|---|
| 相机 | `observation.images.camera_3`、`observation.images.left_wrist`、`observation.images.right_wrist` |
| 状态列 | `observation.state` |
| 动作列 | `action` |
| 状态/动作维度 | 均为 16D、同一 episode 内行数相同 |
| 采样频率 | 30 Hz（以审计结果为准） |
| 标签来源 | Excel 的“对应编号”与 episode ID 显式匹配；绝不按文件名排序猜标签 |

`C1`--`C4` 是 metadata，可用于分层报告；不能据此生成四种不同的
prompt。任何缺失 episode、重复 ID、非 finite 数值、损坏视频或维度不符都
应在训练前停止处理。

## 3. 硬件和软件要求

### 3.1 GPU、CPU 和磁盘

本项目的默认配置按 **NVIDIA A100 单卡**设计：batch size 为 1，使用
gradient accumulation 模拟更大的有效 batch，并使用 bfloat16。A100 应支持
当前安装的 CUDA/PyTorch 组合和 BF16；在作业节点先运行：

```bash
nvidia-smi
"$PYTHON_BIN" -c \
  "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, 'GiB'); print('bf16=', torch.cuda.is_bf16_supported()); assert torch.cuda.is_available()"
```

仓库没有宣称一个经过实测的最低显存数。Efficient-WAM 会同时加载 WAN/video
组件、VAE/latent 和 action expert，正式显存需求取决于 CUDA、flash-attn、
视频尺寸和缓存；若 A100 规格或余量不同，应先用一集 smoke 测量，而不是
擅自把 batch size 调大。当前 launcher 是单进程单卡，不会自动申请或同步多
卡；多卡需要另外改写 launcher 和分布式配置，不能仅设置 `CUDA_VISIBLE_DEVICES`
就宣称使用了多卡。

在共享节点上应显式选择调度器分配给自己的卡，例如：

```bash
export CUDA_VISIBLE_DEVICES=0
export EFFICIENT_DEVICE=cuda:0
export FASTWAM_DEVICE=cuda
```

这里的 `0` 只是作业内可见编号示例；以调度器实际分配的编号为准。

预处理和视频转码建议使用至少 8 个逻辑 CPU 核、32 GiB 主机内存（这是实践
建议，不是代码硬限制），并让 `num_workers` 与机器资源匹配。磁盘至少要能
同时容纳：原始 parquet/视频、FastWAM 约 12 GB checkpoint、Efficient-WAM-RT
约 2 GB checkpoint、WAN/T5/VAE、Efficient mosaic 视频、safetensors shards、
日志和多个 checkpoint；转换和打包会产生数据副本，建议在数据盘预留原始
数据体积的 2--3 倍临时空间。训练前检查：

```bash
df -h "$DATA_ROOT" "$MODEL_ROOT" "$RUN_ROOT"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

### 3.2 Python、CUDA 和关键版本

* Efficient-WAM 环境由 `vendor/Efficient-WAM/envs/train.yaml` 定义：
  Python 3.10、PyTorch 2.7.1、torchvision 0.22.1，以及 transformers、
  deepspeed、decord、easydict、flash-attn 等依赖。
* FastWAM 使用独立环境。当前建议的目标组合是 **PyTorch 2.7.1 + CUDA
  12.8（wheel 标记 `2.7.1+cu128`）**，但 wheel 必须与节点驱动兼容；不能
  把这个环境的包复制到 Efficient-WAM 环境。
* 两个环境都应使用 Linux、bash、Python 3.10 左右版本；需要可执行的
  `ffmpeg`，以及能读取 parquet 的 `pyarrow`。
* `flash-attn` 必须和实际 PyTorch/CUDA ABI 匹配。`undefined symbol`、
  `no kernel image` 或导入时 CUDA 初始化失败都属于环境问题，先修环境再
  训练。

版本信息必须写入实验记录：

```bash
"$PYTHON_BIN" -c "import torch; print(torch.__version__, torch.version.cuda)"
"$FASTWAM_PYTHON" -c "import torch; print(torch.__version__, torch.version.cuda)"
"$PYTHON_BIN" -m pip show flash-attn deepspeed pyarrow decord
```

## 4. 路径和运行时变量

下面的变量是示例名，不绑定任何机器目录。根据实际磁盘位置设置即可：

```bash
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export EXCEL_PATH="${EXCEL_PATH:-$PROJECT_ROOT/task.xlsx}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/classify_objects}"
export DATASET_ROOT="${DATASET_ROOT:-$DATA_ROOT/nero_classify_objects}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export FASTWAM_ROOT="${FASTWAM_ROOT:-$PROJECT_ROOT/vendor/FastWAM}"
export EFFICIENT_WAM_ROOT="${EFFICIENT_WAM_ROOT:-$PROJECT_ROOT/vendor/Efficient-WAM}"
export FASTWAM_PYTHON="${FASTWAM_PYTHON:-$FASTWAM_ROOT/.venv/bin/python}"
mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$RUN_ROOT"
```

### 4.1 安装两个隔离环境

Efficient-WAM 的环境文件同时固定了 Python 3.10、PyTorch 2.7.1 和
torchvision 0.22.1。首次安装：

```bash
conda env create -f vendor/Efficient-WAM/envs/train.yaml
conda run -n EfficientWAM python -m pip install -r adapters/efficient_wam/requirements.txt
conda run -n EfficientWAM python -c "import torch; print(torch.__version__)"
```

`flash-attn` 没有在仓库里钉死一个通用 wheel；环境创建完成后，必须在目标
节点按 `torch.version.cuda` 安装能匹配的版本，然后重新执行第 3 节的导入
检查。若环境名已存在，不要盲目覆盖正在使用的环境，可先 `conda env update`
或复制一份实验环境。

FastWAM 单独使用一个 venv。下面是本项目建议的 PyTorch 2.7.1/CUDA 12.8
（`cu128`）安装方式；若集群驱动不支持 CUDA 12.8，应改用管理员提供的
兼容 wheel，而不是强行安装：

```bash
bash scripts/clone_fastwam.sh
python3 -m venv "$FASTWAM_ROOT/.venv"
"$FASTWAM_PYTHON" -m pip install --upgrade pip
"$FASTWAM_PYTHON" -m pip install \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
"$FASTWAM_PYTHON" -m pip install -e "$FASTWAM_ROOT" \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

上游安装可能带有额外依赖；以 FastWAM checkout 自带的安装元数据为准，
不要把它们安装到 `EfficientWAM` 环境。固定审计 commit 的
`pyproject.toml` 也钉死了上述 Torch/CUDA wheel；若使用其他 upstream ref，
先检查其依赖再安装。最后确认两个解释器分别能导入各自的上游模块和
`torch`。系统还需要 `ffmpeg`：

```bash
command -v ffmpeg
"$FASTWAM_PYTHON" -c "import torch; print(torch.__version__, torch.version.cuda)"
conda run -n EfficientWAM python -c "import torch, pyarrow, deepspeed; print(torch.__version__)"
```

FastWAM 和 Efficient-WAM-RT 必须使用相互独立的 Python 环境。Efficient-WAM
的环境依赖以 `vendor/Efficient-WAM/envs/train.yaml` 为准，并额外检查
`pyarrow`、`decord`、`easydict`、`deepspeed`、`safetensors` 和 ffmpeg。
FastWAM 使用自己的环境，解释器通过 `FASTWAM_PYTHON` 指定：

```bash
export FASTWAM_PYTHON="$FASTWAM_ROOT/.venv/bin/python"
```

在已分配 GPU 的节点上分别执行导入检查。登录节点没有 GPU 时，检查只能说
明配置可导入，不能叫作 GPU smoke：

```bash
"$PYTHON_BIN" -c \
  "import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__); assert torch.cuda.is_available()"
"$FASTWAM_PYTHON" -c \
  "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
```

如果出现 `flash_attn_2_cuda...undefined symbol`，是 PyTorch、CUDA 和
flash-attn ABI 不匹配；不要把 FastWAM 环境的 `site-packages` 注入
Efficient-WAM 环境。

## 5. 获取代码、数据和 checkpoint

代码应从仓库的目标分支取得：

```bash
git clone --branch codex/classify-objects-finetuning --single-branch \
  https://github.com/Zhuang0v0/openloop-real-robot-reproduction.git
cd openloop-real-robot-reproduction
bash scripts/clone_fastwam.sh
# For the parameters documented here, pin the audited upstream revision.
git -C "$FASTWAM_ROOT" checkout --detach 7faa71108368fbb3b6885649f112af607427a2d4
git -C "$FASTWAM_ROOT" rev-parse HEAD
```

### 5.1 数据

`download_modelscope_dataset.py` 下载并校验 ModelScope 的
[`per_episode` 导出](https://modelscope.cn/datasets/XinyaoZhuang/nero_classify_objects/tree/master/per_episode)，
token 只从环境变量或交互式输入读取，不会写入仓库：

```bash
"$PYTHON_BIN" -m pip install modelscope
"$PYTHON_BIN" scripts/download_modelscope_dataset.py \
  --repo XinyaoZhuang/nero_classify_objects \
  --output-dir "$DATA_ROOT/modelscope"
```

脚本会生成一个带 SHA-256 记录的 tar 包。解压后，把 `DATASET_ROOT` 指向
**直接包含 `data/` 和 `videos/` 的目录**，例如：

```bash
tar -xf "$DATA_ROOT/modelscope/nero_classify_objects/nero_classify_objects_per_episode.tar" \
  -C "$DATA_ROOT"
find "$DATA_ROOT" -maxdepth 4 -type d \( -name data -o -name videos \)
```

解压后的 `DATASET_ROOT` 可以是 per-episode parquet，也可以是
`data/chunk-*/file-*.parquet` 的 LeRobot-v3 分片；当前审计器和
Efficient-WAM 转换器都会按 `episode_index` 合并分片，不依赖文件名顺序。
视频目录中的 chunk 编号也不要求固定为 `chunk-000`。确认根目录后执行：

```bash
test -d "$DATASET_ROOT/data" -a -d "$DATASET_ROOT/videos"
```

如果数据已经由管理员上传，跳过下载，只需设置 `DATASET_ROOT`，并保留数据
归档的校验记录。

`任务.xlsx` 是 Excel 映射输入，不由本脚本从 ModelScope 猜测或生成。若它不在
公开数据集归档中，需要由项目提供方单独交付；拿到后将 `EXCEL_PATH` 指向该文件，
并确认其中存在名称完全匹配的 `classify objects` 工作表。

### 5.2 模型

公开 RoboTwin checkpoint 可以用仓库脚本断点下载并校验。已有同一文件时，
不要重复下载：

```bash
"$PYTHON_BIN" scripts/download_fastwam_checkpoint.py \
  "$MODEL_ROOT/robotwin_uncond_3cam_384.pt"
"$PYTHON_BIN" scripts/download_efficientwam_checkpoint.py \
  "$MODEL_ROOT/Efficient-WAM-RT_stage3.pt" \
  --stats-output "$MODEL_ROOT/Efficient-WAM-RT_action_stats.json"
```

脚本内置的来源、文件和校验值如下；若改用 ModelScope 或用户已上传的副本，
仍应核对同一文件内容：

| 用途 | 文件 | 来源 | 期望大小 / SHA-256 |
|---|---|---|---|
| FastWAM RoboTwin 初始化 | `robotwin_uncond_3cam_384.pt` | [`yuanty/fastwam`](https://huggingface.co/yuanty/fastwam) | `12041813092` bytes / `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63` |
| Efficient-WAM-RT RoboTwin 初始化 | `Efficient-WAM-RT_stage3.pt` | [`jiajun0613/Efficient-WAM_RoboTwin/Efficient-WAM-RT`](https://huggingface.co/jiajun0613/Efficient-WAM_RoboTwin/tree/main/Efficient-WAM-RT) | `1976610407` bytes / `209ab4d6f897276633e4d0f36e6b0c573c4938cf0df99d974b05e754ab92340f` |
| Efficient-WAM-RT action 统计 | `Efficient-WAM-RT_action_stats.json` | 同一 Efficient-WAM-RT 发布目录 | `1772` bytes（下载脚本校验大小） |

默认下载走 `hf-mirror.com`，可通过 `FASTWAM_CHECKPOINT_URL`、
`EFFICIENT_CHECKPOINT_URL` 或对应命令行参数切换来源。下载完成后再执行：

```bash
sha256sum "$MODEL_ROOT/robotwin_uncond_3cam_384.pt" \
  "$MODEL_ROOT/Efficient-WAM-RT_stage3.pt"
```

ModelScope 数据脚本的 token 只应通过 `MODELSCOPE_TOKEN` 环境变量或隐藏式
提示输入：

```bash
export MODELSCOPE_TOKEN="${MODELSCOPE_TOKEN:?Inject MODELSCOPE_TOKEN through the platform secret store}"
"$PYTHON_BIN" scripts/download_modelscope_dataset.py \
  --repo XinyaoZhuang/nero_classify_objects --output-dir "$DATA_ROOT/modelscope"
unset MODELSCOPE_TOKEN
```

不要把 token 写进 shell history、`.env`、manifest、日志或 Git；如果平台已
通过凭据助手注入，就不要在文档命令中重复打印 token。

Efficient-WAM-RT 还需要与上游版本匹配的 WAN-2.2 TI2V-5B、VAE 和 UMT5
文件；`WanT5Embedder` 在当前代码中会查找以下相对路径：

```text
$WAN_MODEL_ROOT/Wan2.2_VAE.pth
$WAN_MODEL_ROOT/models_t5_umt5-xxl-enc-bf16.pth
$WAN_MODEL_ROOT/google/umt5-xxl/       # UMT5 tokenizer 文件
```

WAN 的主体配置/权重目录由上游 WAN-2.2 发布说明提供；本仓库不重新分发或
自动下载这些大文件。准备好后可在已分配 GPU 的 EfficientWAM 环境中检查
文本 embedding 入口（实际生成会在数据转换步骤执行）：

```bash
test -f "$WAN_MODEL_ROOT/Wan2.2_VAE.pth"
test -f "$WAN_MODEL_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
test -d "$WAN_MODEL_ROOT/google/umt5-xxl"
```

FastWAM 除初始化 checkpoint 外，还需要上游模型配置引用的 WAN/DiffSynth
公共模型文件。`prepare_fastwam.sh` 缺少
`ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` 时会调用上游
`$FASTWAM_ROOT/scripts/preprocess_action_dit_backbone.py` 生成它；通过
`FASTWAM_MODEL_BASE_PATH` 指向相应缓存，或用
`FASTWAM_ACTION_DIT_PRETRAINED_PATH` 指向已有兼容 payload。生成/下载的
payload 也必须放在 `MODEL_ROOT` 或数据盘，不要提交 Git。

Efficient 下载脚本附带的 `Efficient-WAM-RT_action_stats.json` 是发布的
RoboTwin 参考统计；Classify Objects 正式训练时，launcher 会从所选训练
episode 重新计算 16D `dataset_stats.json`，不要把旧的 14D 统计量当作真机
归一化统计。

## 6. Excel 映射和原始数据审计

第一步只建立“Excel 行 -> episode ID”的 manifest：

```bash
"$PYTHON_BIN" tools/match_classify_excel.py \
  --excel "$EXCEL_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$RUN_ROOT/mapping" \
  --sheet-name "classify objects" \
  --episodes 106
export CLASSIFY_MANIFEST="$RUN_ROOT/mapping/classify_episode_manifest.csv"
```

检查 `classify_unmatched_episodes.csv` 是否为空，以及生成的审计 Markdown。
然后检查 parquet、state/action、时间戳、FPS 和三路视频：

```bash
"$PYTHON_BIN" tools/audit_classify_dataset.py \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$CLASSIFY_MANIFEST" \
  --output-dir "$RUN_ROOT/audit" --probe-videos
```

该命令发现问题时返回非零；不要用 `|| true` 把错误吞掉。动作质量分析是
独立的诊断，不会覆盖原始轨迹：

```bash
"$PYTHON_BIN" tools/analyze_classify_actions.py \
  "$DATASET_ROOT" \
  --output "$RUN_ROOT/action_analysis/report.json" \
  --quality-csv "$RUN_ROOT/action_analysis/classify_action_quality.csv" \
  --anomaly-csv "$RUN_ROOT/action_analysis/classify_anomaly_segments.csv" \
  --report "$RUN_ROOT/action_analysis/classify_quality_report.md" \
  --plot-dir "$RUN_ROOT/action_analysis/plots"
```

## 7. 窗口、筛选和 split

代码支持四个策略：

| 策略 | 保留规则 |
|---|---|
| `baseline` | 结构有效的所有候选窗口 |
| `quality` | 通过 Excel 人工质量分阈值 |
| `action` | 通过速度、加速度、jerk、停顿和对齐检查 |
| `combined` | 同时通过质量和动作规则 |

`quality`/`combined` 默认在存在质量分时使用 `0.5` 阈值；需要比较论文/实验
常用的候选值时，可在两个 launcher 前设置
`CLASSIFY_QUALITY_THRESHOLD=0.8`（也可改为 `0.7` 或 `0.9`），并在报告中
记录该值。未知评分 `？` 不会被当成数值。

窗口在 episode 内产生，默认 launcher 参数为 `window-size=33`、`stride=16`；
FastWAM upstream 将 33 个 observation frames 解释为 32 个 action transitions，
而 16 是每个 action/state 的通道维度。筛选会丢弃整个窗口，不会删除中间
帧后把两段轨迹拼起来。`configs/classify_data_filter.yaml` 中的
`window.action_horizon: 16` 只是通用筛选/报告使用的 action chunk 参考值，
不是 FastWAM 的 `num_frames` 或 temporal horizon。

两个 launcher 都会自动执行下面两步。需要单独查看中间结果时，可显式运行：

```bash
"$PYTHON_BIN" tools/filter_classify_windows.py \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$CLASSIFY_MANIFEST" \
  --strategy baseline \
  --window-size 33 --stride 16 \
  --output "$RUN_ROOT/baseline_windows.json" \
  --quality-csv "$RUN_ROOT/classify_action_quality.csv" \
  --validation-fraction 0.20 --test-fraction 0.0 \
  --seed 20261111 \
  --report "$RUN_ROOT/baseline_quality_report.md"

"$PYTHON_BIN" tools/prepare_classify_data.py \
  --manifest "$CLASSIFY_MANIFEST" \
  --window-manifest "$RUN_ROOT/baseline_windows.json" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$RUN_ROOT/prepared" \
  --strategy baseline --episodes 106 \
  --validation-fraction 0.20 --split-mode random \
  --seed 20261111
```

`prepare_classify_data.py` 先按 episode ID 分配 split，再展开窗口，因此同一
episode 的重叠窗口不会同时出现在 train 和 validation。默认是 80/20；最终
想用 106 个 episode 全部训练时，在两个 launcher 前设置：

```bash
export CLASSIFY_VAL_FRACTION=0
```

这会产生空的 validation manifest，并关闭 FastWAM 的 upstream validation；
它不等于“有 validation”，训练 loss 不能写成 validation loss。

每次准备后至少保留以下文件，并把它们和实验名绑定，避免不同策略互相覆盖：

```text
$RUN_ROOT/mapping/classify_episode_manifest.csv
$RUN_ROOT/mapping/classify_unmatched_episodes.csv
$RUN_ROOT/{strategy}_windows.json
$RUN_ROOT/{strategy}_quality_report.md
$RUN_ROOT/prepared/train_manifest.csv
$RUN_ROOT/prepared/val_manifest.csv
$RUN_ROOT/prepared/prepare_metadata.json
```

Excel 中无法解析的质量值（例如 `？`）会保留为 `unknown`；它不能被默认为
0 或 1。`quality`/`combined` 策略的默认质量阈值来自
`filter_classify_windows.py`（有 manifest 时为 `0.5`）。如需论文式比较的
0.7、0.8 或 0.9 阈值，显式传参并把命令行记录下来：

```bash
"$PYTHON_BIN" tools/filter_classify_windows.py \
  --dataset-root "$DATASET_ROOT" --manifest "$CLASSIFY_MANIFEST" \
  --strategy quality --quality-threshold 0.8 \
  --window-size 33 --stride 16 --validation-fraction 0.20 \
  --test-fraction 0 --seed 20261111 \
  --output "$RUN_ROOT/quality_08_windows.json" \
  --quality-csv "$RUN_ROOT/quality_08_action_quality.csv" \
  --report "$RUN_ROOT/quality_08_report.md"
```

然后将 `CLASSIFY_WINDOW_MANIFEST`、`CLASSIFY_QUALITY_CSV`、
`CLASSIFY_QUALITY_THRESHOLD=0.8` 和 `CLASSIFY_FILTER_STRATEGY=quality` 指向
同一组文件再启动 launcher。若使用
launcher 的默认设置而未显式传阈值，实验记录必须写明实际阈值为 0.5。

四种策略使用同一个 launcher，只改变这些变量；每次实验都应使用独立的
`RUN_ROOT`，避免覆盖中间 manifest 或 checkpoint：

```bash
# 任选一行；quality/combined 可同时设置 CLASSIFY_QUALITY_THRESHOLD。
export CLASSIFY_FILTER_STRATEGY=baseline  # baseline
export CLASSIFY_FILTER_STRATEGY=quality   # quality
export CLASSIFY_FILTER_STRATEGY=action    # action
export CLASSIFY_FILTER_STRATEGY=combined  # combined
export RUN_ROOT="$PROJECT_ROOT/runs/classify_objects/${CLASSIFY_FILTER_STRATEGY}"
bash scripts/train_fastwam_classify.sh       # 或 train_efficientwam_classify.sh
```

上面的四行不是要连续执行；选择其中一行后再执行后两行。`quality` 和
`combined` 若不设置阈值会采用代码默认 `0.5`。

## 8. 14D checkpoint 适配为真机 16D

RoboTwin checkpoint 是 14D，而 Classify Objects 的 state/action 是 16D，
不能直接加载。适配脚本使用显式映射
`0,1,2,3,4,5,7,8,9,10,11,12,13,15`：保留已有关节/夹爪语义，把真机新增的
两个第七关节槽位置零，随后在微调中学习。执行：

```bash
MODEL_ROOT="$MODEL_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/adapt_fastwam_checkpoint.sh
MODEL_ROOT="$MODEL_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/adapt_efficient_checkpoint.sh
```

输出旁边会有 JSON 变换报告；检查报告中的输入/输出 shape 和 SHA-256。适配
脚本只做离线张量变换，不代表模型已经能前向。

## 9. FastWAM 代码实际做什么

### 9.1 上游版本和可训练模块证据

本仓库按 FastWAM upstream commit
`7faa71108368fbb3b6885649f112af607427a2d4` 审计。若使用其他
`FASTWAM_REF`，必须把实际 `git rev-parse HEAD` 写入运行记录，并重新检查
下面的语义。固定 commit 中的关键代码位置是：

* `src/fastwam/trainer.py:82-94` 先调用
  `_apply_dit_only_train_mode`，再把 `list(self.model.dit.parameters())`
  和可选 `proprio_encoder` 参数交给 AdamW。
* `src/fastwam/trainer.py:310-318` 先对整个模型执行
  `requires_grad_(False)`，再对 `model.dit` 和可选 proprio encoder 执行
  `train()`/`requires_grad_(True)`。
* `src/fastwam/models/wan22/fastwam.py:18-48` 将
  `video_expert`、`action_expert` 和 `mot` 注册到模型，并令
  `self.dit = self.mot`；`136-154` 用
  `MoT(mixtures={"video": video_expert, "action": action_expert})`
  构造两支 expert。

因此，当前 `dit_only` **不是“冻结 video、只训练 action”**：video DiT 和
action DiT 都会进入优化器，proprio encoder（若启用）也会训练；WAN 的 VAE、
文本编码器、tokenizer 等非 DiT 组件保持冻结。`freeze_video_backbone` 之类的
说明性字段不会覆盖 upstream trainer，也不能被当成实际冻结开关。

启动命令：

```bash
export FASTWAM_CHECKPOINT="$MODEL_ROOT/robotwin_uncond_3cam_384_16d.pt"
export CLASSIFY_FILTER_STRATEGY=baseline   # baseline|quality|action|combined
export CLASSIFY_QUALITY_THRESHOLD=         # quality/combined: empty => default 0.5
export CLASSIFY_VAL_FRACTION=0.20          # 0 表示 106 个 episode 全部训练
export FASTWAM_LEARNING_RATE=5e-5
bash scripts/train_fastwam_classify.sh
```

`train_fastwam_classify.sh` 的顺序是：

1. 生成窗口报告，并调用 `prepare_classify_data.py` 写出
   `runs/classify_objects/prepared/train_manifest.csv` 和 `val_manifest.csv`。
2. 只用训练 episode 计算 16D `dataset_stats.json`，避免把 validation 数值
   泄漏到归一化统计量。
3. `prepare_fastwam.sh` 从 FastWAM 模板生成 data/task/model YAML，替换数据
   根目录、三路相机 key、16D shape、指令、统计量和 checkpoint；缺少
   ActionDiT payload 时调用上游预处理脚本生成。
4. `adapters/fastwam/run_with_manifest.py` 安装 manifest loader，再进入上游
   Hydra `$FASTWAM_ROOT/scripts/train.py`。loader 同时过滤 episode ID 和显式窗口的局部
   frame starts，避免随机 episode split 和跨 split 泄漏。

默认训练值（可用环境变量覆盖）：

| 参数 | 默认值 |
|---|---:|
| batch size | 1 |
| gradient accumulation | 8 |
| learning rate | `5e-5` |
| max steps | 2000 |
| save/eval interval | 250 |
| mixed precision | bf16 |
| train mode | upstream `dit_only` |
| video DiT | 训练（与 action DiT 一起） |
| WAN/VAE 等非 DiT 组件 | 冻结 |
| observation frames (`num_frames`) | 33 |
| action temporal horizon | 32 transitions (`33 - 1`) |
| action/state channel dimension | 16 |

这里的“video backbone”不能简单写成冻结：当前 FastWAM upstream 的
`dit_only` 模式会冻结非 DiT 组件，但让 video DiT、action DiT 和 proprio
encoder 参与训练。`configs/fastwam_classify.yaml` 中的字段是实验合同记录，
真正生效的 train mode 由上游 `trainer.py` 实现。

正式运行前可用一个单 episode manifest 做 FastWAM 短测试。上游模型要求视频
时间维 `T > 1` 且 `T % 4 == 1`；本项目正式模板是 `T=33`，所以 smoke 也
保持窗口大小 33。`FASTWAM_MAX_STEPS=1` 才是“一步 optimizer 更新”的开关，
不是把视频窗口改成 1。下面的命令只验证路径、前向/反向和 checkpoint 保存，
不是微调结果：

```bash
export FASTWAM_SMOKE_ROOT="$RUN_ROOT/fastwam_smoke"
mkdir -p "$FASTWAM_SMOKE_ROOT"
"$PYTHON_BIN" - "$CLASSIFY_MANIFEST" "$FASTWAM_SMOKE_ROOT/manifest.csv" <<'PY'
import csv
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
with source.open(encoding="utf-8-sig", newline="") as handle:
    rows = sorted(csv.DictReader(handle), key=lambda row: int(float(row["episode_id"])))
if not rows:
    raise SystemExit("the matched manifest is empty")
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerow(rows[0])
PY
export CLASSIFY_MANIFEST="$FASTWAM_SMOKE_ROOT/manifest.csv"
export RUN_ROOT="$FASTWAM_SMOKE_ROOT/run"
export FASTWAM_CHECKPOINT="${FASTWAM_CHECKPOINT:-$MODEL_ROOT/robotwin_uncond_3cam_384_16d.pt}"
test -s "$FASTWAM_CHECKPOINT"
export EXPECTED_EPISODES=1
export CLASSIFY_VAL_FRACTION=0
export CLASSIFY_WINDOW_SIZE=33
export CLASSIFY_WINDOW_STRIDE=16
export FASTWAM_MAX_STEPS=1
export FASTWAM_SAVE_EVERY=1
export FASTWAM_EVAL_EVERY=0
bash scripts/train_fastwam_classify.sh

# Upstream trainer writes both a weights file and an Accelerate state directory.
SMOKE_CKPT="$RUN_ROOT/fastwam/train/checkpoints/weights/step_000001.pt"
SMOKE_STATE="$RUN_ROOT/fastwam/train/checkpoints/state/step_000001"
test -s "$SMOKE_CKPT"
test -f "$SMOKE_STATE/trainer_state.json"
sha256sum "$SMOKE_CKPT" | tee "$RUN_ROOT/fastwam/smoke_checkpoint.sha256"
```

成功时日志应包含 `[done] max_steps reached step=1`，并同时列出
`weights=.../fastwam/train/checkpoints/weights/step_000001.pt` 和
`state=.../fastwam/train/checkpoints/state/step_000001`。若只生成了
`config.yaml`、日志或 `dataset_stats.json`，而没有上述两个 checkpoint 产物，
smoke 必须记为失败。`checkpoints/weights/step_*.pt` 是模型权重；
`checkpoints/state/step_*` 才包含 optimizer/scheduler 等断点状态，不能互相
替代。

## 10. Efficient-WAM-RT 代码实际做什么

启动命令：

```bash
export EFFICIENT_WAM_ROOT="$PROJECT_ROOT/vendor/Efficient-WAM"
export WAN_MODEL_ROOT="$MODEL_ROOT/Wan2.2-TI2V-5B"
export CLASSIFY_FILTER_STRATEGY=baseline
export CLASSIFY_QUALITY_THRESHOLD=
export CLASSIFY_VAL_FRACTION=0.20
# The launcher consumes the two files produced by the default adapter names:
# Efficient-WAM-RT_stage3_16d.pt and Efficient-WAM-RT_compact_16d.pt.
test -f "$MODEL_ROOT/Efficient-WAM-RT_stage3_16d.pt"
test -f "$MODEL_ROOT/Efficient-WAM-RT_compact_16d.pt"
bash scripts/train_efficientwam_classify.sh
```

launcher 的顺序是：

1. 与 FastWAM 相同地生成候选窗口、质量报告和 episode-disjoint split。
2. `build_efficient_real_dataset.sh` 调用
   `adapters/efficient_wam/lerobot_to_robotwin_clean.py`：读取 16D
   `action` 和 `observation.state`，写成分开的 `qpos/`、`state/`；把三路
   30 FPS 视频拼成 320x384 mosaic；写入固定指令和 UMT5/WAN embedding。
3. `render_efficient_config.py` 把 YAML 中的占位符替换成当前运行目录。
4. 上游 `$EFFICIENT_WAM_ROOT/scripts/robotwin/build_train_dataset.py` 在 GPU 上生成 safetensors
   shards，随后 `verify_train_dataset.py` 校验条件 latent 和 shape。
5. 生成 Stage 2 配置并运行 `python -m train.stage2`。默认冻结 WAN/video
   expert，只更新 action expert；validation manifest 不会被打包进训练根，
   因为上游 video-action trainer 没有内置 validation loop。

**当前实现的重要边界：** Efficient converter 的
`--episode-ids-file` 只读取 manifest 中的 `episode_id`，不会读取窗口的
`start/end`。所以 Efficient-WAM 当前按筛选结果选出的 **episode** 转换整条
轨迹；FastWAM 才会进一步按窗口起点过滤。不要在实验记录中声称两个模型已
经做了完全相同的窗口级裁剪。如果需要严格一致，必须先扩展 Efficient 的
packer/dataset 接口，让它接收允许的窗口起点。

### 10.1 10 步训练与 RT 两步视频刷新不是一回事

本仓库的 Efficient 配置明确记录三件不同的事：

* 微调训练 scheduler 采样 **10 个 flow-matching bins**；
* RT 动作推理使用 **10 个 action inference steps**；
* RT 视频分支只在 step `[0, 1]` 刷新，也就是两次 video refresh。

“视频只去噪两步”不能改写成“训练只跑两步”。10-bin 是本项目训练配置，
需要在目标 GPU 上通过 smoke 验证，不能把它当作所有论文的统一结论。

默认 Efficient Stage 2 值：

| 参数 | 默认值 |
|---|---:|
| per-device batch | 1 |
| gradient accumulation | 8 |
| action learning rate | `5e-6` |
| video learning rate | 0（WAN frozen） |
| max steps | 1500 |
| checkpoint interval | 250 |
| training bins | 10 |

Stage 3 joint continuation 默认关闭。只有 Stage 2 能完成前向、反向、保存和
重新加载后，才应单独规划短 Stage 3，并记录其解冻范围和学习率。

## 11. Smoke 和评估

Efficient-WAM 的一集一更新 GPU smoke：

```bash
export CLASSIFY_MANIFEST="$RUN_ROOT/mapping/classify_episode_manifest.csv"
export CLASSIFY_SMOKE_ROOT="$RUN_ROOT/smoke"
bash scripts/smoke_efficientwam_classify.sh
```

该脚本把 manifest 缩成一个 episode，设置 `max_steps=1`，检查转换器、
packer、模型 loss/backward 和 checkpoint 保存/导出。没有 GPU、WAN 缓存或
匹配 flash-attn 时必须记为“未执行”，不能把 shell 启动成功当成 smoke 通过。

`evaluate_classify.sh` 目前是离线数据审计入口，不会自动执行机器人，也不会
把 prediction error 变成任务成功率：

```bash
export DATASET_ROOT="$DATASET_ROOT"
export CLASSIFY_MANIFEST="$RUN_ROOT/mapping/classify_episode_manifest.csv"
export CLASSIFY_EVAL_CHECKPOINT="${CLASSIFY_EVAL_CHECKPOINT:?Set CLASSIFY_EVAL_CHECKPOINT to an existing checkpoint}"
bash scripts/evaluate_classify.sh
```

模型特定的预测评估仍需调用相应上游 runtime。真机评测应使用未参与微调的
回放，并单独记录 C1--C4、碰撞、超时、完成率和动作空间/单位假设。

## 12. 训练输出、断点续训和复现

### 12.1 训练输出

FastWAM 的 launcher 会把生成的 YAML、`dataset_stats.json`、ActionDiT
payload 和训练结果放在 `$RUN_ROOT/fastwam/`。训练结果的精确布局是：

```text
$RUN_ROOT/fastwam/train/config.yaml
$RUN_ROOT/fastwam/train/checkpoints/weights/step_000250.pt
$RUN_ROOT/fastwam/train/checkpoints/state/step_000250/  # Accelerate 状态
```

其中 `000250` 会随 `save_every` 或最终 `max_steps` 改变；最终一步会额外
保存一次。上游 checkout 内还会生成对应的
`$FASTWAM_ROOT/configs/data`、`$FASTWAM_ROOT/configs/task` 和
`$FASTWAM_ROOT/configs/model` 副本。Efficient-WAM
会在 `$RUN_ROOT/efficient_source_train/` 写 clean source，在
`$RUN_ROOT/efficient_train_dataset/` 写 safetensors shards，在
`$RUN_ROOT/efficient_checkpoints/` 写 Accelerate resume state 和导出的模型。
这些目录都应位于数据盘，并在实验结束后按保留策略清理；不要提交 Git。

训练日志中的 `global_step` 是 optimizer step，不是 dataloader batch 数。
有效 batch 还要结合 `gradient_accumulation_steps` 解释。保存间隔和最终
checkpoint 的 SHA-256 应写入实验记录。

### 12.2 断点续训

先保留与首次运行相同的代码版本、manifest、split seed、数据统计量和模型
配置。不要把“初始化权重”和“训练恢复状态”混为一类：

* FastWAM 默认把 `FASTWAM_CHECKPOINT` 写入生成 task 配置的 `resume` 字段。
  如果该路径是 `.pt`，上游只加载权重，optimizer、scheduler 和 step 不会
  恢复。若要恢复完整状态，把 `FASTWAM_RESUME` 设置为上游生成的训练状态
  目录；launcher 会把它写入同一字段：

```bash
export FASTWAM_RESUME="${FASTWAM_RESUME:?Set this to a FastWAM full-state checkpoint directory}"
bash scripts/train_fastwam_classify.sh
```

  续训前检查生成的 `$FASTWAM_ROOT/configs/task/*.yaml` 中的 `resume`，并
  确认该 checkout 的状态目录格式。不同 FastWAM 版本的目录命名可能不同。
* Efficient-WAM 的 `model.efficient_wam_init_checkpoint` 是导出的 `.pt`
  模型初始化权重；真正的 optimizer/scheduler 恢复使用 `resume_from` 指向
  Accelerate 的 `step_*` 目录。当前 launcher 没有替这个参数提供环境变量，
  因此应在首次准备完成后直接调用上游 CLI：

```bash
export RESUME_DIR="${RESUME_DIR:?Set RESUME_DIR to an existing Efficient-WAM step directory}"
test -d "$RESUME_DIR"
cd "$EFFICIENT_WAM_ROOT"
PYTHONPATH="$EFFICIENT_WAM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m train.stage2 \
  --config "$RUN_ROOT/stage2_classify_16d.yaml" \
  --resume_from "$RESUME_DIR" \
  --device cuda --per_device_batch_size 1 \
  --num_workers 2 --gradient_accumulation_steps 8 \
  --max_steps 1500 --checkpoint_interval 250 --report_to tensorboard
```

不要用导出的 `.pt` 文件冒充 `resume_from` 目录；如果只想从一个导出的
模型重新开始，保持 `resume_from` 为空，并把该 `.pt` 写入模型配置对应的
初始化字段。

### 12.3 复现实验记录

建议在 `$RUN_ROOT` 保存一份不含 secret 的运行清单：

```bash
git rev-parse HEAD > "$RUN_ROOT/code_revision.txt"
"$PYTHON_BIN" -m pip freeze > "$RUN_ROOT/efficient_python_packages.txt"
"$FASTWAM_PYTHON" -m pip freeze > "$RUN_ROOT/fastwam_python_packages.txt"
sha256sum "$CLASSIFY_MANIFEST" \
  "$RUN_ROOT/prepared/train_manifest.csv" \
  "$RUN_ROOT/prepared/val_manifest.csv" \
  > "$RUN_ROOT/manifest_sha256.txt"
```

同时记录过滤策略、seed、训练/验证 episode ID、有效窗口数、GPU 型号、
CUDA/PyTorch/flash-attn 版本、所有非默认环境变量、初始和最终 checkpoint
SHA-256。不要保存 `MODELSCOPE_TOKEN`、密码、TOTP 或连接配置。

### 12.4 运行目录

建议保留以下可复现实验记录：

```text
dataset archive SHA-256
matched manifest SHA-256
selected strategy and seed
train/validation episode IDs and window counts
checkpoint SHA-256 and 14D->16D adapter report
GPU 型号、显存、CUDA/PyTorch/flash-attn 版本
effective batch size、学习率、累积步数和 max steps
best/final checkpoint、loss 曲线和任何未解决告警
```

典型运行目录（不提交 Git）：

```text
data/                         原始导出和归档
models/                       checkpoint、WAN/T5/VAE 和缓存
runs/classify_objects/
  mapping/                    Excel 映射结果
  audit/                      结构与视频审计
  action_analysis/            动作质量报告和图
  prepared/                   train/val manifest
  fastwam/                    FastWAM 统计、配置、ActionDiT 和 checkpoint
  efficient_source_train/    Efficient clean source
  efficient_train_dataset/   safetensors shards
  efficient_checkpoints/     Efficient-WAM checkpoint
```

不要把这些目录、模型权重、optimizer state、视频、parquet、token、密码或
个人路径提交到 GitHub。

## 13. 仓库结构

```text
README.md                         仓库入口和任务摘要
README_MODEL_TRAINING.md          完整训练流程概览
docs/TRAINING_CLASSIFY_OBJECTS.md 本文：代码数据流和运行手册
configs/                          任务合同、筛选和模型默认值
tools/                            Excel 映射、审计、动作分析、split
scripts/                          下载、适配、准备、训练和评估入口
adapters/fastwam/                 FastWAM manifest loader
adapters/efficient_wam/           Efficient 转换器、16D 适配和配置模板
vendor/Efficient-WAM/              训练所需 Efficient-WAM 源码副本
vendor/FastWAM/                   clone_fastwam.sh 生成的本地上游 checkout（被忽略）
reports/                           调研和质量报告模板
tests/                             不依赖大权重的单元测试
```

`data/`、`models/`、`runs/`、`vendor/FastWAM/` 和所有大文件由 `.gitignore`
排除；它们应由运行环境单独管理。

## 14. 故障排查

### `classify_episode_manifest.csv` 有 unmatched 或 episode 数量不对

确认 Excel 工作表名严格为 `classify objects`，`EXCEL_PATH` 指向正确工作簿，
并让数据根目录直接包含 `data/`、`videos/`。不要按视频文件名重命名 episode；
修正映射后从第 6 节重新生成 manifest。

### state/action 不是 16D，或行数、FPS 不一致

停止训练，查看 `$RUN_ROOT/audit/classify_dataset_audit.md` 和 inspection JSON。
确认使用的是 Classify Objects 真机导出而不是 RoboTwin 原始 14D 数据；不要用
裁剪或零填充掩盖 schema 错误。

### `flash_attn` undefined symbol / no kernel image

在同一个解释器中打印 `torch.__version__`、`torch.version.cuda` 和
`flash_attn.__version__`，重新安装匹配 wheel。不要混用 FastWAM 和
Efficient-WAM 的 `site-packages`；登录节点没有 GPU 时不要运行模型 smoke。

### 找不到 `pyarrow`、`decord`、`easydict` 或 `deepspeed`

确认当前 shell 激活的是 Efficient-WAM 环境，并执行：

```bash
conda run -n EfficientWAM python -m pip install -r adapters/efficient_wam/requirements.txt
conda run -n EfficientWAM python -c "import pyarrow, decord, easydict, deepspeed"
```

### Efficient 找不到 WAN/T5/VAE 或 UMT5 shape 错误

确认 `WAN_MODEL_ROOT` 指向完整的 WAN-2.2 TI2V-5B 目录，并检查
`Wan2.2_VAE.pth`、文本编码器文件和 `umt5_wan/*.pt`。重新运行
`build_efficient_real_dataset.sh` 前先删除对应 run 的半成品 clean/latent
目录，避免把不同 WAN 版本的缓存混在一起。

### 显存不足或作业被杀

先保持 batch size 1、bf16、gradient accumulation 8；减少 `num_workers`、
关闭不必要的并行作业，并用一集 smoke 确认峰值显存。不要未经记录地改变
视频尺寸、action horizon 或训练 bins；这些改变会破坏可比性。

### 训练脚本提示 checkpoint 不存在或 shape mismatch

确认原始 14D 文件先经过对应的 `adapt_*_checkpoint.sh`，并检查默认输出名
`robotwin_uncond_3cam_384_16d.pt`、`Efficient-WAM-RT_stage3_16d.pt` 和
`Efficient-WAM-RT_compact_16d.pt`。读取 adapter JSON 报告和 SHA-256，不要
直接把 14D 权重传给 16D 模型。

### 训练“成功结束”但没有可用 checkpoint

检查日志中的 loss/backward、导出路径和文件大小；Efficient smoke 还必须找到
`efficient_checkpoints/` 下的 `exports/*stage2*_step_*.pt`。只有文件可加载、
shape 正确并且 hash 已记录，才算 checkpoint 产出成功。

## 15. 当前验证边界

仓库提交前可以在无 GPU 环境执行 Python 单元测试、Python 编译检查和 Bash
语法检查；这些检查只覆盖数据合同、manifest、配置渲染和路径逻辑。以下项目
必须在真正分配到 A100（或兼容 GPU）的节点上补跑，当前文档不把它们标记为
已成功：

* 下载后的真实 `nero_classify_objects` 数据审计和视频 probe；
* FastWAM 一 episode/一步 smoke 以及正式微调；
* Efficient-WAM-RT 一 episode/一步 smoke、Stage 2/Stage 3 训练；
* 模型特定的离线预测评估和未参与微调的真机成功率评测。

如果调度器暂时没有可用 GPU，只记录“未执行”和原因，不要用登录节点的
`nvidia-smi` 缺失、CPU 导入成功或 shell 启动成功替代 GPU 结果。
