# Efficient-WAM-RT 与 Fast-WAM 真机数据微调

本文档是本仓库的完整操作说明。目标是在任意可用的 CUDA GPU 机器上，使用 ModelScope 数据集 [`XinyaoZhuang/nero_building_block_tower`](https://modelscope.cn/datasets/XinyaoZhuang/nero_building_block_tower) 微调 RoboTwin 预训练的 **Efficient-WAM-RT** 和 **Fast-WAM**，再用独立采集的真机回放做评测。

本轮数据集有 **100 个 episode（编号 `0–99`）**，**全部用于训练**，不建立 held-out eval split。离线 smoke 只检查数据、模型和 launcher 是否能运行，不是评测集；模型选择使用训练完成后的真机成功率、碰撞和超时指标。

## 1. 代码、目录和可配置变量

代码来源：

- [Efficient-WAM](https://github.com/efficientwam/Efficient-WAM)，模型 [Efficient-WAM_RoboTwin/Efficient-WAM-RT](https://huggingface.co/jiajun0613/Efficient-WAM_RoboTwin/tree/main/Efficient-WAM-RT)
- [Fast-WAM](https://github.com/yuantianyuan01/FastWAM)，模型 [yuanty/fastwam](https://huggingface.co/yuanty/fastwam)

仓库中的 `vendor/Efficient-WAM` 是便于复现的源码副本；Fast-WAM 由 `scripts/clone_fastwam.sh` 在下载后放到工作目录。权重和数据不会放进 Git。

从仓库根目录设置变量。所有变量都可以改成另一块磁盘、另一台机器或集群的路径：

```bash
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export EFFICIENT_WAM_ROOT="${EFFICIENT_WAM_ROOT:-$PROJECT_ROOT/vendor/Efficient-WAM}"
export FASTWAM_ROOT="${FASTWAM_ROOT:-$PROJECT_ROOT/vendor/FastWAM}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-$MODEL_ROOT/Wan2.2-TI2V-5B}"
export FASTWAM_MODEL_BASE_PATH="${FASTWAM_MODEL_BASE_PATH:-$MODEL_ROOT/fastwam_models}"
export DATASET_ROOT="${DATASET_ROOT:-$DATA_ROOT/building_block_towers}"
export FASTWAM_PYTHON="${FASTWAM_PYTHON:-$FASTWAM_ROOT/.venv/bin/python}"
mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$RUN_ROOT"
```

约定的目录含义：

| 变量 | 内容 |
|---|---|
| `DATA_ROOT` | 原始 parquet、视频和数据集解压目录 |
| `MODEL_ROOT` | 两个 RoboTwin checkpoint 和其他模型文件 |
| `RUN_ROOT` | packed dataset、统计量、日志和训练 checkpoint |
| `EFFICIENT_WAM_ROOT` | Efficient-WAM 源码 |
| `FASTWAM_ROOT` | Fast-WAM 源码 |
| `WAN_MODEL_ROOT` | Efficient-WAM 直接读取的 WAN-2.2 TI2V-5B、VAE 和 UMT5 文件目录 |
| `FASTWAM_MODEL_BASE_PATH` | Fast-WAM/DiffSynth 模型缓存根目录；其下应有 `Wan-AI/Wan2.2-TI2V-5B` 等模型 ID 子目录 |

不要把真实密码、token、其他凭据或模型大文件写进这些目录的 Git 提交。

## 2. 数据协议

转换前的数据根目录必须能找到每个 episode 的 parquet 和三路同步视频。常见布局如下；如果导出工具使用了不同的 `chunk-*` 层级，只要保持字段和相机名一致即可：

```text
<DATASET_ROOT>/
├── data/**/episode_000000.parquet
├── videos/observation.images.camera_3/**/episode_000000.mp4
├── videos/observation.images.left_wrist/**/episode_000000.mp4
└── videos/observation.images.right_wrist/**/episode_000000.mp4
```

必需字段：

- `action`: `[T, 16]`
- `observation.state`: `[T, 16]`
- `observation.images.camera_3`
- `observation.images.left_wrist`
- `observation.images.right_wrist`
- instruction：`Stack the blocks into a tower.`

16D 顺序固定为：`[L1..L7, Lgrip, R1..R7, Rgrip]`。状态和动作都必须是 finite float32，且每个 episode 的状态/动作行数一致。

先执行独立检查：

```bash
"$PYTHON_BIN" scripts/validate_dataset.py "$DATASET_ROOT" --episodes 100
```

检查必须报告 `episodes: 100`、`held_out_evaluation_split: false` 和三路相机完整。发现缺失 episode、相机或维度错误时，先修复数据，不要让训练脚本静默跳过样本。

## 3. 下载代码、数据和基础模型

### 3.1 克隆源码和安装环境

```bash
git clone https://github.com/Zhuang0v0/openloop-real-robot-reproduction.git
cd openloop-real-robot-reproduction
```

Efficient-WAM 官方环境文件在 `vendor/Efficient-WAM/envs/train.yaml`。按机器的 CUDA/驱动选择 conda 或 venv，核心依赖至少包括 PyTorch、torchvision、transformers、diffusers、deepspeed、safetensors、opencv、ffmpeg/decord 和 `pyarrow`：

```bash
conda env create -f vendor/Efficient-WAM/envs/train.yaml
conda activate EfficientWAM
"$PYTHON_BIN" -m pip install -r adapters/efficient_wam/requirements.txt
```

Fast-WAM 源码和环境单独准备；不要假设虚拟环境一定叫 `.venv`：

```bash
bash scripts/clone_fastwam.sh
"$PYTHON_BIN" -m venv "$FASTWAM_ROOT/.venv"
"$FASTWAM_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$FASTWAM_ROOT/.venv/bin/python" -m pip install -e "$FASTWAM_ROOT"
export FASTWAM_PYTHON="$FASTWAM_ROOT/.venv/bin/python"
```

如果上游 Fast-WAM 的 `README` 或 `pyproject.toml` 对 CUDA/PyTorch 有更严格的版本要求，以源码版本说明为准；本仓库脚本只依赖 `FASTWAM_ROOT` 和 `FASTWAM_PYTHON`。

### 3.2 下载真机数据

如果当前环境还没有 ModelScope SDK，先安装它（只用于下载，不会把 token 写入仓库）：

```bash
"$PYTHON_BIN" -m pip install modelscope
```

ModelScope token 只从环境变量 `MODELSCOPE_TOKEN` 或交互式提示读取，不会写入文件。下载脚本会校验每个文件的大小和 `file_manifest.json` 中的 SHA-256，并生成 archive hash：

```bash
"$PYTHON_BIN" scripts/download_modelscope_dataset.py \
  --repo XinyaoZhuang/nero_building_block_tower \
  --output-dir "$DATA_ROOT/modelscope"
```

解压脚本输出的 `*_per_episode.tar` 到数据盘。将 `DATASET_ROOT` 设置为**包含 `data/` 和 `videos/` 的解压目录**；例如：

```bash
ARCHIVE="$(find "$DATA_ROOT/modelscope" -name '*_per_episode.tar' -print -quit)"
mkdir -p "$DATA_ROOT/raw"
tar -xf "$ARCHIVE" -C "$DATA_ROOT/raw"
# 下载脚本创建的顶层目录名是 <repo-name>_per_episode；确认其中直接有 data/ 和 videos/
DATASET_NAME="${MODELSCOPE_DATASET_REPO:-XinyaoZhuang/nero_building_block_tower}"
DATASET_NAME="${DATASET_NAME##*/}"
export DATASET_ROOT="$DATA_ROOT/raw/${DATASET_NAME}_per_episode"
test -d "$DATASET_ROOT/data" -a -d "$DATASET_ROOT/videos"
"$PYTHON_BIN" scripts/validate_dataset.py "$DATASET_ROOT" --episodes 100
```

如果使用 PowerShell，请用 `Get-ChildItem -Recurse -Filter '*_per_episode.tar'` 找到 archive 后再用支持 tar 的解压工具展开。不要把 archive、视频或 parquet 加入 Git。

### 3.3 下载模型文件

Efficient-WAM-RT 的 RoboTwin stage-3 checkpoint 可直接从公开 HF 仓库的镜像下载；脚本支持断点续传，并校验官方文件大小和 SHA-256。它还可以同时下载匹配的 RoboTwin action statistics：

```bash
"$PYTHON_BIN" scripts/download_efficientwam_checkpoint.py \
  "$MODEL_ROOT/Efficient-WAM-RT_stage3.pt" \
  --stats-output "$MODEL_ROOT/Efficient-WAM-RT_action_stats.json"
```

默认 URL 是 `hf-mirror.com/jiajun0613/Efficient-WAM_RoboTwin`。如果所在网络可直接访问 Hugging Face，可通过 `EFFICIENT_CHECKPOINT_URL` 和 `EFFICIENT_STATS_URL` 指向同名 `resolve/main/` 文件；无论下载来源，都要保留脚本打印的 SHA-256。

将 Fast-WAM 的 `robotwin_uncond_3cam_384.pt` 放到：

```bash
"$PYTHON_BIN" scripts/download_fastwam_checkpoint.py \
  "$MODEL_ROOT/robotwin_uncond_3cam_384.pt"
```

脚本默认使用 hf-mirror、支持断点续传，并校验已知大小和 SHA-256。上游文件更新时，用 `FASTWAM_CHECKPOINT_URL`、`FASTWAM_EXPECTED_BYTES` 和 `FASTWAM_CHECKPOINT_SHA256` 覆盖默认值。

Fast-WAM 的官方 RoboTwin `dataset_stats.json` 也要保存到 `$MODEL_ROOT`，用于真机评测的反归一化；微调统计量会在第 7 节由 100 个真机 episode 重新生成，不要把两者混用：

```bash
curl -L --fail --retry 3 \
  "https://hf-mirror.com/yuanty/fastwam/resolve/main/robotwin_uncond_3cam_384_dataset_stats.json" \
  -o "$MODEL_ROOT/robotwin_uncond_3cam_384_dataset_stats.json"
```

如果镜像不可达，也可以把同名文件从官方 ModelScope/Hugging Face 镜像复制到 `$MODEL_ROOT`，再记录 SHA-256；不要把模型文件提交到 Git。

此外，按 Efficient-WAM 模型卡下载 WAN-2.2 TI2V-5B、`Wan2.2_VAE.pth` 和 UMT5-XXL 所需文件到 `$WAN_MODEL_ROOT`。Fast-WAM 使用 DiffSynth 的模型 ID 目录布局，建议让它把同一组权重缓存到 `$FASTWAM_MODEL_BASE_PATH`；如果使用自动下载，设置 `DIFFSYNTH_DOWNLOAD_SOURCE=modelscope`（或 `huggingface`）即可。两个变量不要混为同一个目录，除非你已经按两种布局建立了软链接。这些大文件只作为运行时依赖，不提交到本仓库。

## 4. checkpoint 准备

### Efficient-WAM-RT

真机数据是 16D，而 RoboTwin checkpoint 是 14D。适配脚本把 RoboTwin 通道映射到真机通道：

```text
[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]
```

新增的第 7、15 位置（0-based）零初始化，随后由真机数据微调：

```bash
MODEL_ROOT="$MODEL_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/adapt_efficient_checkpoint.sh
```

脚本会生成：

```text
$MODEL_ROOT/Efficient-WAM-RT_stage3_16d.pt
$MODEL_ROOT/Efficient-WAM-RT_compact_16d.pt
$MODEL_ROOT/Efficient-WAM-RT_stage3_16d.json
```

适配后的 16D checkpoint 才能用于后续训练，不能把 14D 权重直接塞进 16D 配置。适配报告中的 SHA-256 应写入实验记录。

### Fast-WAM

Fast-WAM 同样先从 14D 扩展到 16D：

```bash
MODEL_ROOT="$MODEL_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/adapt_fastwam_checkpoint.sh
```

输出为 `$MODEL_ROOT/robotwin_uncond_3cam_384_16d.pt` 和对应 JSON 报告。原始 checkpoint 在适配输出通过大小和 SHA-256 校验前不要删除。

## 5. Efficient-WAM-RT 数据预处理

Efficient-WAM packer 需要每个 episode 的 state/action、视频和 UMT5/WAN 文本 embedding。100 个 episode 全部转换：

```bash
PROJECT_ROOT="$PROJECT_ROOT" \
DATA_ROOT="$DATA_ROOT" \
DATASET_ROOT="$DATASET_ROOT" \
RUN_ROOT="$RUN_ROOT" \
WAN_MODEL_ROOT="$WAN_MODEL_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/build_efficient_real_dataset.sh
```

脚本默认使用 `--start-episode 0 --end-episode 99`，并检查 qpos、state、视频和文本 embedding 各有 100 份。中间结果默认写到 `$RUN_ROOT/efficient_wam_source_16d`，不会覆盖源码目录。

渲染 packer 配置并构建 packed dataset：

```bash
"$PYTHON_BIN" scripts/render_efficient_config.py \
  adapters/efficient_wam/train_dataset_real_16d.yaml \
  "$RUN_ROOT/train_dataset_real_16d.yaml"

cd "$EFFICIENT_WAM_ROOT"
PYTHONPATH="$EFFICIENT_WAM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" scripts/robotwin/build_train_dataset.py \
  --config "$RUN_ROOT/train_dataset_real_16d.yaml" \
  --output-root "$RUN_ROOT/efficient_wam_train_dataset_16d" \
  --device cuda:0 --batch-size 1 --num-workers 2 \
  --shard-size 4096 --overwrite --verify-condition-latent

PYTHONPATH="$EFFICIENT_WAM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" scripts/robotwin/verify_train_dataset.py \
  "$RUN_ROOT/efficient_wam_train_dataset_16d"
cd "$PROJECT_ROOT"
```

`action_normalization` 的统计量来自全部 100 个 episode；没有 validation shard，也不能从 packed dataset 选择离线最佳模型。

## 6. Efficient-WAM-RT 微调

先做一小段 GPU smoke。smoke 只验证 launcher、shape、latent 和 checkpoint 保存：

```bash
PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" DATASET_ROOT="$DATASET_ROOT" \
RUN_ROOT="$RUN_ROOT" WAN_MODEL_ROOT="$WAN_MODEL_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/smoke_efficient_real_dataset.sh
```

正式训练从 16D RT checkpoint 开始，先做 frozen-WAN 的 Stage 2：

| 参数 | Stage 2 |
|---|---:|
| GPU | 1 × A100（或显存等价 GPU） |
| steps | 2000 |
| per-device batch | 1 |
| gradient accumulation | 8 |
| action learning rate | `1e-5` |
| WAN | frozen |
| action/video loss | `1.0 / 0.01` |

渲染配置并运行：

```bash
"$PYTHON_BIN" scripts/render_efficient_config.py \
  adapters/efficient_wam/stage2_action_16d.yaml \
  "$RUN_ROOT/stage2_action_16d.yaml"

cd "$EFFICIENT_WAM_ROOT"
PYTHONPATH="$EFFICIENT_WAM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m train.stage2 \
  --config "$RUN_ROOT/stage2_action_16d.yaml" \
  --device cuda --per_device_batch_size 1 --num_workers 2 \
  --gradient_accumulation_steps 8 --report_to tensorboard
cd "$PROJECT_ROOT"
```

Stage 2 通过后可选做 joint Stage 3。将 `STAGE2_CHECKPOINT` 指向 Stage 2 最后或经真机预检的 checkpoint，再渲染 Stage 3 配置：

| 参数 | Stage 3 |
|---|---:|
| steps | 1000 |
| action learning rate | `5e-6` |
| video learning rate | `1e-6` |
| WAN | unfrozen |

```bash
export STAGE2_CHECKPOINT="$RUN_ROOT/efficient_wam_checkpoints/stage2_step_2000.pt"
"$PYTHON_BIN" scripts/render_efficient_config.py \
  adapters/efficient_wam/stage3_joint_16d.yaml \
  "$RUN_ROOT/stage3_joint_16d.yaml"

cd "$EFFICIENT_WAM_ROOT"
PYTHONPATH="$EFFICIENT_WAM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m train.stage3 \
  --config "$RUN_ROOT/stage3_joint_16d.yaml" \
  --device cuda --per_device_batch_size 1 --num_workers 2 \
  --gradient_accumulation_steps 8 --report_to tensorboard
cd "$PROJECT_ROOT"
```

**去噪设置必须区分训练和推理：** Efficient-WAM-RT 推理使用 `num_inference_steps: 10`，并使用 `video_refresh_steps: [0, 1]`，即 10 个总推理步中只有两次视频刷新/去噪。训练代码使用 flow-matching 训练时间表（当前实现内部为 1000 个训练 timesteps），不能因为 RT 推理只刷新两步就把训练 scheduler 改成 2 步。

## 7. Fast-WAM 数据预处理与微调

先统计全部 100 个 episode 的 16D state/action，统计范围不能混入验证集：

```bash
DATA_ROOT="$DATA_ROOT" DATASET_ROOT="$DATASET_ROOT" RUN_ROOT="$RUN_ROOT" \
PYTHON_BIN="$PYTHON_BIN" EXPECTED_EPISODES=100 \
  bash scripts/generate_fastwam_stats.sh
```

再生成 Fast-WAM 的 data/model/task 配置。脚本会替换三路相机名、16D state/action、instruction、ActionDiT 路径，并关闭 held-out validation：

```bash
PROJECT_ROOT="$PROJECT_ROOT" FASTWAM_ROOT="$FASTWAM_ROOT" \
MODEL_ROOT="$MODEL_ROOT" RUN_ROOT="$RUN_ROOT" DATASET_ROOT="$DATASET_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/prepare_fastwam.sh
```

在 A100 或等价 GPU 上生成 16D ActionDiT backbone 和文本 embedding。具体脚本名以 Fast-WAM checkout 的版本为准；典型命令如下：

```bash
cd "$FASTWAM_ROOT"
DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_MODEL_BASE_PATH" \
  "$FASTWAM_PYTHON" scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_real.yaml \
  --output "$RUN_ROOT/fastwam_real/ActionDiT_linear_interp_Wan22_16d.pt" \
  --device cuda --dtype bfloat16

DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_MODEL_BASE_PATH" \
  "$FASTWAM_PYTHON" scripts/precompute_text_embeds.py \
  task=building_block_towers_3cam_384_1e-4 \
  model=fastwam_real data=building_block_towers \
  'override_instruction=Stack the blocks into a tower.'
cd "$PROJECT_ROOT"
```

先用上游训练入口的 50-step smoke，再做正式微调。由于本实验没有离线 eval split，生成配置把 `data.val` 设为 `null`，并把 `eval_every` 设为 `0`；所有 episode 只进入训练 loader。推荐设置：

| 参数 | 推荐值 |
|---|---:|
| GPU | 1 × A100（或显存等价 GPU） |
| batch size | 1 |
| gradient accumulation | 8 |
| learning rate | `1e-4`（cosine） |
| weight decay | `1e-2` |
| mixed precision | `bf16` |
| epochs | 5，或用 `max_steps` 固定步数 |
| validation split | 0；100 个 episode 全部训练 |

典型正式命令：

```bash
cd "$FASTWAM_ROOT"
DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_MODEL_BASE_PATH" \
  "$FASTWAM_PYTHON" scripts/train.py \
  task=building_block_towers_3cam_384_1e-4 \
  model=fastwam_real data=building_block_towers \
  batch_size=1 num_workers=2 gradient_accumulation_steps=8 \
  max_steps=50 save_every=50 eval_every=0 wandb.enabled=false
```

确认 smoke 能保存权重后，将 `max_steps` 改为正式步数（示例为 2000）：

```bash
DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_MODEL_BASE_PATH" \
  "$FASTWAM_PYTHON" scripts/train.py \
  task=building_block_towers_3cam_384_1e-4 \
  model=fastwam_real data=building_block_towers \
  batch_size=1 num_workers=2 gradient_accumulation_steps=8 \
  max_steps=2000 save_every=250 eval_every=0 wandb.enabled=false
cd "$PROJECT_ROOT"
```

`eval_every=0` 明确关闭离线评估；正式结果不从离线 eval loss 得出。训练 checkpoint 应保存在 `RUN_ROOT` 下，并记录实际 Fast-WAM commit、配置和 hash。

## 8. 参数选择依据

参数以两篇方法的公开训练入口和小样本真机微调的保守设置为基线，而不是把 RoboTwin 的大规模训练 batch 直接照搬到一张 A100：

- [Efficient-WAM, arXiv:2606.10040](https://arxiv.org/abs/2606.10040) 的 Stage 2 先冻结视频专家训练 action expert，再用较小学习率做 joint Stage 3；因此这里采用 `1e-5` 的 Stage 2 action LR、`5e-6/1e-6` 的 Stage 3 action/video LR。
- [Fast-WAM, arXiv:2603.16666](https://arxiv.org/abs/2603.16666) 的 RoboTwin task config 使用 cosine、`1e-4`、bf16；真机数据量较小时固定 batch 1、gradient accumulation 8，保持有效 batch 8 并降低显存峰值。
- 100 个 episode 全量训练意味着不能用离线 validation loss 选择“最佳” checkpoint；应保存定期 checkpoint，最后用真机成功率、碰撞和超时指标选择部署版本。
- 预训练模型的 action/state 通道首先扩展到 16D，再微调新增通道；不要在训练过程中临时改变相机数量、action 顺序或归一化统计范围。

## 9. 真机评测

训练完成后，分别选择 Efficient-WAM Stage 2/3 和 Fast-WAM 的 checkpoint，在目标机器的真机 runner 中加载对应的 16D preprocessing/postprocessing 配置。评测时：

1. 保持三路相机命名与训练一致：`camera_3`、`left_wrist`、`right_wrist`。
2. 保持 state/action 顺序为 `[L1..L7, Lgrip, R1..R7, Rgrip]`。
3. 使用同一条 instruction：`Stack the blocks into a tower.`
4. 先 dry-run 读取 observation，不执行 actuation；确认 action shape `[*,16]`、数值 finite、gripper 范围合理后再接机器人。
5. 用独立采集的真机任务统计成功率、堆叠完成率、碰撞和超时；不要把训练集 loss 当作真机成功率。

Efficient-WAM-RT 推理固定 10 total steps 和 `video_refresh_steps=[0,1]`。Fast-WAM 的推理步数从其 runtime config 读取，评测记录实际 `num_inference_steps`、源码 commit 和 checkpoint SHA-256。

## 10. 可复现记录

每个正式 run 至少保存：

- Git commit / source tree hash；
- 原始 checkpoint 与适配后 checkpoint 的 SHA-256；
- 数据集 archive、`file_manifest.json` 和 `dataset_stats.json` 的 SHA-256；
- 渲染后的训练 config、GPU 数量、step、batch、gradient accumulation、学习率；
- Efficient-WAM/Fast-WAM 源码 commit；
- 最终 checkpoint 路径和真机评测结果。

如果使用 Slurm 或其他调度器，把 job ID 和日志目录写进实验记录即可；不要把调度器的用户目录写死进仓库脚本。

## 11. 清理规则

训练和真机评测完成、并且最终 checkpoint 已通过 hash 校验后，才清理中间产物：

```bash
rm -rf -- "$RUN_ROOT/efficient_wam_source_16d"
rm -rf -- "$RUN_ROOT/efficient_wam_train_dataset_smoke_16d"
rm -rf -- "$RUN_ROOT/fastwam_real/text_embeds"
rm -f  -- "$RUN_ROOT/fastwam_real/ActionDiT_linear_interp_Wan22_16d.pt"
```

保留最终 checkpoint、最终 config、hash/manifest、训练日志和真机评测报告。不要在唯一原始 checkpoint 尚未完成组装和校验前删除它。大模型、视频和 packed shards 不应留在本地 Git 工作树。

## 12. 常见问题

- **`decord` 缺失**：转换器可使用 OpenCV/ffmpeg fallback；仍失败时检查 ffmpeg、OpenCV 和视频 codec。
- **`ftfy` 或 `wcwidth` 缺失**：按上游环境文件安装依赖，不要修改 checkpoint；安装后重新跑文本 embedding smoke。
- **维度 mismatch**：确认两个模型都加载了 16D adapted checkpoint，且配置中的 state/action/proprio/output dim 全为 16。
- **出现 `validation` episode**：检查使用的是本仓库最新转换器；本实验要求 0–99 全部 `split: train`，不建立 eval 集。
- **YAML 里仍有 `__MODEL_ROOT__` 等字符串**：先用 `scripts/render_efficient_config.py` 渲染模板，再把渲染后的文件传给训练入口；不要依赖 YAML 自动展开环境变量。
- **训练无 GPU**：检查调度器状态和 `CUDA_VISIBLE_DEVICES`，确认作业真正进入运行态；不要在登录节点运行正式训练。
- **OOM**：保持 batch size 1，增加 gradient accumulation；Efficient-WAM 优先减少 `num_workers`，不要降低 16D 或删除相机输入。
