# Classify Objects 真机微调

本仓库只包含使用 106 个 Classify Objects 真机 episode 微调
**FastWAM** 和 **Efficient-WAM-RT** 的代码。固定 task prompt 为：

```text
Put all fruits into the large basket first, then put all blocks into the small basket.
```

数据来源是 [ModelScope `XinyaoZhuang/nero_classify_objects`](https://modelscope.cn/datasets/XinyaoZhuang/nero_classify_objects)。
Excel 仅读取名称完全匹配的 `classify objects` 工作表；C1--C4 是场景
metadata，不是不同 prompt。数据、权重和运行产物均由环境变量指向仓库外
目录，不会提交到 Git。

从 [README_MODEL_TRAINING.md](README_MODEL_TRAINING.md) 开始，按顺序完成：

1. 安装上游 CUDA/PyTorch 环境并下载代码、数据和 checkpoint。
2. 用 Excel manifest 匹配 episode，审计 16D state/action、时间戳和三路视频。
3. 运行动作质量分析，比较 `baseline`、`quality`、`action`、`combined`。
4. 将两个 RoboTwin 14D checkpoint 显式适配为 16D，并生成训练统计量。
5. 使用同一份 episode-disjoint split 分别启动 FastWAM 和 Efficient-WAM-RT。
6. 先做前向/反向/checkpoint smoke，再进行正式微调和独立真机评测。

Efficient-WAM-RT 的训练配置使用 10 个 flow-matching bins；RT 推理使用
10 个 action steps，video branch 只在 `[0, 1]` refresh。这两个步数概念
不能混用。默认验证比例为 20%；最终全量训练可设置
`CLASSIFY_VAL_FRACTION=0`，但此时没有离线 validation。

仓库不包含任何连接教程、凭据、数据集、模型大文件或个人路径。详见主文档。
