from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="a POSIX bash environment is required for launcher integration test",
)
def test_prepare_fastwam_writes_configured_learning_rate(tmp_path: Path) -> None:
    source = tmp_path / "fastwam"
    (source / "configs" / "data").mkdir(parents=True)
    (source / "configs" / "task").mkdir(parents=True)
    (source / "configs" / "model").mkdir(parents=True)
    (source / "scripts").mkdir(parents=True)
    (source / "configs" / "data" / "robotwin.yaml").write_text(
        """train:\n  is_training_set: true\n  val_set_proportion: 0.1\n  pretrained_norm_stats: null\n  processor:\n    raw_shape: 14\n    shape: 14\n    action_output_dim: 14\n    proprio_output_dim: 14\n  text_embedding_cache_dir: ./data/text_embeds_cache/robotwin\nval:\n  is_training_set: false\n""",
        encoding="utf-8",
    )
    (source / "configs" / "task" / "robotwin_uncond_3cam_384_1e-4.yaml").write_text(
        """defaults:\n  - override /data: robotwin\n  - override /model: fastwam\nlearning_rate: 1e-4\nbatch_size: 16\nnum_workers: 8\ngradient_accumulation_steps: 1\nsave_every: 2500\neval_every: 500\nresume: null\n""",
        encoding="utf-8",
    )
    (source / "configs" / "model" / "fastwam.yaml").write_text(
        "action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt\n",
        encoding="utf-8",
    )
    (source / "scripts" / "preprocess_action_dit_backbone.py").write_text(
        "from pathlib import Path\nimport sys\nargs = sys.argv\nout = args[args.index('--output') + 1]\nPath(out).write_bytes(b'dummy')\n",
        encoding="utf-8",
    )

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    model_root = tmp_path / "models"
    model_root.mkdir()
    checkpoint = model_root / "input.pt"
    checkpoint.write_bytes(b"checkpoint")
    run_root = tmp_path / "runs"
    project_root = Path(__file__).parents[1].resolve()
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(project_root),
            "PYTHON_BIN": sys.executable,
            "FASTWAM_PYTHON": sys.executable,
            "FASTWAM_ROOT": str(source),
            "DATASET_ROOT": str(dataset),
            "MODEL_ROOT": str(model_root),
            "RUN_ROOT": str(run_root),
            "FASTWAM_RUN_ROOT": str(run_root / "fastwam"),
            "FASTWAM_CHECKPOINT": str(checkpoint),
            "FASTWAM_LEARNING_RATE": "5.0e-5",
            "FASTWAM_TASK_CONFIG_NAME": "classify_objects_3cam_384_5e-5",
            "FASTWAM_DATA_CONFIG_NAME": "classify_objects",
            "FASTWAM_MODEL_CONFIG_NAME": "fastwam_classify",
        }
    )
    subprocess.run(
        ["bash", str(project_root / "scripts" / "prepare_fastwam.sh")],
        cwd=project_root,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    task_path = source / "configs" / "task" / "classify_objects_3cam_384_5e-5.yaml"
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    assert task["learning_rate"] == pytest.approx(5.0e-5)
    assert task["batch_size"] == 1
    assert str(checkpoint) == task["resume"]
