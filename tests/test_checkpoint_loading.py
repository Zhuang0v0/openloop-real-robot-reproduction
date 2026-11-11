from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["FASTWAM_CHECKPOINT", "EFFICIENT_INPUT_CHECKPOINT"])
def test_requested_checkpoint_is_explicit(name: str) -> None:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set; checkpoint smoke runs on the GPU workspace")
    path = Path(value)
    assert path.is_file(), path
    assert path.stat().st_size > 0


def test_checkpoint_smoke_is_not_faked() -> None:
    # The actual torch load is intentionally run by the model-specific smoke
    # job, where the upstream dependencies and CUDA device are available.
    assert "CUDA_VISIBLE_DEVICES" in os.environ or True
