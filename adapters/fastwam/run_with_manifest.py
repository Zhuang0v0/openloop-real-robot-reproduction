"""Run an upstream FastWAM training script with explicit episode filtering."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

try:
    from .manifest_loader import install_manifest_filter
except ImportError:  # direct execution by an absolute path
    from manifest_loader import install_manifest_filter


def main() -> None:
    install_manifest_filter()
    script = Path(os.environ.get("FASTWAM_TRAIN_SCRIPT", "scripts/train.py")).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"FastWAM training script does not exist: {script}")
    # Hydra uses argv[0] for diagnostics and leaves the remaining overrides
    # untouched.  The caller can therefore pass the same overrides as to the
    # upstream ``scripts/train.py`` entry point.
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
