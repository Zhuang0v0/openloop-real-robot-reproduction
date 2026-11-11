"""Render an Efficient-WAM YAML template with local experiment paths.

The checked-in YAML files intentionally contain ``__NAME__`` placeholders so
they are portable.  This small renderer keeps path substitution explicit and
fails if a placeholder was forgotten.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


PLACEHOLDER = re.compile(r"__[A-Z0-9_]+__")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    project_root = Path(
        os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])
    ).expanduser().resolve()
    data_root = Path(os.environ.get("DATA_ROOT", project_root / "data")).expanduser().resolve()
    model_root = Path(os.environ.get("MODEL_ROOT", project_root / "models")).expanduser().resolve()
    run_root = Path(os.environ.get("RUN_ROOT", project_root / "runs")).expanduser().resolve()
    wan_root = Path(
        os.environ.get("WAN_MODEL_ROOT", model_root / "Wan2.2-TI2V-5B")
    ).expanduser().resolve()
    efficient_source_root = Path(
        os.environ.get("EFFICIENT_SOURCE_ROOT", run_root / "efficient_source")
    ).expanduser().resolve()
    stage2 = Path(
        os.environ.get(
            "STAGE2_CHECKPOINT",
            run_root / "efficient_wam_checkpoints" / "stage2_step_2000.pt",
        )
    ).expanduser().resolve()

    replacements = {
        "__PROJECT_ROOT__": str(project_root),
        "__DATA_ROOT__": str(data_root),
        "__MODEL_ROOT__": str(model_root),
        "__RUN_ROOT__": str(run_root),
        "__EFFICIENT_SOURCE_ROOT__": str(efficient_source_root),
        "__WAN_MODEL_ROOT__": str(wan_root),
        "__STAGE2_CHECKPOINT__": str(stage2),
    }
    text = args.template.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    unresolved = sorted(set(PLACEHOLDER.findall(text)))
    if unresolved:
        raise ValueError(f"Unresolved path placeholders: {', '.join(unresolved)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"rendered={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
