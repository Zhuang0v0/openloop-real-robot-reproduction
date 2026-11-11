#!/usr/bin/env python3
"""Print Efficient-WAM checkpoint metadata and action-expert tensor shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _strip_wrappers(key: str) -> str:
    value = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "state_dict."):
            if value.startswith(prefix):
                value = value[len(prefix) :]
                changed = True
                break
    return value


def _is_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:256].decode("ascii", "replace")
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if _is_lfs_pointer(args.checkpoint):
        raise SystemExit(
            f"{args.checkpoint} is a Git-LFS pointer, not model bytes; download the 1.98GB RT object first"
        )
    payload = _load(args.checkpoint)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = payload["model"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise SystemExit("Unsupported checkpoint payload")
    config = payload.get("config") if isinstance(payload, dict) else None
    action_keys = {
        str(key): [int(value) for value in value.shape]
        for key, value in state.items()
        if _strip_wrappers(str(key)).startswith("action_expert.") and hasattr(value, "shape")
    }
    dimension_sensitive = {}
    suffixes = {
        "state_input_weight": "action_expert.input_encoder.state_encoder.0.weight",
        "action_input_weight": "action_expert.input_encoder.action_encoder.0.weight",
        "action_output_weight": "action_expert.decoder.action_head.0.weight",
        "action_output_bias": "action_expert.decoder.action_head.0.bias",
    }
    for key, value in state.items():
        bare = _strip_wrappers(str(key))
        for name, suffix in suffixes.items():
            if bare.endswith(suffix) and hasattr(value, "shape"):
                dimension_sensitive[name] = {
                    "key": str(key),
                    "shape": [int(item) for item in value.shape],
                }
                break
    report: Dict[str, Any] = {
        "path": str(args.checkpoint.resolve()),
        "bytes": int(args.checkpoint.stat().st_size),
        "sha256": _sha256_file(args.checkpoint),
        "payload_keys": sorted(str(key) for key in payload) if isinstance(payload, dict) else [],
        "tensor_count": len(state),
        "action_expert_tensors": action_keys,
        "dimension_sensitive_tensors": dimension_sensitive,
        "config": config,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
