#!/usr/bin/env python3
"""Expand an Efficient-WAM action expert checkpoint from 14D to 16D.

The released RT checkpoint is configured for RoboTwin's 14-dimensional robot
(six arm joints plus one gripper per arm).  The Classify Objects real-robot
recording has separate 16-dimensional state and action vectors (seven joints
plus one gripper per arm).  This tool copies the pretrained dimensions
according to an explicit old-index -> new-index map and initializes only the
two new joint dimensions in the state/action input MLPs and action output
head.  It never silently crops dimensions and it fails if an unexpected shape
mismatch is found.

This is an offline checkpoint transform; it does not instantiate WAN or run a
forward pass.  Run it in the Efficient-WAM environment where PyTorch is
installed, then validate the resulting checkpoint with a short model-load
smoke test before training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


# These are the only parameter tensors whose shape depends on the robot I/O
# dimensionality in the upstream ActionExpert implementation.  Keeping this
# list explicit prevents a partially adapted checkpoint from being accepted
# when an upstream export changes parameter names.
REQUIRED_LAYERS = {
    "state_input",
    "action_input",
    "action_output_weight",
    "action_output_bias",
}

# RoboTwin's released 14D policy uses
#   [left joint 1..6, left gripper, right joint 1..6, right gripper].
# The real-robot LeRobot recording is
#   [left joint 1..7, left gripper, right joint 1..7, right gripper].
# This map is old-index -> new-index.  The two new 7th-joint slots are zero
# initialized; all pretrained gripper/joint channels retain their semantics.
# RoboTwin's 14D channel order mapped to the 16D real-robot recording used by
# Classify Objects.  The name is task-neutral because the same physical layout
# is shared by both model adapters.
ROBOTWIN_14_TO_REAL_ROBOT_16 = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15)


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older torch has no weights_only argument.
        return torch.load(path, map_location="cpu")


def _torch_save(payload: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _state_dict(payload: Any) -> Tuple[Dict[str, Any], str]:
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        return payload["model"], "model"
    if isinstance(payload, dict) and all(isinstance(key, str) for key in payload):
        # A raw state dict is accepted for diagnostics and adaptation.
        return payload, "root"
    raise TypeError("Checkpoint must be a state dict or a mapping containing a 'model' state dict")


def _config_dims(payload: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(payload, dict):
        return None, None
    candidates = []
    config = payload.get("config")
    if isinstance(config, dict):
        candidates.append(config.get("model"))
    candidates.append(payload.get("model_config"))
    for model_cfg in candidates:
        if isinstance(model_cfg, dict):
            action_cfg = model_cfg.get("action_expert")
            if isinstance(action_cfg, dict):
                state_dim = action_cfg.get("state_dim")
                action_dim = action_cfg.get("action_dim")
                return (
                    int(state_dim) if state_dim is not None else None,
                    int(action_dim) if action_dim is not None else None,
                )
    return None, None


def _set_config_dims(payload: Any, state_dim: int, action_dim: int) -> List[str]:
    changed: List[str] = []
    if not isinstance(payload, dict):
        return changed
    config = payload.get("config")
    if isinstance(config, dict) and isinstance(config.get("model"), dict):
        action_cfg = config["model"].get("action_expert")
        if isinstance(action_cfg, dict):
            action_cfg["state_dim"] = int(state_dim)
            action_cfg["action_dim"] = int(action_dim)
            changed.extend(["config.model.action_expert.state_dim", "config.model.action_expert.action_dim"])
    model_cfg = payload.get("model_config")
    if isinstance(model_cfg, dict) and isinstance(model_cfg.get("action_expert"), dict):
        model_cfg["action_expert"]["state_dim"] = int(state_dim)
        model_cfg["action_expert"]["action_dim"] = int(action_dim)
        changed.extend(["model_config.action_expert.state_dim", "model_config.action_expert.action_dim"])
    return changed


def _compact_payload_from_full(payload: Any, state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract a Stage-1-compatible compact WAN payload from a full export."""

    compact_state: Dict[str, Any] = {}
    for key, value in state_dict.items():
        bare = _key_without_wrappers(str(key))
        if bare.startswith("compact_wan."):
            compact_state[bare[len("compact_wan.") :]] = value
        elif bare.startswith("student."):
            compact_state[bare[len("student.") :]] = value
    if not compact_state:
        raise ValueError("No compact_wan.* or student.* tensors found; cannot write compact checkpoint")

    compact_cfg: Dict[str, Any] = {}
    if isinstance(payload, dict):
        raw_cfg = payload.get("compact_wan_config")
        if isinstance(raw_cfg, dict):
            compact_cfg.update(raw_cfg)
        config = payload.get("config")
        if isinstance(config, dict) and isinstance(config.get("model"), dict):
            model_cfg = config["model"].get("compact_wan")
            if isinstance(model_cfg, dict):
                for key, value in model_cfg.items():
                    compact_cfg.setdefault(key, value)
    compact_cfg.setdefault("hidden_anchor_layers", [1, 3, 5, 7, 9, 12])
    compact_cfg.setdefault("motion_anchor_layers", [6, 8, 10, 12])
    compact_cfg.setdefault("teacher_layer_mapping", [1, 2, 4, 6, 8, 11, 14, 17, 20, 23, 26, 30])
    compact_payload: Dict[str, Any] = {
        "model": compact_state,
        "compact_wan_config": compact_cfg,
        "source": "efficient_wam_full_checkpoint_compact_extract",
    }
    if isinstance(payload, dict) and "global_step" in payload:
        compact_payload["global_step"] = payload["global_step"]
    return compact_payload


def _validate_index_map(index_map: Sequence[int], source_dim: int, target_dim: int) -> Tuple[int, ...]:
    values = tuple(int(item) for item in index_map)
    if len(values) != source_dim:
        raise ValueError(
            f"Index map has {len(values)} entries, but source dimension is {source_dim}"
        )
    if len(set(values)) != len(values):
        raise ValueError("Index map entries must be unique")
    if any(item < 0 or item >= target_dim for item in values):
        raise ValueError(f"Index map entries must be in [0, {target_dim})")
    return values


def _copy_expand_columns(value: Any, target_dim: int, index_map: Optional[Sequence[int]] = None) -> Any:
    import torch

    if value.ndim != 2 or target_dim < value.shape[1]:
        raise ValueError(f"Cannot expand input weight with shape {tuple(value.shape)} to dim {target_dim}")
    expanded = torch.zeros((value.shape[0], target_dim), dtype=value.dtype, device=value.device)
    if index_map is None:
        expanded[:, : value.shape[1]] = value
    else:
        mapping = _validate_index_map(index_map, int(value.shape[1]), target_dim)
        for old_index, new_index in enumerate(mapping):
            expanded[:, new_index] = value[:, old_index]
    return expanded


def _copy_expand_rows(value: Any, target_dim: int, index_map: Optional[Sequence[int]] = None) -> Any:
    import torch

    if value.ndim != 2 or target_dim < value.shape[0]:
        raise ValueError(f"Cannot expand output weight with shape {tuple(value.shape)} to dim {target_dim}")
    expanded = torch.zeros((target_dim, value.shape[1]), dtype=value.dtype, device=value.device)
    if index_map is None:
        expanded[: value.shape[0], :] = value
    else:
        mapping = _validate_index_map(index_map, int(value.shape[0]), target_dim)
        for old_index, new_index in enumerate(mapping):
            expanded[new_index, :] = value[old_index, :]
    return expanded


def _copy_expand_vector(value: Any, target_dim: int, index_map: Optional[Sequence[int]] = None) -> Any:
    import torch

    if value.ndim != 1 or target_dim < value.shape[0]:
        raise ValueError(f"Cannot expand output bias with shape {tuple(value.shape)} to dim {target_dim}")
    expanded = torch.zeros((target_dim,), dtype=value.dtype, device=value.device)
    if index_map is None:
        expanded[: value.shape[0]] = value
    else:
        mapping = _validate_index_map(index_map, int(value.shape[0]), target_dim)
        for old_index, new_index in enumerate(mapping):
            expanded[new_index] = value[old_index]
    return expanded


def _key_without_wrappers(key: str) -> str:
    value = key
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "state_dict."):
            if value.startswith(prefix):
                value = value[len(prefix) :]
                changed = True
                break
    return value


def adapt_state_dict(
    state_dict: Mapping[str, Any],
    *,
    state_dim: int,
    action_dim: int,
    old_state_dim: Optional[int] = None,
    old_action_dim: Optional[int] = None,
    index_map: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return an expanded state dict and a machine-readable transform report."""

    if state_dim <= 0 or action_dim <= 0:
        raise ValueError("Target state/action dimensions must be positive")
    if (
        index_map is None
        and state_dim == 16
        and action_dim == 16
        and old_state_dim in (None, 14)
        and old_action_dim in (None, 14)
    ):
        # Make the known RoboTwin -> real-robot layout the safe default
        # even for callers that omit checkpoint metadata.  A non-14D source
        # then fails map validation instead of silently using positional append.
        index_map = ROBOTWIN_14_TO_REAL_ROBOT_16
    output = dict(state_dict)
    transformed: List[Dict[str, Any]] = []
    recognized = set()
    already_target: List[Dict[str, Any]] = []
    for key, value in state_dict.items():
        if not hasattr(value, "shape"):
            continue
        bare = _key_without_wrappers(str(key))
        if bare.startswith(("input_encoder.", "decoder.")):
            # Action-only init checkpoints are sometimes saved with the
            # `action_expert.` prefix already stripped by the caller.
            bare = "action_expert." + bare
        shape = tuple(int(item) for item in value.shape)
        if bare.endswith("action_expert.input_encoder.state_encoder.0.weight"):
            recognized.add("state_input")
            if shape[1] != state_dim:
                if old_state_dim is not None and shape[1] != old_state_dim:
                    raise ValueError(f"Unexpected state input shape for {key}: {shape}; expected old dim {old_state_dim}")
                output[key] = _copy_expand_columns(value, state_dim, index_map)
                transformed.append({"key": str(key), "kind": "state_input_weight", "from": list(shape), "to": list(output[key].shape), "new_values": "zero"})
            else:
                already_target.append({"key": str(key), "kind": "state_input_weight", "shape": list(shape)})
        elif bare.endswith("action_expert.input_encoder.action_encoder.0.weight"):
            recognized.add("action_input")
            if shape[1] != action_dim:
                if old_action_dim is not None and shape[1] != old_action_dim:
                    raise ValueError(f"Unexpected action input shape for {key}: {shape}; expected old dim {old_action_dim}")
                output[key] = _copy_expand_columns(value, action_dim, index_map)
                transformed.append({"key": str(key), "kind": "action_input_weight", "from": list(shape), "to": list(output[key].shape), "new_values": "zero"})
            else:
                already_target.append({"key": str(key), "kind": "action_input_weight", "shape": list(shape)})
        elif bare.endswith("action_expert.decoder.action_head.0.weight"):
            recognized.add("action_output_weight")
            if shape[0] != action_dim:
                if old_action_dim is not None and shape[0] != old_action_dim:
                    raise ValueError(f"Unexpected action output shape for {key}: {shape}; expected old dim {old_action_dim}")
                output[key] = _copy_expand_rows(value, action_dim, index_map)
                transformed.append({"key": str(key), "kind": "action_output_weight", "from": list(shape), "to": list(output[key].shape), "new_values": "zero"})
            else:
                already_target.append({"key": str(key), "kind": "action_output_weight", "shape": list(shape)})
        elif bare.endswith("action_expert.decoder.action_head.0.bias"):
            recognized.add("action_output_bias")
            if shape[0] != action_dim:
                if old_action_dim is not None and shape[0] != old_action_dim:
                    raise ValueError(f"Unexpected action output bias shape for {key}: {shape}; expected old dim {old_action_dim}")
                output[key] = _copy_expand_vector(value, action_dim, index_map)
                transformed.append({"key": str(key), "kind": "action_output_bias", "from": list(shape), "to": list(output[key].shape), "new_values": "zero"})
            else:
                already_target.append({"key": str(key), "kind": "action_output_bias", "shape": list(shape)})

    missing_layers = sorted(REQUIRED_LAYERS - recognized)
    # A full Efficient-WAM export must contain all four I/O-dependent tensors.
    # Without this check a typo or an upstream key rename could leave one 14D
    # layer in the output and fail much later during model construction.
    if missing_layers:
        raise ValueError(
            "Checkpoint is missing required ActionExpert dimension-dependent "
            f"layers: {', '.join(missing_layers)}. "
            "Use inspect_checkpoint.py to review the exported key names."
        )

    report = {
        "target_state_dim": int(state_dim),
        "target_action_dim": int(action_dim),
        "old_state_dim": old_state_dim,
        "old_action_dim": old_action_dim,
        "index_map_old_to_new": list(index_map) if index_map is not None else None,
        "index_map_semantics": (
            "RoboTwin14_[L1..L6,Lgrip,R1..R6,Rgrip] -> "
            "realrobot16_[L1..L7,Lgrip,R1..R7,Rgrip]"
            if tuple(index_map or ()) == ROBOTWIN_14_TO_REAL_ROBOT_16
            else None
        ),
        "recognized_layers": sorted(recognized),
        "missing_layers": missing_layers,
        "transformed": transformed,
        "already_target": already_target,
        "transform_count": int(len(transformed)),
    }
    return output, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Exported Efficient-WAM checkpoint (.pt)")
    parser.add_argument("output", type=Path, help="Output checkpoint (.pt)")
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--old-state-dim", type=int, default=None)
    parser.add_argument("--old-action-dim", type=int, default=None)
    parser.add_argument(
        "--index-map",
        default=None,
        help=(
            "Comma-separated old-index to new-index mapping used for every expanded "
            "state/action channel. For the RoboTwin 14D -> real-robot 16D layout "
            "the default is 0,1,2,3,4,5,7,8,9,10,11,12,13,15."
        ),
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON transform report")
    parser.add_argument("--compact-output", type=Path, default=None, help="Also extract a Stage-1-compatible compact WAN checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and report without writing output")
    parser.add_argument("--allow-noop", action="store_true", help="Allow a checkpoint that is already target-dimensional")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Input checkpoint not found: {args.input}")
    payload = _torch_load(args.input)
    state_dict, state_key = _state_dict(payload)
    inferred_state_dim, inferred_action_dim = _config_dims(payload)
    old_state_dim = args.old_state_dim if args.old_state_dim is not None else inferred_state_dim
    old_action_dim = args.old_action_dim if args.old_action_dim is not None else inferred_action_dim
    index_map: Optional[Tuple[int, ...]] = None
    if args.index_map:
        try:
            index_map = tuple(int(item.strip()) for item in str(args.index_map).split(",") if item.strip())
        except ValueError as exc:
            raise SystemExit(f"Invalid --index-map {args.index_map!r}; expected comma-separated integers") from exc
    elif (
        args.state_dim == 16
        and args.action_dim == 16
        and old_state_dim in (None, 14)
        and old_action_dim in (None, 14)
    ):
        # The released RT model is a RoboTwin 14D checkpoint and this adapter's
        # target is the supplied 7-DoF-per-arm real-robot recording.
        index_map = ROBOTWIN_14_TO_REAL_ROBOT_16
    adapted_state, report = adapt_state_dict(
        state_dict,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        old_state_dim=old_state_dim,
        old_action_dim=old_action_dim,
        index_map=index_map,
    )
    report.update({
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "state_dict_location": state_key,
        "inferred_state_dim": inferred_state_dim,
        "inferred_action_dim": inferred_action_dim,
    })
    if report["transform_count"] == 0 and not args.allow_noop:
        raise SystemExit(
            "No dimension-expanding layers were transformed. Check checkpoint key names/config, "
            "or pass --allow-noop only when it is already target-dimensional."
        )
    if report["transform_count"] and report["already_target"]:
        mixed = ", ".join(item["kind"] for item in report["already_target"])
        raise SystemExit(
            "Checkpoint has a mixture of source and target-dimensional ActionExpert layers "
            f"({mixed}); refusing to write a partially adapted export."
        )
    if report["transform_count"] not in (0, len(REQUIRED_LAYERS)):
        raise SystemExit(
            "Expected all four dimension-dependent tensors to be transformed, "
            f"but transformed {report['transform_count']}."
        )
    if not args.dry_run:
        if state_key == "model":
            # Mutating the freshly loaded payload avoids a second multi-GB
            # in-memory copy of the RT checkpoint.
            output_payload = payload
            output_payload["model"] = adapted_state
        else:
            output_payload = adapted_state
        report["config_fields_updated"] = _set_config_dims(output_payload, args.state_dim, args.action_dim)
        _torch_save(output_payload, args.output)
        if args.compact_output is not None:
            compact_payload = _compact_payload_from_full(output_payload, adapted_state)
            _torch_save(compact_payload, args.compact_output)
            report["compact_output"] = str(args.compact_output.resolve())
            report["compact_tensor_count"] = len(compact_payload["model"])
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".json")
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
