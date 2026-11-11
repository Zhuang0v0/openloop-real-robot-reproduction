"""Expand a 14-D RoboTwin Fast-WAM checkpoint to the 16-D real-robot layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

INDEX_MAP = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15)


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_location(payload: Any):
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(payload.get(key), dict):
                return payload[key], key
        if payload and all(isinstance(k, str) for k in payload):
            return payload, None
    raise TypeError("Fast-WAM checkpoint does not contain a state dict")


def expand_cols(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros((x.shape[0], 16), dtype=x.dtype)
    for old, new in enumerate(INDEX_MAP):
        y[:, new] = x[:, old]
    return y


def expand_rows(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros((16, x.shape[1]), dtype=x.dtype)
    for old, new in enumerate(INDEX_MAP):
        y[new, :] = x[old, :]
    return y


def expand_bias(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros((16,), dtype=x.dtype)
    for old, new in enumerate(INDEX_MAP):
        y[new] = x[old]
    return y


def update_config(obj: Any) -> int:
    changed = 0
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in {"action_dim", "proprio_dim", "state_dim"} and value == 14:
                obj[key] = 16
                changed += 1
            else:
                changed += update_config(value)
    elif isinstance(obj, list):
        for value in obj:
            changed += update_config(value)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    payload = load(args.input)
    state, location = state_location(payload)
    output = dict(state)
    transformed = []
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        k = key.lower()
        shape = tuple(value.shape)
        if value.ndim == 2 and shape[1] == 14 and any(token in k for token in ("action_encoder", "proprio_encoder", "state_encoder")):
            output[key] = expand_cols(value)
            transformed.append({"key": key, "kind": "input_columns", "from": list(shape), "to": list(output[key].shape)})
        elif value.ndim == 2 and shape[0] == 14 and any(token in k for token in ("action_head", "head")):
            output[key] = expand_rows(value)
            transformed.append({"key": key, "kind": "output_rows", "from": list(shape), "to": list(output[key].shape)})
        elif value.ndim == 1 and shape[0] == 14 and any(token in k for token in ("action_head", "head")):
            output[key] = expand_bias(value)
            transformed.append({"key": key, "kind": "output_bias", "from": list(shape), "to": list(output[key].shape)})
    if not transformed:
        raise RuntimeError("No 14-D Fast-WAM input/output tensors matched; inspect checkpoint keys before adapting")
    if location is None:
        result = output
    else:
        result = payload
        result[location] = output
    config_updates = update_config(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".partial")
    torch.save(result, tmp)
    tmp.replace(args.output)
    report = {
        "input": str(args.input), "output": str(args.output),
        "index_map_old_to_new": list(INDEX_MAP),
        "transformed": transformed, "config_updates": config_updates,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
