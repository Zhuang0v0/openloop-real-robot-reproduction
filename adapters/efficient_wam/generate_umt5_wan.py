#!/usr/bin/env python3
"""Generate Efficient-WAM UMT5 embedding lists for a clean task folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True, help="Efficient-WAM checkout")
    parser.add_argument("--wan-root", type=Path, required=True, help="WAN-2.2 TI2V-5B model directory")
    parser.add_argument("--task-root", type=Path, required=True, help="clean/<task> directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(args.repo_root.resolve()))
    from data.common.text_embeddings import WanT5Embedder

    meta_dir = args.task_root / "metas"
    output_dir = args.task_root / "umt5_wan"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_files = sorted(meta_dir.glob("*.txt"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata files under {meta_dir}")

    embedder = WanT5Embedder(
        wan_root=args.wan_root,
        device=args.device,
        text_len=args.text_len,
        dtype=torch.bfloat16,
    )
    written = 0
    for meta_path in meta_files:
        target = output_dir / f"{meta_path.stem}.pt"
        if target.exists() and not args.overwrite:
            continue
        prompts = [line.strip() for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"No non-empty instruction in {meta_path}")
        embeddings = embedder.encode(prompts)
        if not embeddings or any(value.ndim != 2 or value.shape[1] != 4096 for value in embeddings):
            raise ValueError(f"Unexpected UMT5 output for {meta_path}: {[tuple(value.shape) for value in embeddings]}")
        tmp = target.with_suffix(target.suffix + ".partial")
        torch.save(embeddings, tmp)
        tmp.replace(target)
        written += 1
        print(f"{written}/{len(meta_files)} {target.name} shape={[tuple(value.shape) for value in embeddings]}", flush=True)
    print(f"generated={written} existing={len(meta_files) - written} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
