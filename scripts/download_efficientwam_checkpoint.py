#!/usr/bin/env python3
"""Download and verify the public Efficient-WAM-RT RoboTwin export."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import requests


CHECKPOINT_URL = (
    "https://hf-mirror.com/jiajun0613/Efficient-WAM_RoboTwin/resolve/main/"
    "Efficient-WAM-RT/Efficient-WAM-RT_stage3.pt"
)
CHECKPOINT_BYTES = 1_976_610_407
CHECKPOINT_SHA256 = "209ab4d6f897276633e4d0f36e6b0c573c4938cf0df99d974b05e754ab92340f"
STATS_URL = (
    "https://hf-mirror.com/jiajun0613/Efficient-WAM_RoboTwin/resolve/main/"
    "Efficient-WAM-RT/Efficient-WAM-RT_action_stats.json"
)
STATS_BYTES = 1_772


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path, *, expected_bytes: int = 0, expected_sha256: str = "") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0

    while True:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(30, 120),
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            range_ok = bool(
                offset
                and response.status_code == 206
                and re.match(rf"bytes {offset}-\d+/(\d+|\*)$", content_range)
            )
            if offset and not range_ok:
                partial.unlink(missing_ok=True)
                offset = 0
                continue

            content_length = int(response.headers.get("Content-Length", "0"))
            total = expected_bytes or (
                int(content_range.rsplit("/", 1)[1])
                if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit()
                else offset + content_length
            )
            if expected_bytes and total and total != expected_bytes:
                raise RuntimeError(f"Unexpected download size: {total} (expected {expected_bytes})")

            completed = offset
            mode = "ab" if offset else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        completed += len(chunk)
                        if completed % (256 * 1024 * 1024) < len(chunk):
                            print(f"downloaded={completed}/{total or 'unknown'}", flush=True)
                handle.flush()
                os.fsync(handle.fileno())
            break

    if expected_bytes and completed != expected_bytes:
        raise RuntimeError(f"Incomplete download: {completed} bytes (expected {expected_bytes})")
    actual_sha256 = _sha256(partial)
    print(f"sha256={actual_sha256} file={partial}", flush=True)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch: {actual_sha256} != {expected_sha256}")
    partial.replace(output)
    print(f"ready={output} bytes={output.stat().st_size}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Destination for Efficient-WAM-RT_stage3.pt")
    parser.add_argument("--url", default=os.environ.get("EFFICIENT_CHECKPOINT_URL", CHECKPOINT_URL))
    parser.add_argument(
        "--expected-bytes",
        type=int,
        default=int(os.environ.get("EFFICIENT_EXPECTED_BYTES", str(CHECKPOINT_BYTES))),
        help="Expected checkpoint size; use 0 to disable the size check",
    )
    parser.add_argument(
        "--sha256",
        default=os.environ.get("EFFICIENT_CHECKPOINT_SHA256", CHECKPOINT_SHA256),
        help="Expected checkpoint SHA-256; pass an empty string to skip the hash check",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional destination for the matching Efficient-WAM action statistics JSON",
    )
    args = parser.parse_args()
    download(
        args.url,
        args.output,
        expected_bytes=args.expected_bytes,
        expected_sha256=args.sha256,
    )
    if args.stats_output is not None:
        stats_url = os.environ.get("EFFICIENT_STATS_URL", STATS_URL)
        download(stats_url, args.stats_output, expected_bytes=STATS_BYTES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
