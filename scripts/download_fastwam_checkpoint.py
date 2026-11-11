"""Stream the public FastWAM RoboTwin checkpoint with resumable verification."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import requests


URL = "https://hf-mirror.com/yuanty/fastwam/resolve/main/robotwin_uncond_3cam_384.pt"
EXPECTED_BYTES = 12_041_813_092
EXPECTED_SHA256 = "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default=os.environ.get("FASTWAM_CHECKPOINT_URL", URL))
    parser.add_argument(
        "--expected-bytes",
        type=int,
        default=int(os.environ.get("FASTWAM_EXPECTED_BYTES", str(EXPECTED_BYTES))),
        help="Expected size; use 0 to disable the size check",
    )
    parser.add_argument(
        "--sha256",
        default=os.environ.get("FASTWAM_CHECKPOINT_SHA256", EXPECTED_SHA256),
        help="Expected SHA-256; pass an empty string to skip the hash check",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(args.url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as response:
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        expected_total = args.expected_bytes or None
        if offset and (
            response.status_code != 206
            or (expected_total and not re.match(rf"bytes {offset}-\d+/{expected_total}$", content_range))
        ):
            partial.unlink()
            offset = 0
            response.close()
            return main()
        total = expected_total if content_range and expected_total else offset + int(response.headers.get("Content-Length", "0"))
        if expected_total and total and total != expected_total:
            raise RuntimeError(f"Unexpected download size: {total} (expected {expected_total})")
        mode = "ab" if offset else "wb"
        completed = offset
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    completed += len(chunk)
                    if completed % (256 * 1024 * 1024) < len(chunk):
                        print(f"downloaded={completed}/{total or 'unknown'}", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
    if expected_total and completed != expected_total:
        raise RuntimeError(f"Incomplete download: {completed} bytes (expected {expected_total})")
    digest = hashlib.sha256()
    with partial.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    print(f"sha256={actual_sha256}", flush=True)
    if args.sha256 and actual_sha256 != args.sha256:
        raise RuntimeError(f"SHA-256 mismatch: {actual_sha256} != {args.sha256}")
    partial.replace(args.output)
    print(f"ready={args.output} bytes={args.output.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
