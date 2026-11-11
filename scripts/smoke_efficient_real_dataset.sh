#!/usr/bin/env bash
set -euo pipefail

# Backwards-compatible name retained for existing automation. The actual
# smoke path is now Classify Objects only and performs a one-episode,
# one-step Efficient-WAM check.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/smoke_efficientwam_classify.sh" "$@"
