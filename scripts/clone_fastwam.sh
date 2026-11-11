set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
FASTWAM_ROOT="${FASTWAM_ROOT:-$PROJECT_ROOT/vendor/FastWAM}"
FASTWAM_REF="${FASTWAM_REF:-main}"
mkdir -p "$(dirname -- "$FASTWAM_ROOT")"
if [ ! -d "$FASTWAM_ROOT/.git" ]; then
  git clone --branch "$FASTWAM_REF" --depth 1 https://github.com/yuantianyuan01/FastWAM.git "$FASTWAM_ROOT"
else
  git -C "$FASTWAM_ROOT" fetch --depth 1 origin "$FASTWAM_REF"
  git -C "$FASTWAM_ROOT" reset --hard "origin/$FASTWAM_REF"
fi
git -C "$FASTWAM_ROOT" rev-parse HEAD
