#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${REPO_ROOT}/dist"
VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-}"

usage() {
    cat <<'EOF'
Usage: bash scripts/build_wheel.sh [options]
  --output-dir DIR  Wheel destination (default: repository/dist)
  --python PATH     Python interpreter (default: active python)
  --version VALUE   Explicit version, required for sources without Git metadata
  -h, --help        Show help

Builds only SGLang, without installing it or upgrading runtime dependencies.
Uses preinstalled pip, setuptools, wheel and setuptools-scm; no network access.
Rust extensions default to none for HTTP serving. Set SGLANG_BUILD_RUST_EXTS
explicitly to build them with the corresponding Rust/PyTorch toolchain.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir|--python|--version)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo "Error: $1 requires a value." >&2
                exit 1
            fi
            case "$1" in
                --output-dir) OUTPUT_DIR="$2" ;;
                --python) PYTHON="$2" ;;
                --version) VERSION="$2" ;;
            esac
            shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
"$PYTHON" -c 'import pip, setuptools_scm, wheel'
if [ -z "$VERSION" ] && [ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null || true)" != "$REPO_ROOT" ]; then
    echo 'Error: sources without Git metadata require --version VALUE.' >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
export SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"
if [ -n "$VERSION" ]; then
    export SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION"
fi

# Stale build output can otherwise be recursively included in the next wheel.
"$PYTHON" - "$REPO_ROOT/python" <<'PY'
from pathlib import Path
import shutil
import sys

source = Path(sys.argv[1]).resolve()
build = source / 'build'
if build.is_symlink():
    raise SystemExit(f'Refusing to remove symlink: {build}')
if build.exists():
    shutil.rmtree(build)
PY

cd "$REPO_ROOT/python"
"$PYTHON" -m pip wheel --no-index --no-deps --no-build-isolation . -w "$OUTPUT_DIR"
printf 'Wheel output: %s\n' "$OUTPUT_DIR"
