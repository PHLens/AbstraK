#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this script instead of executing it" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate-volume-tools.sh"
export ABSTRAK_ROOT="${ABSTRAK_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export ABSTRAK_VOLUME_ROOT="${ABSTRAK_VOLUME_ROOT:-$(dirname -- "$ABSTRAK_ROOT")}"
export KERNELBENCH_ROOT="${KERNELBENCH_ROOT:-$ABSTRAK_VOLUME_ROOT/KernelBench}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ABSTRAK_VOLUME_ROOT/.uv/python}"
export ABSTRAK_GPU_VENV="${ABSTRAK_GPU_VENV:-/tmp/abstrak-gpu-venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/abstrak-uv-cache}"
export PYTHONPATH="$ABSTRAK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

test -x "$ABSTRAK_GPU_VENV/bin/python" || {
  echo "missing GPU environment; run scripts/bootstrap-a100.sh first" >&2
  return 1
}
source "$ABSTRAK_GPU_VENV/bin/activate"

abstrak-doctor() { python -m abstrak.doctor "$@"; }
abstrak-kernelbench() { python -m abstrak.evaluation.cli "$@"; }
abstrak-provider() { python -m abstrak.providers.cli "$@"; }

cd "$ABSTRAK_ROOT"
