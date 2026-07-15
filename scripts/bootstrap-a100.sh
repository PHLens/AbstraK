#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate-volume-tools.sh"
ABSTRAK_ROOT="${ABSTRAK_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VOLUME_ROOT="${ABSTRAK_VOLUME_ROOT:-$(dirname -- "$ABSTRAK_ROOT")}"
KERNELBENCH_ROOT="${KERNELBENCH_ROOT:-$VOLUME_ROOT/KernelBench}"
WHEELHOUSE_ARCHIVE="${ABSTRAK_WHEELHOUSE_ARCHIVE:-$VOLUME_ROOT/abstrak-gpu-wheelhouse-py310.tar}"
WHEELHOUSE="${ABSTRAK_WHEELHOUSE:-/tmp/abstrak-gpu-wheelhouse-py310}"
GPU_VENV="${ABSTRAK_GPU_VENV:-/tmp/abstrak-gpu-venv}"
GPU_CACHE="${ABSTRAK_GPU_CACHE:-/tmp/abstrak-uv-cache}"
EXPECTED_KERNELBENCH_COMMIT="423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
EXPECTED_WHEELHOUSE_SHA256="ae644076dd76cd3ed8e47931e1ca4bc044881e244024556a1cb4d05767520caf"

export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$VOLUME_ROOT/.uv/python}"
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required in PATH" >&2; exit 1; }
test -d "$ABSTRAK_ROOT/.git" || { echo "missing AbstraK checkout" >&2; exit 1; }
test -d "$KERNELBENCH_ROOT/.git" || { echo "missing KernelBench checkout" >&2; exit 1; }

for checkout in "$ABSTRAK_ROOT" "$KERNELBENCH_ROOT"; do
  if ! git config --global --get-all safe.directory 2>/dev/null | grep -Fqx "$checkout"; then
    git config --global --add safe.directory "$checkout"
  fi
done

actual_commit="$(git -C "$KERNELBENCH_ROOT" rev-parse HEAD)"
test "$actual_commit" = "$EXPECTED_KERNELBENCH_COMMIT" || {
  echo "KernelBench commit mismatch: $actual_commit" >&2
  exit 1
}
test -z "$(git -C "$KERNELBENCH_ROOT" status --porcelain=v1)" || {
  echo "KernelBench checkout is not clean" >&2
  exit 1
}

if ! test -d "$WHEELHOUSE"; then
  test -f "$WHEELHOUSE_ARCHIVE" || {
    echo "missing GPU wheelhouse archive: $WHEELHOUSE_ARCHIVE" >&2
    exit 1
  }
  staged_archive="/tmp/$(basename -- "$WHEELHOUSE_ARCHIVE")"
  cp "$WHEELHOUSE_ARCHIVE" "$staged_archive"
  actual_wheelhouse_sha256="$(sha256sum "$staged_archive" | cut -d' ' -f1)"
  test "$actual_wheelhouse_sha256" = "$EXPECTED_WHEELHOUSE_SHA256" || {
    echo "GPU wheelhouse archive checksum mismatch" >&2
    exit 1
  }
  tar -xf "$staged_archive" -C "$(dirname -- "$WHEELHOUSE")"
fi

if ! test -x "$GPU_VENV/bin/python"; then
  uv venv --python 3.10 "$GPU_VENV"
fi
UV_CACHE_DIR="$GPU_CACHE" uv pip install \
  --python "$GPU_VENV/bin/python" \
  --no-index \
  --find-links "$WHEELHOUSE" \
  --requirement "$WHEELHOUSE/requirements.txt"
UV_CACHE_DIR="$GPU_CACHE" uv pip check --python "$GPU_VENV/bin/python"

KERNELBENCH_ROOT="$KERNELBENCH_ROOT" "$GPU_VENV/bin/python" - <<'PY'
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, found {platform.python_version()}")

kernelbench_root = Path(os.environ["KERNELBENCH_ROOT"]).resolve()
sys.path.insert(0, str(kernelbench_root / "src"))

import torch
import triton
import tilelang
from cutlass import cute
from kernelbench import eval as kernelbench_eval

if torch.__version__.split("+")[0] != "2.13.0":
    raise SystemExit(f"expected PyTorch 2.13.0, found {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.get_device_capability(0) != (8, 0):
    raise SystemExit(f"expected A100 SM80, found {torch.cuda.get_device_capability(0)}")

driver = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ],
    text=True,
).splitlines()[0]
if tuple(int(part) for part in driver.split(".")) < (575, 51, 3):
    raise SystemExit(f"CuTe DSL requires NVIDIA driver >=575.51.03, found {driver}")

x = torch.arange(1024, device="cuda", dtype=torch.float16)
y = x + 1
torch.cuda.synchronize()
if not torch.equal(y, x + 1):
    raise SystemExit("Torch CUDA smoke check failed")

packages = {}
for name in (
    "torch",
    "triton",
    "tilelang",
    "nvidia-cutlass-dsl",
    "cuda-python",
    "cuda-bindings",
):
    packages[name] = importlib.metadata.version(name)

print(
    json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "driver": driver,
            "device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "packages": packages,
            "imports": {
                "triton": triton.__file__,
                "tilelang": tilelang.__file__,
                "cute": cute.__file__,
                "kernelbench_eval": kernelbench_eval.__file__,
            },
        },
        indent=2,
        sort_keys=True,
    )
)
PY
