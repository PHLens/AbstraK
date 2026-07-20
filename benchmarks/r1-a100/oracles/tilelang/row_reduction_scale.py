"""Trusted TileLang implementation for the static row-reduction canary."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

ROWS = 1024
COLUMNS = 1024
SCALE = 0.5


def _build_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, COLUMNS), T.float16),
        output: T.Tensor((ROWS,), T.float16),
    ):
        with T.Kernel(ROWS, threads=128) as row:
            values = T.alloc_fragment((COLUMNS,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            total[0] = T.float32(0.0)
            for column in T.Parallel(COLUMNS):
                values[column] = T.cast(x[row, column], T.float32)
            T.reduce_sum(values, total, clear=False)
            if T.get_thread_binding() == 0:
                output[row] = total[0] * SCALE

    return tilelang.compile(kernel, out_idx=1, target="cuda")


class ModelNew(nn.Module):
    """Shape-specialized trusted candidate; the evaluator owns validation."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x)
