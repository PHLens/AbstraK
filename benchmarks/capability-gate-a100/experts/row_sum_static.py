"""B-legal FP32 row-sum expert for the capability-gate study."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

ROWS = 16384
COLUMNS = 4096


def _build_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, COLUMNS), T.float16),
        output: T.Tensor((ROWS,), T.float32),
    ):
        with T.Kernel(ROWS, threads=128) as row:
            values = T.alloc_fragment((COLUMNS,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for column in T.Parallel(COLUMNS):
                values[column] = T.cast(x[row, column], T.float32)
            total[0] = T.float32(0.0)
            T.reduce_sum(values, total, clear=False)
            output[row] = total[0]

    return tilelang.compile(kernel, out_idx=1, target="cuda")


class ModelNew(nn.Module):
    """Shape-specialized B expert; evaluation owns correctness and timing."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x)
