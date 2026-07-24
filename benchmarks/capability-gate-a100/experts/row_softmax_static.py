"""B-legal stable row-softmax expert for the capability-gate study."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

ROWS = 8192
COLUMNS = 4096


def _build_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, COLUMNS), T.float16),
        output: T.Tensor((ROWS, COLUMNS), T.float16),
    ):
        with T.Kernel(ROWS, threads=128) as row:
            x_shared = T.alloc_shared((1, COLUMNS), T.float16)
            x_local = T.alloc_fragment((1, COLUMNS), T.float32)
            exponentials = T.alloc_fragment((1, COLUMNS), T.float32)
            row_max = T.alloc_fragment((1,), T.float32)
            row_sum = T.alloc_fragment((1,), T.float32)
            T.copy(x[row, 0], x_shared)
            for _, column in T.Parallel(1, COLUMNS):
                x_local[0, column] = T.cast(x_shared[0, column], T.float32)
            T.reduce_max(x_local, row_max, dim=1, clear=True)
            for _, column in T.Parallel(1, COLUMNS):
                exponentials[0, column] = T.exp(x_local[0, column] - row_max[0])
            T.reduce_sum(exponentials, row_sum, dim=1, clear=True)
            for _, column in T.Parallel(1, COLUMNS):
                x_shared[0, column] = T.cast(exponentials[0, column] / row_sum[0], T.float16)
            T.copy(x_shared, output[row, 0])

    return tilelang.compile(kernel, out_idx=1, target="cuda")


class ModelNew(nn.Module):
    """Shape-specialized B expert; evaluation owns correctness and timing."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x)
