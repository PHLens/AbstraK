"""B-legal wide RMSNorm expert for the capability-gate study."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

ROWS = 8192
COLUMNS = 4096
EPSILON = 1e-5


def _build_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, COLUMNS), T.float16),
        gamma: T.Tensor((COLUMNS,), T.float16),
        output: T.Tensor((ROWS, COLUMNS), T.float16),
    ):
        with T.Kernel(ROWS, threads=128) as row:
            x_shared = T.alloc_shared((1, COLUMNS), T.float16)
            gamma_shared = T.alloc_shared((COLUMNS,), T.float16)
            x_local = T.alloc_fragment((1, COLUMNS), T.float32)
            square_local = T.alloc_fragment((1, COLUMNS), T.float32)
            square_sum = T.alloc_fragment((1,), T.float32)
            T.copy(x[row, 0], x_shared)
            T.copy(gamma, gamma_shared)
            for _, column in T.Parallel(1, COLUMNS):
                value = T.cast(x_shared[0, column], T.float32)
                x_local[0, column] = value
                square_local[0, column] = value * value
            T.reduce_sum(square_local, square_sum, dim=1, clear=True)
            inverse_rms = T.rsqrt(square_sum[0] / T.float32(COLUMNS) + T.float32(EPSILON))
            for _, column in T.Parallel(1, COLUMNS):
                x_shared[0, column] = T.cast(
                    x_local[0, column] * inverse_rms * T.cast(gamma_shared[column], T.float32),
                    T.float16,
                )
            T.copy(x_shared, output[row, 0])

    return tilelang.compile(kernel, out_idx=2, target="cuda")


class ModelNew(nn.Module):
    """Shape-specialized B expert; evaluation owns correctness and timing."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, gamma)
