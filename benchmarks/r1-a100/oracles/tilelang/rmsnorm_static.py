"""Trusted TileLang implementation for static FP16 RMSNorm."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

ROWS = 4096
COLUMNS = 4096
EPSILON = 1e-5


def _build_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, COLUMNS), T.float16),
        gamma: T.Tensor((COLUMNS,), T.float16),
        output: T.Tensor((ROWS, COLUMNS), T.float16),
    ):
        with T.Kernel(ROWS, threads=256) as row:
            x_shared = T.alloc_shared((1, COLUMNS), T.float16)
            gamma_shared = T.alloc_shared((COLUMNS,), T.float16)
            x_local = T.alloc_fragment((1, COLUMNS), T.float32)
            square_local = T.alloc_fragment((1, COLUMNS), T.float32)
            square_sum = T.alloc_fragment((1,), T.float32)
            T.copy(x[row, 0], x_shared)
            T.copy(gamma, gamma_shared)
            for _, column in T.Parallel(1, COLUMNS):
                value = T.Cast(T.float32, x_shared[0, column])
                x_local[0, column] = value
                square_local[0, column] = value * value
            T.reduce_sum(square_local, square_sum, dim=1)
            inverse_rms = T.rsqrt(square_sum[0] / COLUMNS + EPSILON)
            for _, column in T.Parallel(1, COLUMNS):
                x_shared[0, column] = T.Cast(
                    T.float16,
                    x_local[0, column] * inverse_rms * T.Cast(T.float32, gamma_shared[column]),
                )
            T.copy(x_shared, output[row, 0])

    return tilelang.compile(kernel, out_idx=2, target="cuda")


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, gamma)
