"""B-legal masked expert for the irregular small-K GEMM workload."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

M = 8191
N = 8179
K = 80
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16


def _build_kernel():
    @T.prim_func
    def kernel(
        a: T.Tensor((M, K), T.float16),
        b: T.Tensor((K, N), T.float16),
        output: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N, BLOCK_N),
            T.ceildiv(M, BLOCK_M),
            threads=128,
        ) as (block_n, block_m):
            a_shared = T.alloc_shared((BLOCK_M, BLOCK_K), T.float16)
            b_shared = T.alloc_shared((BLOCK_K, BLOCK_N), T.float16)
            accumulator = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            T.clear(accumulator)
            for block_k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=0):
                for row, column in T.Parallel(BLOCK_M, BLOCK_K):
                    global_row = block_m * BLOCK_M + row
                    global_column = block_k * BLOCK_K + column
                    a_shared[row, column] = T.if_then_else(
                        (global_row < M) and (global_column < K),
                        a[global_row, global_column],
                        T.float16(0.0),
                    )
                for row, column in T.Parallel(BLOCK_K, BLOCK_N):
                    global_row = block_k * BLOCK_K + row
                    global_column = block_n * BLOCK_N + column
                    b_shared[row, column] = T.if_then_else(
                        (global_row < K) and (global_column < N),
                        b[global_row, global_column],
                        T.float16(0.0),
                    )
                T.gemm(a_shared, b_shared, accumulator)
            for row, column in T.Parallel(BLOCK_M, BLOCK_N):
                global_row = block_m * BLOCK_M + row
                global_column = block_n * BLOCK_N + column
                if (global_row < M) and (global_column < N):
                    output[global_row, global_column] = T.cast(accumulator[row, column], T.float16)

    return tilelang.compile(kernel, out_idx=2, target="cuda")


class ModelNew(nn.Module):
    """Shape-specialized B expert; tail masking is part of the frozen source."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b)
