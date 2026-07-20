"""Trusted TileLang implementation for the static FP16 GEMM."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

M = 1024
N = 4096
K = 4096
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32


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
            for block_k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
                T.copy(a[block_m * BLOCK_M, block_k * BLOCK_K], a_shared)
                T.copy(b[block_k * BLOCK_K, block_n * BLOCK_N], b_shared)
                T.gemm(a_shared, b_shared, accumulator)
            T.copy(accumulator, output[block_m * BLOCK_M, block_n * BLOCK_N])

    return tilelang.compile(kernel, out_idx=2, target="cuda")


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b)
