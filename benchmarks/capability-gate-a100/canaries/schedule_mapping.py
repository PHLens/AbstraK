"""Full schedule-plus-mapping canary with an explicit synchronization point."""

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
            threads=256,
        ) as (block_n, block_m):
            a_shared = T.alloc_shared((BLOCK_M, BLOCK_K), T.float16)
            b_shared = T.alloc_shared((BLOCK_K, BLOCK_N), T.float16)
            accumulator = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            local = T.alloc_local((1,), T.float32)
            lane = T.get_thread_binding(0)
            T.clear(accumulator)
            for block_k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=1):
                T.copy(a[block_m * BLOCK_M, block_k * BLOCK_K], a_shared)
                T.copy(b[block_k * BLOCK_K, block_n * BLOCK_N], b_shared)
                T.gemm(
                    a_shared,
                    b_shared,
                    accumulator,
                    policy=T.GemmWarpPolicy.FullCol,
                )
            local[0] = T.float32(lane)
            T.sync_threads()
            T.copy(accumulator, output[block_m * BLOCK_M, block_n * BLOCK_N])

    return tilelang.compile(kernel, out_idx=2, target="cuda")


class ModelNew(nn.Module):
    """Interaction canary bound to ``gemm-large-k-static``; sync is frozen."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b)
