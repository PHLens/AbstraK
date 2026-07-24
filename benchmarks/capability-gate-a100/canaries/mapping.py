"""Mapping-only TileLang canary with an explicit warp reduction."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

ROWS = 16384
COLUMNS = 4096
ITEMS_PER_THREAD = 32
WARPS = 4


def _build_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, COLUMNS), T.float16),
        output: T.Tensor((ROWS,), T.float32),
    ):
        with T.Kernel(ROWS, threads=128) as row:
            lane = T.get_thread_binding(0)
            local = T.alloc_local((1,), T.float32)
            warp_sums = T.alloc_shared((WARPS,), T.float32)
            local[0] = T.float32(0.0)
            for item in T.serial(ITEMS_PER_THREAD):
                local[0] += T.cast(x[row, item * 128 + lane], T.float32)
            local[0] = T.warp_reduce_sum(local[0])
            if lane % 32 == 0:
                warp_sums[lane // 32] = local[0]
            T.sync_threads()
            if lane < 32:
                if lane < WARPS:
                    local[0] = warp_sums[lane]
                else:
                    local[0] = T.float32(0.0)
                local[0] = T.warp_reduce_sum(local[0])
                if lane == 0:
                    output[row] = local[0]

    return tilelang.compile(kernel, out_idx=1, target="cuda")


class ModelNew(nn.Module):
    """Mapping canary bound to the ``row-sum-static`` contract."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x)
