"""B-legal exact GELU expert for the capability-gate study."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

ROWS = 8192
COLUMNS = 4096
BLOCK = 128
INV_SQRT_TWO = 0.7071067811865476


def _build_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, COLUMNS), T.float16),
        output: T.Tensor((ROWS, COLUMNS), T.float16),
    ):
        with T.Kernel(T.ceildiv(COLUMNS, BLOCK), ROWS, threads=128) as (block_col, row):
            output_tile = T.alloc_fragment((BLOCK,), T.float16)
            for column in T.Parallel(BLOCK):
                value = T.cast(x[row, block_col * BLOCK + column], T.float32)
                gelu = 0.5 * value * (1.0 + T.erf(value * INV_SQRT_TWO))
                output_tile[column] = T.cast(gelu, T.float16)
            T.copy(output_tile, output[row, block_col * BLOCK])

    return tilelang.compile(kernel, out_idx=1, target="cuda")


class ModelNew(nn.Module):
    """Shape-specialized B expert; evaluation owns correctness and timing."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x)
