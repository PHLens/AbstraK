"""B-legal gated-SiLU expert for the capability-gate study."""

import tilelang
import tilelang.language as T
import torch
from torch import nn

ROWS = 8192
COLUMNS = 4096
BLOCK = 128


def _build_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, COLUMNS), T.float16),
        gate: T.Tensor((ROWS, COLUMNS), T.float16),
        output: T.Tensor((ROWS, COLUMNS), T.float16),
    ):
        with T.Kernel(T.ceildiv(COLUMNS, BLOCK), ROWS, threads=128) as (block_col, row):
            output_tile = T.alloc_fragment((BLOCK,), T.float16)
            for column in T.Parallel(BLOCK):
                x_value = T.cast(x[row, block_col * BLOCK + column], T.float32)
                gate_value = T.cast(gate[row, block_col * BLOCK + column], T.float32)
                sigmoid = 1.0 / (1.0 + T.exp(-x_value))
                output_tile[column] = T.cast(x_value * sigmoid * gate_value, T.float16)
            T.copy(output_tile, output[row, block_col * BLOCK])

    return tilelang.compile(kernel, out_idx=2, target="cuda")


class ModelNew(nn.Module):
    """Shape-specialized B expert; evaluation owns correctness and timing."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, gate)
