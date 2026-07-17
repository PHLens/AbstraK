"""Trusted Triton implementation for the static row-reduction canary."""

import torch
import triton
import triton.language as tl
from torch import nn

ROWS = 1024
COLUMNS = 1024
SCALE = 0.5


@triton.jit
def _row_reduction_scale_kernel(
    x_pointer,
    output_pointer,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
    SCALE_FACTOR: tl.constexpr,
):
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x_pointer + row * row_stride + columns)
    result = tl.sum(values.to(tl.float32), axis=0) * SCALE_FACTOR
    tl.store(output_pointer + row, result)


class ModelNew(nn.Module):
    """Shape-specialized trusted candidate; the evaluator owns validation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty((ROWS,), dtype=torch.float16, device=x.device)
        _row_reduction_scale_kernel[(ROWS,)](
            x,
            output,
            x.stride(0),
            BLOCK_SIZE=COLUMNS,
            SCALE_FACTOR=SCALE,
            num_warps=8,
        )
        return output
