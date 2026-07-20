"""Trusted Triton implementation for static FP16 RMSNorm."""

import torch
import triton
import triton.language as tl
from torch import nn

ROWS = 4096
COLUMNS = 4096
EPSILON = 1e-5


@triton.jit
def _rmsnorm_kernel(
    x_pointer,
    gamma_pointer,
    output_pointer,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_pointer + row * row_stride + columns).to(tl.float32)
    gamma = tl.load(gamma_pointer + columns).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / BLOCK_SIZE
    output = x * tl.rsqrt(mean_square + EPS) * gamma
    tl.store(output_pointer + row * row_stride + columns, output)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        _rmsnorm_kernel[(ROWS,)](
            x,
            gamma,
            output,
            x.stride(0),
            BLOCK_SIZE=COLUMNS,
            EPS=EPSILON,
            num_warps=8,
        )
        return output
