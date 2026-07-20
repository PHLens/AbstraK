"""Trusted Triton implementation for static FP16 LayerNorm."""

import torch
import triton
import triton.language as tl
from torch import nn

ROWS = 4096
COLUMNS = 4096
EPSILON = 1e-5


@triton.jit
def _layernorm_kernel(
    x_pointer,
    gamma_pointer,
    beta_pointer,
    output_pointer,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_pointer + row * row_stride + columns).to(tl.float32)
    mean = tl.sum(x, axis=0) / BLOCK_SIZE
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / BLOCK_SIZE
    gamma = tl.load(gamma_pointer + columns).to(tl.float32)
    beta = tl.load(beta_pointer + columns).to(tl.float32)
    output = centered * tl.rsqrt(variance + EPS) * gamma + beta
    tl.store(output_pointer + row * row_stride + columns, output)


class ModelNew(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty_like(x)
        _layernorm_kernel[(ROWS,)](
            x,
            gamma,
            beta,
            output,
            x.stride(0),
            BLOCK_SIZE=COLUMNS,
            EPS=EPSILON,
            num_warps=8,
        )
        return output
