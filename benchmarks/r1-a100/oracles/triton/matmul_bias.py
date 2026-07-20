"""Trusted Triton implementation for the static matmul-plus-bias canary."""

import torch
import triton
import triton.language as tl
from torch import nn

M = 256
N = 256
K = 256


@triton.jit
def _matmul_bias_kernel(
    a_pointer,
    b_pointer,
    bias_pointer,
    output_pointer,
    M_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_m = tl.program_id(axis=0)
    block_n = tl.program_id(axis=1)
    offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    a_pointers = a_pointer + offsets_m[:, None] * K_SIZE + offsets_k[None, :]
    b_pointers = b_pointer + offsets_k[:, None] * N_SIZE + offsets_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K_SIZE, BLOCK_K)):
        a_values = tl.load(
            a_pointers,
            mask=(offsets_m[:, None] < M_SIZE) & (offsets_k[None, :] < K_SIZE),
            other=0.0,
        )
        b_values = tl.load(
            b_pointers,
            mask=(offsets_k[:, None] < K_SIZE) & (offsets_n[None, :] < N_SIZE),
            other=0.0,
        )
        accumulator += tl.dot(a_values, b_values, out_dtype=tl.float32)
        a_pointers += BLOCK_K
        b_pointers += BLOCK_K * N_SIZE
        offsets_k += BLOCK_K

    bias = tl.load(bias_pointer + offsets_n, mask=offsets_n < N_SIZE, other=0.0)
    output = accumulator + bias[None, :].to(tl.float32)
    output_offsets = offsets_m[:, None] * N_SIZE + offsets_n[None, :]
    output_mask = (offsets_m[:, None] < M_SIZE) & (offsets_n[None, :] < N_SIZE)
    tl.store(output_pointer + output_offsets, output, mask=output_mask)


class ModelNew(nn.Module):
    """Shape-specialized trusted candidate; the evaluator owns validation."""

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty((M, N), dtype=torch.float16, device=a.device)
        _matmul_bias_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
            a,
            b,
            bias,
            output,
            M_SIZE=M,
            N_SIZE=N,
            K_SIZE=K,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
            num_stages=3,
        )
        return output
