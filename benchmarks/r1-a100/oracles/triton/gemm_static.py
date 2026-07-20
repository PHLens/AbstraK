"""Trusted Triton implementation for the static FP16 GEMM."""

import torch
import triton
import triton.language as tl
from torch import nn

M = 1024
N = 4096
K = 4096
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 32


@triton.jit
def _gemm_kernel(
    a_pointer,
    b_pointer,
    output_pointer,
    M_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(axis=0)
    blocks_n = tl.cdiv(N_SIZE, BN)
    block_m = block // blocks_n
    block_n = block % blocks_n
    offsets_m = block_m * BM + tl.arange(0, BM)
    offsets_n = block_n * BN + tl.arange(0, BN)
    offsets_k = tl.arange(0, BK)
    a_pointers = a_pointer + offsets_m[:, None] * K_SIZE + offsets_k[None, :]
    b_pointers = b_pointer + offsets_k[:, None] * N_SIZE + offsets_n[None, :]
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K_SIZE, BK):
        a = tl.load(a_pointers)
        b = tl.load(b_pointers)
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
        a_pointers += BK
        b_pointers += BK * N_SIZE
    output_offsets = offsets_m[:, None] * N_SIZE + offsets_n[None, :]
    tl.store(output_pointer + output_offsets, accumulator)


class ModelNew(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        output = torch.empty((M, N), dtype=torch.float16, device=a.device)
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _gemm_kernel[grid](
            a,
            b,
            output,
            M_SIZE=M,
            N_SIZE=N,
            K_SIZE=K,
            BM=BLOCK_M,
            BN=BLOCK_N,
            BK=BLOCK_K,
            num_warps=8,
            num_stages=4,
        )
        return output
