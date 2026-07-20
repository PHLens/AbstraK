"""Trusted CuTe DSL implementation for the static matmul-plus-bias canary."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from torch import nn

M = 256
N = 256
K = 256


@cute.kernel
def _matmul_bias_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
    m_size: int,
    n_size: int,
    k_size: int,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    block_size, _, _ = cute.arch.block_dim()
    index = block * block_size + thread
    if index < m_size * n_size:
        row = index // n_size
        column = index % n_size
        accumulator = cutlass.Float32(0.0)
        for k_index in cutlass.range(k_size):
            a_value = a[row, k_index].to(cutlass.Float32)
            b_value = b[k_index, column].to(cutlass.Float32)
            accumulator += a_value * b_value
        result = accumulator + bias[column].to(cutlass.Float32)
        output[row, column] = result.to(output.element_type)


@cute.jit
def _launch_matmul_bias(
    a: cute.Tensor,
    b: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
    m_size: int,
    n_size: int,
    k_size: int,
):
    threads = 256
    blocks = cute.ceil_div(m_size * n_size, threads)
    _matmul_bias_kernel(a, b, bias, output, m_size, n_size, k_size).launch(
        grid=(blocks, 1, 1),
        block=(threads, 1, 1),
    )


class ModelNew(nn.Module):
    """Shape-specialized trusted candidate; the evaluator owns validation."""

    def __init__(self) -> None:
        super().__init__()
        self.compiled = None

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty((M, N), dtype=torch.float16, device=a.device)
        cute_a = from_dlpack(a, assumed_align=16)
        cute_b = from_dlpack(b, assumed_align=16)
        cute_bias = from_dlpack(bias, assumed_align=16)
        cute_output = from_dlpack(output, assumed_align=16)
        if self.compiled is None:
            self.compiled = cute.compile(
                _launch_matmul_bias,
                cute_a,
                cute_b,
                cute_bias,
                cute_output,
                M,
                N,
                K,
            )
        self.compiled(cute_a, cute_b, cute_bias, cute_output, M, N, K)
        return output
