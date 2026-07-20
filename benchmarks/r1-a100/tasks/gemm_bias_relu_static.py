"""Static FP16 GEMM+bias+ReLU task for the A100 R1 formal matrix."""

import torch
from torch import nn

M = 1024
N = 4096
K = 4096


class Model(nn.Module):
    """Compute ReLU(A @ B + bias) with FP32 accumulation and return FP16."""

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        product = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        return torch.relu(product + bias.to(torch.float32)).to(torch.float16)


def make_inputs(
    case_kind: str,
    seed: int,
    value: float | None = None,
) -> list[torch.Tensor]:
    """Construct one deterministic evaluator-owned CPU input case."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bias = torch.empty((N,), dtype=torch.float16)
    bias.uniform_(-1.0, 1.0, generator=generator)
    if case_kind == "random":
        if value is not None:
            raise ValueError("random inputs do not accept value")
        a = torch.empty((M, K), dtype=torch.float16)
        b = torch.empty((K, N), dtype=torch.float16)
        a.uniform_(-1.0, 1.0, generator=generator)
        b.uniform_(-1.0, 1.0, generator=generator)
    elif case_kind == "zero":
        if value is not None:
            raise ValueError("zero inputs do not accept value")
        a = torch.zeros((M, K), dtype=torch.float16)
        b = torch.zeros((K, N), dtype=torch.float16)
        bias.fill_(-0.5)
    elif case_kind == "constant":
        if value is None:
            raise ValueError("constant inputs require value")
        a = torch.full((M, K), value, dtype=torch.float16)
        b = torch.full((K, N), value, dtype=torch.float16)
        bias.fill_(-0.5)
    else:
        raise ValueError(f"unsupported case kind: {case_kind}")
    return [a, b, bias]


def get_inputs() -> list[torch.Tensor]:
    return make_inputs("random", seed=20260717)


def get_init_inputs() -> list[object]:
    return []
