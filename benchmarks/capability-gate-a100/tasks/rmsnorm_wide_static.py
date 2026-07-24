"""Wide RMSNorm workload for the TileLang capability-gate study."""

import torch
from torch import nn

ROWS = 8192
COLUMNS = 4096
EPSILON = 1e-5


class Model(nn.Module):
    """Apply row-wise RMSNorm with FP32 statistics and an FP16 gamma."""

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        mean_square = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
        normalized = x_fp32 * torch.rsqrt(mean_square + EPSILON)
        return (normalized * gamma.to(torch.float32)).to(torch.float16)


def make_inputs(
    case_kind: str,
    seed: int,
    value: float | None = None,
) -> list[torch.Tensor]:
    """Construct one deterministic evaluator-owned CPU input case."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if case_kind == "random":
        if value is not None:
            raise ValueError("random inputs do not accept value")
        x = torch.empty((ROWS, COLUMNS), dtype=torch.float16, device="cpu")
        gamma = torch.empty((COLUMNS,), dtype=torch.float16, device="cpu")
        x.uniform_(-1.0, 1.0, generator=generator)
        gamma.uniform_(0.5, 1.5, generator=generator)
    elif case_kind == "zero":
        if value is not None:
            raise ValueError("zero inputs do not accept value")
        x = torch.zeros((ROWS, COLUMNS), dtype=torch.float16, device="cpu")
        gamma = torch.ones((COLUMNS,), dtype=torch.float16, device="cpu")
    elif case_kind == "constant":
        if value is None:
            raise ValueError("constant inputs require value")
        x = torch.full((ROWS, COLUMNS), value, dtype=torch.float16, device="cpu")
        gamma = torch.full((COLUMNS,), value, dtype=torch.float16, device="cpu")
    else:
        raise ValueError(f"unsupported case kind: {case_kind}")
    return [x, gamma]


def get_inputs() -> list[torch.Tensor]:
    return make_inputs("random", seed=20260724)


def get_init_inputs() -> list[object]:
    return []
