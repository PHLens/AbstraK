"""Exact GELU workload for the TileLang capability-gate study."""

import torch
from torch import nn

ROWS = 8192
COLUMNS = 4096
INV_SQRT_TWO = 0.7071067811865476


class Model(nn.Module):
    """Apply exact erf-based GELU in FP32 and return FP16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        gelu = 0.5 * x_fp32 * (1.0 + torch.erf(x_fp32 * INV_SQRT_TWO))
        return gelu.to(torch.float16)


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
        x.uniform_(-3.0, 3.0, generator=generator)
    elif case_kind == "zero":
        if value is not None:
            raise ValueError("zero inputs do not accept value")
        x = torch.zeros((ROWS, COLUMNS), dtype=torch.float16, device="cpu")
    elif case_kind == "constant":
        if value is None:
            raise ValueError("constant inputs require value")
        x = torch.full((ROWS, COLUMNS), value, dtype=torch.float16, device="cpu")
    else:
        raise ValueError(f"unsupported case kind: {case_kind}")
    return [x]


def get_inputs() -> list[torch.Tensor]:
    return make_inputs("random", seed=20260724)


def get_init_inputs() -> list[object]:
    return []
