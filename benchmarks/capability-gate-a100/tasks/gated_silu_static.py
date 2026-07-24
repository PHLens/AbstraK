"""Gated-SiLU workload for the TileLang capability-gate study."""

import torch
from torch import nn

ROWS = 8192
COLUMNS = 4096


class Model(nn.Module):
    """Compute silu(x) times gate in FP32 and return FP16."""

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        gate_fp32 = gate.to(torch.float32)
        gated = (x_fp32 * torch.sigmoid(x_fp32)) * gate_fp32
        return gated.to(torch.float16)


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
        gate = torch.empty((ROWS, COLUMNS), dtype=torch.float16, device="cpu")
        x.uniform_(-3.0, 3.0, generator=generator)
        gate.uniform_(-1.0, 1.0, generator=generator)
    elif case_kind == "zero":
        if value is not None:
            raise ValueError("zero inputs do not accept value")
        x = torch.zeros((ROWS, COLUMNS), dtype=torch.float16, device="cpu")
        gate = torch.ones((ROWS, COLUMNS), dtype=torch.float16, device="cpu")
    elif case_kind == "constant":
        if value is None:
            raise ValueError("constant inputs require value")
        x = torch.full((ROWS, COLUMNS), value, dtype=torch.float16, device="cpu")
        gate = torch.full((ROWS, COLUMNS), value, dtype=torch.float16, device="cpu")
    else:
        raise ValueError(f"unsupported case kind: {case_kind}")
    return [x, gate]


def get_inputs() -> list[torch.Tensor]:
    return make_inputs("random", seed=20260724)


def get_init_inputs() -> list[object]:
    return []
