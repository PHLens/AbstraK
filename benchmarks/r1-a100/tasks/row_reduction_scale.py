"""Static row-reduction canary used by the A100 R1 shakeout."""

import torch
from torch import nn

ROWS = 1024
COLUMNS = 1024
SCALE = 0.5


class Model(nn.Module):
    """Sum each FP16 row in FP32, scale it, and return FP16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.sum(x, dim=1, dtype=torch.float32) * SCALE).to(torch.float16)


def make_inputs(
    case_kind: str,
    seed: int,
    value: float | None = None,
) -> list[torch.Tensor]:
    """Construct one deterministic evaluator-owned input case on CPU."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if case_kind == "random":
        if value is not None:
            raise ValueError("random inputs do not accept value")
        x = torch.empty((ROWS, COLUMNS), dtype=torch.float16)
        x.uniform_(-1.0, 1.0, generator=generator)
    elif case_kind == "zero":
        if value is not None:
            raise ValueError("zero inputs do not accept value")
        x = torch.zeros((ROWS, COLUMNS), dtype=torch.float16)
    elif case_kind == "constant":
        if value is None:
            raise ValueError("constant inputs require value")
        x = torch.full((ROWS, COLUMNS), value, dtype=torch.float16)
    else:
        raise ValueError(f"unsupported case kind: {case_kind}")
    return [x]


def get_inputs() -> list[torch.Tensor]:
    return make_inputs("random", seed=20260717)


def get_init_inputs() -> list[object]:
    return []
