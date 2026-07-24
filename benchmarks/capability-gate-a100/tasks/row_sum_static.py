"""Wide row-sum workload for the TileLang capability-gate study."""

import torch
from torch import nn

ROWS = 16384
COLUMNS = 4096


class Model(nn.Module):
    """Reduce the final dimension in FP32 and return FP32 row sums."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1, dtype=torch.float32)


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
        x.uniform_(-1.0, 1.0, generator=generator)
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
