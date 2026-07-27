"""One-element copy used only to measure the TileLang execution floor."""

import torch
from torch import nn


class Model(nn.Module):
    """Return an independent one-element FP16 output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


def make_inputs(
    case_kind: str,
    seed: int,
    value: float | None = None,
) -> list[torch.Tensor]:
    """Construct one deterministic evaluator-owned CPU input."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if case_kind == "random":
        if value is not None:
            raise ValueError("random inputs do not accept value")
        x = torch.empty((1,), dtype=torch.float16, device="cpu")
        x.uniform_(-1.0, 1.0, generator=generator)
    elif case_kind == "zero":
        if value is not None:
            raise ValueError("zero inputs do not accept value")
        x = torch.zeros((1,), dtype=torch.float16, device="cpu")
    elif case_kind == "constant":
        if value is None:
            raise ValueError("constant inputs require value")
        x = torch.full((1,), value, dtype=torch.float16, device="cpu")
    else:
        raise ValueError(f"unsupported case kind: {case_kind}")
    return [x]


def get_inputs() -> list[torch.Tensor]:
    return make_inputs("random", seed=20260727)


def get_init_inputs() -> list[object]:
    return []
