"""B-legal minimum TileLang kernel used to measure the execution floor."""

import tilelang
import tilelang.language as T
import torch
from torch import nn


def _build_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((1,), T.float16),
        output: T.Tensor((1,), T.float16),
    ):
        with T.Kernel(1, threads=128):
            for index in T.Parallel(1):
                output[index] = x[index]

    return tilelang.compile(kernel, out_idx=1, target="cuda")


class ModelNew(nn.Module):
    """Expose the frozen single-kernel launch-floor implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = _build_kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(x)
