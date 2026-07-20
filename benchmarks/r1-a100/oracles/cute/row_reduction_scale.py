"""Trusted CuTe DSL implementation for the static row-reduction canary."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from torch import nn

ROWS = 1024
COLUMNS = 1024
SCALE = 0.5


@cute.kernel
def _row_reduction_scale_kernel(
    x: cute.Tensor,
    output: cute.Tensor,
    columns: cutlass.Int32,
    scale: cutlass.Float32,
):
    thread, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    if thread == 0:
        accumulator = cutlass.Float32(0.0)
        for column in cutlass.range(columns):
            accumulator += x[row, column].to(cutlass.Float32)
        output[row] = (accumulator * scale).to(output.element_type)


@cute.jit
def _launch_row_reduction_scale(
    x: cute.Tensor,
    output: cute.Tensor,
    columns: cutlass.Int32,
    scale: cutlass.Float32,
):
    _row_reduction_scale_kernel(x, output, columns, scale).launch(
        grid=(x.shape[0], 1, 1),
        block=(128, 1, 1),
    )


class ModelNew(nn.Module):
    """Shape-specialized trusted candidate; the evaluator owns validation."""

    def __init__(self) -> None:
        super().__init__()
        self.compiled = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty((ROWS,), dtype=torch.float16, device=x.device)
        cute_x = from_dlpack(x, assumed_align=16)
        cute_output = from_dlpack(output, assumed_align=16)
        columns = cutlass.Int32(COLUMNS)
        scale = cutlass.Float32(SCALE)
        if self.compiled is None:
            self.compiled = cute.compile(
                _launch_row_reduction_scale,
                cute_x,
                cute_output,
                columns,
                scale,
            )
        self.compiled(cute_x, cute_output, columns, scale)
        return output
