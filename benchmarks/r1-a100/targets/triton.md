# Triton 3.7.1 / NVIDIA A100 Target Card

## Submission contract

Return one complete Python implementation that defines `ModelNew`. The evaluator imports the
module, constructs `ModelNew` with the task's initialization arguments, and calls `forward` with
CUDA tensors. Keep all task computation inside Triton kernels. PyTorch may be used for module
structure, output allocation, shapes, strides, and launch metadata, but not as a fallback for the
core operator.

Use these imports:

```python
import torch
import triton
import triton.language as tl
from torch import nn
```

No separate build command is needed. Functions decorated with `@triton.jit` compile when launched.
Launch a kernel with `kernel[grid](arguments, META_PARAMETER=value)`, where `grid` is a tuple or a
callable receiving the launch metadata.

## Device and numeric rules

- Inputs are contiguous CUDA tensors on an NVIDIA A100 (SM80).
- Allocate outputs on the input device with the exact shape and dtype required by the task.
- Follow the task's stated accumulation precision. Use explicit `tl.float32` conversion when FP32
  arithmetic is required.
- Do not move data to the CPU, call a PyTorch implementation of the core operator, or compile a
  custom CUDA extension.
- Shape specialization and compile-time constants are allowed for the frozen task shape.

Common Triton building blocks include `tl.program_id`, `tl.arange`, masked `tl.load`/`tl.store`,
pointer arithmetic, `tl.where`, `tl.dot`, and `tl.constexpr` meta-parameters. `triton.cdiv` is useful
for launch-grid sizing. Every out-of-bounds load and store must use a correct mask.

## Model scaffold and launch example

This VectorAdd example demonstrates the required module, allocation, masking, and launch structure.
It is unrelated to the benchmark tasks.

```python
import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _vector_add_kernel(
    x_pointer,
    y_pointer,
    output_pointer,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_pointer + offsets, mask=mask)
    y = tl.load(y_pointer + offsets, mask=mask)
    tl.store(output_pointer + offsets, x + y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_elements = x.numel()
        block_size = 256
        grid = (triton.cdiv(n_elements, block_size),)
        _vector_add_kernel[grid](
            x,
            y,
            output,
            n_elements,
            BLOCK_SIZE=block_size,
        )
        return output
```

## Common failures

- Defining `Model` instead of `ModelNew`, returning extra values, or changing the expected signature.
- Allocating an output on the CPU or with the wrong dtype.
- Omitting masks for a launch dimension that can exceed the logical tensor extent.
- Passing runtime values where a `tl.constexpr` meta-parameter is required.
- Launching too many registers or excessive shared memory per program; reduce tile size or warps.
- Reading an input after overwriting it, or modifying evaluator-owned inputs in place.
