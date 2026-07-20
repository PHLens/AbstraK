# TileLang 0.1.12 / NVIDIA A100 Target Card

## Submission contract

Return one complete Python implementation that defines `ModelNew`. The evaluator imports the
module, constructs `ModelNew` with the task's initialization arguments, and calls `forward` with
contiguous CUDA tensors. Keep the task computation inside TileLang kernels. PyTorch may be used for
module structure, output allocation, shapes, strides, and launch metadata, but not as a fallback for
the core operator.

Use these imports:

```python
import torch
import tilelang
import tilelang.language as T
from torch import nn
```

Define a kernel with `@T.prim_func`, then compile it with
`tilelang.compile(kernel, out_idx=OUTPUT_ARGUMENT_INDEX, target="cuda")`. `out_idx` tells the
runtime which kernel argument it should allocate and return. Compilation may be cached in
`ModelNew.__init__` or on first use.

## Device and numeric rules

- Inputs are contiguous CUDA tensors on an NVIDIA A100 (SM80).
- Declare every tensor's exact static shape and dtype with `T.Tensor`.
- Follow the task's stated accumulation precision. Use `T.float32` fragments or explicit `T.cast`
  when FP32 arithmetic is required.
- Do not move data to the CPU, call a PyTorch implementation of the core operator, or compile a
  custom CUDA extension.
- Shape specialization and compile-time constants are allowed for the frozen task shape.

Useful TileLang primitives include `T.Kernel`, `T.Parallel`, `T.Pipelined`, `T.alloc_shared`,
`T.alloc_fragment`, `T.copy`, `T.gemm`, `T.reduce_sum`, `T.clear`, `T.cast`, `T.max`, and
`T.ceildiv`. Use bounds checks whenever a launch or tile can exceed a logical tensor dimension.

## Model scaffold and launch example

This VectorAdd example demonstrates the required module, output contract, and compilation path. It
is unrelated to the benchmark tasks.

```python
import torch
import tilelang
import tilelang.language as T
from torch import nn

LENGTH = 4096
BLOCK = 256


def _build_vector_add():
    @T.prim_func
    def kernel(
        x: T.Tensor((LENGTH,), T.float16),
        y: T.Tensor((LENGTH,), T.float16),
        output: T.Tensor((LENGTH,), T.float16),
    ):
        with T.Kernel(T.ceildiv(LENGTH, BLOCK), threads=128) as block:
            for offset in T.Parallel(BLOCK):
                index = block * BLOCK + offset
                if index < LENGTH:
                    output[index] = x[index] + y[index]

    return tilelang.compile(kernel, out_idx=2, target="cuda")


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = _build_vector_add()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)
```

## Common failures

- Defining `Model` instead of `ModelNew`, returning extra values, or changing the expected signature.
- Using the wrong `out_idx`, tensor shape, or dtype in `T.Tensor` declarations.
- Reading outside a partial tile or launching a grid with swapped axes.
- Accumulating reductions or matrix products in FP16 when the task requires FP32.
- Calling `T.gemm` with shared/fragment shapes that are not compatible with the SM80 MMA path.
- Recompiling the same static kernel on every timed `forward` call.
- Reading an input after overwriting it, or modifying evaluator-owned inputs in place.
