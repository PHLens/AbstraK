# NVIDIA CuTe DSL 4.6.1 / NVIDIA A100 Target Card

## Submission contract

Return one complete Python implementation that defines `ModelNew`. The evaluator imports the
module, constructs `ModelNew` with the task's initialization arguments, and calls `forward` with
contiguous CUDA tensors. Keep the task computation inside CuTe DSL kernels. PyTorch may be used for
module structure, output allocation, shapes, strides, and launch metadata, but not as a fallback for
the core operator.

Use these imports:

```python
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from torch import nn
```

Define device code with `@cute.kernel` and a host launcher with `@cute.jit`. Convert PyTorch CUDA
tensors using `from_dlpack`, compile the launcher with `cute.compile`, and cache the compiled
callable. Launch device code with `.launch(grid=(...), block=(...))`.

## Device and numeric rules

- Inputs are contiguous CUDA tensors on an NVIDIA A100 (SM80).
- Allocate outputs on the input device with the exact shape and dtype required by the task.
- Follow the task's stated accumulation precision. Convert scalar values with
  `.to(cutlass.Float32)` and initialize FP32 accumulators with `cutlass.Float32(0.0)` when needed.
- Do not move data to the CPU, call a PyTorch implementation of the core operator, or compile a
  custom CUDA extension.
- Shape specialization and compile-time constants are allowed for the frozen task shape.

Useful CuTe DSL building blocks include `cute.arch.thread_idx`, `cute.arch.block_idx`,
`cute.arch.block_dim`, `cutlass.range`, `cute.ceil_div`, `cute.make_layout`, `cute.copy`,
`cute.make_rmem_tensor`, and the SM80 primitives under `cute.nvgpu`. Predication is required when a
grid or tile can exceed a logical tensor dimension.

## Model scaffold and launch example

This VectorAdd example demonstrates the required kernel, DLPack bridge, compilation cache, and
launch structure. It is unrelated to the benchmark tasks.

```python
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from torch import nn


@cute.kernel
def _vector_add_kernel(x: cute.Tensor, y: cute.Tensor, output: cute.Tensor, length: int):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    block_size, _, _ = cute.arch.block_dim()
    index = block * block_size + thread
    if index < length:
        output[index] = x[index] + y[index]


@cute.jit
def _launch_vector_add(
    x: cute.Tensor,
    y: cute.Tensor,
    output: cute.Tensor,
    length: int,
):
    threads = 256
    _vector_add_kernel(x, y, output, length).launch(
        grid=(cute.ceil_div(length, threads), 1, 1),
        block=(threads, 1, 1),
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.compiled = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        cute_x = from_dlpack(x, assumed_align=16)
        cute_y = from_dlpack(y, assumed_align=16)
        cute_output = from_dlpack(output, assumed_align=16)
        length = x.numel()
        if self.compiled is None:
            self.compiled = cute.compile(
                _launch_vector_add,
                cute_x,
                cute_y,
                cute_output,
                length,
            )
        self.compiled(cute_x, cute_y, cute_output, length)
        return output
```

## Common failures

- Defining `Model` instead of `ModelNew`, returning extra values, or changing the expected signature.
- Forgetting the `from_dlpack` bridge, passing CPU tensors, or allocating outputs on the CPU.
- Compiling an `@cute.kernel` directly instead of compiling an `@cute.jit` host launcher.
- Using a grid with no tail predicate, or indexing a multidimensional tensor with the wrong layout.
- Letting FP16 input arithmetic determine the accumulator type when FP32 is required.
- Recompiling the same static launcher on every timed `forward` call.
- Reading an input after overwriting it, or modifying evaluator-owned inputs in place.
