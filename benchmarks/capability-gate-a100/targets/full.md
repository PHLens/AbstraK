# TileLang 0.1.12 / NVIDIA A100 / tileops-full

Target ID: `tilelang-a100-full`
Capability contract SHA-256: `038a007372a38251461a561cce3f1a1c70e64238a118b5ab30bb09d55321d02e`

Return one Python source defining `ModelNew`. Use exactly these imports:

```python
import torch
import tilelang
import tilelang.language as T
from torch import nn
```

Define kernels with `@T.prim_func` and compile them for `target="cuda"`.
Only direct calls from the surface below are accepted.

## Base surface

- `T.Kernel`
- `T.Parallel`
- `T.Pipelined`
- `T.Tensor`
- `T.alloc_fragment`
- `T.alloc_shared`
- `T.cast`
- `T.ceildiv`
- `T.clear`
- `T.copy`
- `T.erf`
- `T.exp`
- `T.fill`
- `T.float16`
- `T.float32`
- `T.gemm`
- `T.if_then_else`
- `T.infinity`
- `T.int32`
- `T.int64`
- `T.max`
- `T.min`
- `T.pow`
- `T.prim_func`
- `T.reduce_max`
- `T.reduce_min`
- `T.reduce_sum`
- `T.rsqrt`
- `T.sqrt`
- `T.tanh`
- `tilelang.compile`

## Control domains

- `T.Kernel(..., threads=...)`: {64, 128, 256}
- `T.Pipelined(..., num_stages=...)`: {0, 1, 2, 3}
- `T.Parallel(...)`: positional extents only
- shared/fragment allocation: positional shape; dtype positional or keyword
- `T.gemm` transpose flags: literal booleans only
- high-level reductions: literal `dim` and `clear` controls only
- `T.gemm(..., policy=...)`: {T.GemmWarpPolicy.Square, T.GemmWarpPolicy.FullRow, T.GemmWarpPolicy.FullCol}

## Mapping surface

- `T.alloc_local`
- `T.annotate_layout`
- `T.get_thread_binding`
- `T.serial`
- `T.sync_threads`
- `T.unroll`
- `T.vectorized`
- `T.warp_reduce_max`
- `T.warp_reduce_min`
- `T.warp_reduce_sum`
- `tilelang.layout.make_swizzled_layout`
- `T.get_thread_binding`: dimension 0 only
- `T.alloc_local`: positional shape; dtype positional or keyword
- `T.serial`: positive literal ranges of at most 4096 iterations
- `T.vectorized`: widths 2, 4, or 8
- `T.unroll`: extent and factor at most 16
- `T.sync_threads`: no arguments
- layout dictionaries: automatic swizzle of the same buffer only

Use FP32 intermediates wherever the task contract requires them.
The evaluator imports the source and calls `ModelNew.forward` with CUDA tensors.
