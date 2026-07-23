# TileLang 0.1.12 / NVIDIA A100 / tileops-core

Target ID: `tilelang-a100-core`
Capability contract SHA-256: `c59a1076df133ecb2ea4e1bf4f18c5d613d81a05155f2b0614a280236f02a116`

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

- `T.Kernel(..., threads=...)`: {128}
- `T.Pipelined(..., num_stages=...)`: {0}
- `T.Parallel(...)`: positional extents only
- shared/fragment allocation: positional shape; dtype positional or keyword
- `T.gemm` transpose flags: literal booleans only
- high-level reductions: literal `dim` and `clear` controls only
- `T.gemm(...)`: default policy only

Use FP32 intermediates wherever the task contract requires them.
The evaluator imports the source and calls `ModelNew.forward` with CUDA tensors.
