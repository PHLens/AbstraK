from __future__ import annotations

from abstrak.canary.fallback import validate_candidate_source

TRUSTED_TRITON = """
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def copy_kernel(x_ptr, output_ptr, size: tl.constexpr):
    offsets = tl.arange(0, size)
    values = tl.load(x_ptr + offsets)
    tl.store(output_ptr + offsets, values)

class ModelNew(nn.Module):
    def forward(self, x):
        output = torch.empty_like(x)
        copy_kernel[(1,)](x, output, x.numel())
        return output
"""


def test_accepts_trusted_triton_candidate_and_torch_scaffold() -> None:
    result = validate_candidate_source(TRUSTED_TRITON, "triton")

    assert result.valid
    assert result.errors == ()


def test_rejects_torch_sum_fallback_even_with_valid_backend_signature() -> None:
    source = TRUSTED_TRITON.replace("return output", "return torch.sum(output, dim=-1)")

    result = validate_candidate_source(source, "triton")

    assert not result.valid
    assert "framework_compute_fallback" in result.error_codes
    assert any("torch.sum" in issue.message for issue in result.errors)


def test_rejects_candidate_for_a_different_backend() -> None:
    result = validate_candidate_source(TRUSTED_TRITON, "tilelang")

    assert not result.valid
    assert "missing_backend_signature" in result.error_codes


def test_rejects_syntax_error_before_source_inspection() -> None:
    result = validate_candidate_source("def broken(:\n    pass\n", "triton")

    assert not result.valid
    assert result.error_codes == ("syntax_error",)
    assert result.errors[0].line == 1
