from __future__ import annotations

import ast
import hashlib
import subprocess
import sys

import pytest

from abstrak.canary.baselines import (
    BASELINE_VARIANTS,
    FORMAL_TASK_IDS,
    BaselineRegistryError,
    get_baseline_source,
    list_baseline_task_ids,
    list_baseline_variants,
    load_baseline_source,
    validate_baseline_registry,
    validate_baseline_source,
)


def test_registry_has_three_hash_bound_sources_for_each_formal_task() -> None:
    assert list_baseline_task_ids() == tuple(sorted(FORMAL_TASK_IDS))
    records = []
    for task_id in FORMAL_TASK_IDS:
        assert list_baseline_variants(task_id) == BASELINE_VARIANTS
        for variant in BASELINE_VARIANTS:
            record = get_baseline_source(task_id, variant)
            source = load_baseline_source(task_id, variant)
            assert record.task_id == task_id
            assert record.variant == variant
            assert source == record.source
            assert hashlib.sha256(source.encode("utf-8")).hexdigest() == record.source_sha256
            assert any(
                isinstance(node, ast.ClassDef) and node.name == "ModelNew"
                for node in ast.parse(source).body
            )
            assert validate_baseline_source(
                task_id,
                source,
                source_sha256=record.source_sha256,
            ) == record
            records.append(record)

    assert len(records) == 12
    assert len({record.source_sha256 for record in records}) == 12
    validate_baseline_registry()


def test_variants_use_the_frozen_execution_paths() -> None:
    for task_id in FORMAL_TASK_IDS:
        eager = load_baseline_source(task_id, "eager")
        compiled = load_baseline_source(task_id, "compile")
        vendor = load_baseline_source(task_id, "vendor")
        assert "max-autotune-no-cudagraphs" not in eager
        assert '@torch.compile(mode="max-autotune-no-cudagraphs")' in compiled
        if task_id == "rmsnorm-static":
            assert "F.rms_norm" in vendor
        elif task_id == "layernorm-static":
            assert "F.layer_norm" in vendor
        else:
            assert "torch.matmul(a, b)" in vendor
            assert "a.to(torch.float32)" in eager
        if task_id == "gemm-bias-relu-static":
            assert "product.to(torch.float32)" in vendor


def test_registry_rejects_unknown_or_modified_sources() -> None:
    source = load_baseline_source("rmsnorm-static", "eager")
    with pytest.raises(BaselineRegistryError, match="unregistered baseline source"):
        validate_baseline_source("rmsnorm-static", source + "\n")
    with pytest.raises(BaselineRegistryError, match="declared SHA-256"):
        validate_baseline_source("rmsnorm-static", source, source_sha256="0" * 64)
    with pytest.raises(BaselineRegistryError, match="no baselines registered"):
        list_baseline_variants("row-reduction-scale")


def test_importing_registry_does_not_import_torch() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import abstrak.canary.baselines; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
