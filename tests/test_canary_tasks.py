from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from abstrak.canary.tasks import (
    PinnedAsset,
    TaskRegistryError,
    get_task_pack,
    list_task_ids,
    load_oracle_source,
    load_pinned_asset,
    load_task_source,
    validate_task_registry,
)


def test_row_reduction_task_pack_freezes_cases_and_semantics() -> None:
    task = get_task_pack("row-reduction-scale")

    assert list_task_ids() == ("row-reduction-scale",)
    assert task.dtype == "fp16"
    assert "sum each row" in task.specification
    assert task.reference_precision == "fp32"
    assert task.input_shapes == ((1024, 1024),)
    assert task.parameter_map == {
        "rows": 1024,
        "columns": 1024,
        "scale": 0.5,
        "output_dtype": "fp16",
    }
    assert task.atol == task.rtol == 1e-2
    assert task.fallback_policy == "forbid_framework_ops"
    assert len(task.dev_cases) == 2
    assert [case.kind for case in task.dev_cases] == ["random", "random"]
    assert len(task.sealed_cases) == 5
    assert [case.kind for case in task.sealed_cases] == [
        "random",
        "random",
        "random",
        "random",
        "constant",
    ]
    assert len({case.seed for case in (*task.dev_cases, *task.sealed_cases)}) == 7
    assert task.sealed_cases[-1].value == 0.25


def test_registered_task_and_oracle_are_hash_verified() -> None:
    validate_task_registry()

    task_source = load_task_source("row-reduction-scale")
    oracle_source = load_oracle_source("row-reduction-scale", "triton")

    assert "def make_inputs(" in task_source
    assert "dtype=torch.float32" in task_source
    assert "class ModelNew" in oracle_source
    assert "values.to(tl.float32)" in oracle_source


def test_unknown_task_and_oracle_are_rejected() -> None:
    with pytest.raises(TaskRegistryError, match="unknown task pack"):
        get_task_pack("missing")
    with pytest.raises(TaskRegistryError, match="no tilelang oracle"):
        load_oracle_source("row-reduction-scale", "tilelang")


def test_pinned_asset_rejects_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside", encoding="utf-8")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()

    with pytest.raises(TaskRegistryError, match="unsafe task asset path"):
        load_pinned_asset(PinnedAsset("../outside.py", digest), asset_root=root)


def test_pinned_asset_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside", encoding="utf-8")
    (root / "linked.py").symlink_to(outside)
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()

    with pytest.raises(TaskRegistryError, match="escaped or is missing"):
        load_pinned_asset(PinnedAsset("linked.py", digest), asset_root=root)


def test_pinned_asset_rejects_content_tampering(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    (root / "task.py").write_text("changed", encoding="utf-8")

    with pytest.raises(TaskRegistryError, match="SHA-256 mismatch"):
        load_pinned_asset(PinnedAsset("task.py", "0" * 64), asset_root=root)


def test_importing_task_registry_does_not_import_torch() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import abstrak.canary.tasks; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
