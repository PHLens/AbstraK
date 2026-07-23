from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

import abstrak.canary.tasks as task_registry
from abstrak.canary.tasks import (
    CAPABILITY_GATE_SCOPE,
    R1_SCOPE,
    PinnedAsset,
    TaskRegistryError,
    get_task_assets,
    get_task_pack,
    list_task_ids,
    load_oracle_source,
    load_pinned_asset,
    load_task_source,
    validate_task_registry,
)


def test_row_reduction_task_pack_freezes_cases_and_semantics() -> None:
    task = get_task_pack("row-reduction-scale")

    assert list_task_ids() == (
        "gemm-bias-relu-static",
        "gemm-static",
        "layernorm-static",
        "matmul-bias",
        "rmsnorm-static",
        "row-reduction-scale",
    )
    assert list_task_ids(scope=R1_SCOPE) == list_task_ids()
    assert list_task_ids(scope=CAPABILITY_GATE_SCOPE) == ()
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

    for task_id in list_task_ids():
        for backend in ("triton", "tilelang", "cute"):
            assert "class ModelNew" in load_oracle_source(task_id, backend)


def test_matmul_bias_task_pack_freezes_cases_and_semantics() -> None:
    task = get_task_pack("matmul-bias")

    assert task.input_shapes == ((256, 256), (256, 256), (256,))
    assert task.parameter_map["epilogue"] == "bias"
    assert task.reference_precision == "fp32"
    assert len(task.dev_cases) == 2
    assert len(task.sealed_cases) == 5
    assert task.sealed_cases[-1].kind == "constant"
    assert task.sealed_cases[-1].value == 0.125


def test_scientific_task_packs_match_the_frozen_matrix() -> None:
    expected_shapes = {
        "rmsnorm-static": ((4096, 4096), (4096,)),
        "layernorm-static": ((4096, 4096), (4096,), (4096,)),
        "gemm-static": ((1024, 4096), (4096, 4096)),
        "gemm-bias-relu-static": ((1024, 4096), (4096, 4096), (4096,)),
    }

    for task_id, shapes in expected_shapes.items():
        task = get_task_pack(task_id)
        assert task.input_shapes == shapes
        assert task.dtype == "fp16"
        assert task.reference_precision == "fp32"
        assert task.atol == task.rtol == 1e-2
        assert len(task.dev_cases) == 2
        assert len(task.sealed_cases) == 5
        assert task.sealed_cases[-1].kind == "constant"
        assert task.sealed_cases[-1].value == 0.25
        for backend in ("triton", "tilelang", "cute"):
            assert "class ModelNew" in load_oracle_source(task_id, backend)


def test_unknown_task_and_oracle_are_rejected() -> None:
    with pytest.raises(TaskRegistryError, match="unknown task pack"):
        get_task_pack("missing")
    with pytest.raises(TaskRegistryError, match="no cuda oracle"):
        load_oracle_source("row-reduction-scale", "cuda")


def test_unknown_task_registry_scope_is_rejected() -> None:
    with pytest.raises(TaskRegistryError, match="unknown task registry scope: missing"):
        list_task_ids(scope="missing")
    with pytest.raises(TaskRegistryError, match="unknown task registry scope: missing"):
        validate_task_registry(scope="missing")


def test_registry_validation_is_isolated_by_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_paths: list[str] = []
    loaded_roots: list[Path] = []

    def record_load(asset: PinnedAsset, *, asset_root: object = None) -> str:
        loaded_paths.append(asset.relative_path)
        assert isinstance(asset_root, Path)
        loaded_roots.append(asset_root)
        return "source"

    monkeypatch.setattr(task_registry, "load_pinned_asset", record_load)

    validate_task_registry(scope=CAPABILITY_GATE_SCOPE)
    assert loaded_paths == []

    validate_task_registry()
    assert len(loaded_paths) == 24
    assert {path.split("/", 1)[0] for path in loaded_paths} == {"tasks", "oracles"}
    assert {root.name for root in loaded_roots} == {"r1-a100"}


def test_global_lookup_indexes_tasks_from_every_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = get_task_pack("row-reduction-scale").model_copy(update={"id": "capability-only"})
    assets = get_task_assets("row-reduction-scale")
    packs, indexed_assets = task_registry._build_global_indexes(
        {
            R1_SCOPE: task_registry._TaskRegistryScope(packs={}, assets={}),
            CAPABILITY_GATE_SCOPE: task_registry._TaskRegistryScope(
                packs={"capability-only": task},
                assets={"capability-only": assets},
            ),
        }
    )
    monkeypatch.setattr(task_registry, "_TASK_PACKS", packs)
    monkeypatch.setattr(task_registry, "_TASK_ASSETS", indexed_assets)

    assert get_task_pack("capability-only").id == "capability-only"
    assert get_task_assets("capability-only") == assets


def test_global_task_ids_must_be_unique_across_scopes() -> None:
    task = get_task_pack("row-reduction-scale")
    assets = get_task_assets("row-reduction-scale")
    duplicate = task_registry._TaskRegistryScope(
        packs={task.id: task},
        assets={task.id: assets},
    )

    with pytest.raises(TaskRegistryError, match="registered in multiple scopes"):
        task_registry._build_global_indexes(
            {
                R1_SCOPE: duplicate,
                CAPABILITY_GATE_SCOPE: duplicate,
            }
        )


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
