from __future__ import annotations

import hashlib
from textwrap import dedent, indent

import pytest
from pydantic import ValidationError

from abstrak.canary.capabilities import (
    BASE_CAPABILITY_BIT,
    CORE_PACK,
    FULL_PACK,
    MAP_PACK,
    MAPPING_CAPABILITY_BIT,
    SCHED_PACK,
    SCHEDULE_CAPABILITY_BIT,
    CapabilityPackSpec,
    get_capability_pack,
    list_capability_pack_ids,
    minimum_pack_for_bitmask,
    render_tilelang_target_card,
    validate_tilelang_capability_source,
)

CORE_SOURCE = (
    dedent(
        """
    import torch
    import tilelang
    import tilelang.language as T
    from torch import nn

    LENGTH = 256
    BLOCK = 128

    def _build():
        @T.prim_func
        def kernel(
            x: T.Tensor((LENGTH,), T.float16),
            output: T.Tensor((LENGTH,), T.float16),
        ):
            with T.Kernel(T.ceildiv(LENGTH, BLOCK), threads=128) as block:
                values = T.alloc_fragment((BLOCK,), T.float32)
                for offset in T.Parallel(BLOCK):
                    index = block * BLOCK + offset
                    values[offset] = T.cast(x[index], T.float32)
                T.copy(values, output[block * BLOCK])
        return tilelang.compile(kernel, out_idx=1, target="cuda")

    class ModelNew(nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel = _build()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.kernel(x)
    """
    ).strip()
    + "\n"
)


def _inside_kernel(source: str, statement: str) -> str:
    marker = "            values = T.alloc_fragment"
    inserted = indent(statement, "            ")
    return source.replace(marker, f"{inserted}\n{marker}", 1)


def _mapping_source(source: str = CORE_SOURCE) -> str:
    return _inside_kernel(
        source,
        dedent(
            """
            thread = T.get_thread_binding(0)
            local = T.alloc_local((1,), T.float32)
            for lane in T.vectorized(4):
                local[0] = thread + lane
            T.sync_threads()
            """
        ).strip(),
    )


def test_pack_registry_is_frozen_hashable_and_monotonic() -> None:
    assert list_capability_pack_ids() == (
        "tileops-core",
        "tileops-sched",
        "tileops-map",
        "tileops-full",
    )
    assert (CORE_PACK.bitmask, SCHED_PACK.bitmask, MAP_PACK.bitmask, FULL_PACK.bitmask) == (
        1,
        3,
        5,
        7,
    )
    assert CORE_PACK.bitmask & SCHED_PACK.bitmask == CORE_PACK.bitmask
    assert CORE_PACK.bitmask & MAP_PACK.bitmask == CORE_PACK.bitmask
    assert SCHED_PACK.bitmask | MAP_PACK.bitmask == FULL_PACK.bitmask
    assert len({pack.sha256 for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)}) == 4
    assert all(len(pack.sha256) == 64 for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK))
    assert hash(CORE_PACK) == hash(get_capability_pack(CORE_PACK.id))

    with pytest.raises(ValidationError, match="frozen"):
        CORE_PACK.bitmask = 7
    payload = CORE_PACK.model_dump()
    payload["kernel_threads"] = (64, 128, 256)
    with pytest.raises(ValidationError, match="thread domain"):
        CapabilityPackSpec.model_validate(payload)


def test_minimum_pack_inference_uses_independent_schedule_and_mapping_bits() -> None:
    assert minimum_pack_for_bitmask(0).id == CORE_PACK.id
    assert minimum_pack_for_bitmask(BASE_CAPABILITY_BIT).id == CORE_PACK.id
    assert minimum_pack_for_bitmask(SCHEDULE_CAPABILITY_BIT).id == SCHED_PACK.id
    assert minimum_pack_for_bitmask(MAPPING_CAPABILITY_BIT).id == MAP_PACK.id
    assert minimum_pack_for_bitmask(SCHEDULE_CAPABILITY_BIT | MAPPING_CAPABILITY_BIT).id == (
        FULL_PACK.id
    )
    with pytest.raises(ValueError, match="invalid capability bitmask"):
        minimum_pack_for_bitmask(8)


def test_machine_rendered_cards_are_stable_and_hide_disabled_surfaces() -> None:
    cards = {
        pack.id: render_tilelang_target_card(pack)
        for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)
    }
    assert all(card.endswith("\n") for card in cards.values())
    assert all(
        render_tilelang_target_card(pack.id) == cards[pack.id]
        for pack in (
            CORE_PACK,
            SCHED_PACK,
            MAP_PACK,
            FULL_PACK,
        )
    )
    for required in (
        "T.prim_func",
        "T.float16",
        "T.float32",
        "T.int32",
        "T.int64",
        "T.infinity",
        "T.pow",
        "T.if_then_else",
    ):
        assert required in cards[CORE_PACK.id]
    assert "T.alloc_local" not in cards[CORE_PACK.id]
    assert "GemmWarpPolicy" not in cards[CORE_PACK.id]
    assert "threads=64" not in cards[CORE_PACK.id]
    assert "T.alloc_local" not in cards[SCHED_PACK.id]
    assert "T.GemmWarpPolicy.FullRow" in cards[SCHED_PACK.id]
    assert "T.alloc_local" in cards[MAP_PACK.id]
    assert "GemmWarpPolicy" not in cards[MAP_PACK.id]
    assert "64, 128, 256" in cards[FULL_PACK.id]
    assert "T.alloc_local" in cards[FULL_PACK.id]
    assert {
        pack_id: hashlib.sha256(card.encode()).hexdigest() for pack_id, card in cards.items()
    } == {
        "tileops-core": "57d79d14cf585232508dcf72aee280b1151c5913159a40ba094a33fb6c509154",
        "tileops-sched": "58f2e39c69fc1cae3a1bb0dc1224111e6268be01366358716df2da8083416ba6",
        "tileops-map": "07d7fbe1235c416e256b1a19819b23dd535d6bdb61aa678dd081a58c9b2bc386",
        "tileops-full": "a9ed048414cd3062ee95e58dc75a6ba7273d2e82eda60d047664c4d5e09b123a",
    }


def test_core_source_is_accepted_monotonically_by_every_pack() -> None:
    results = {
        pack.id: validate_tilelang_capability_source(CORE_SOURCE, pack)
        for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)
    }

    assert all(result.valid for result in results.values())
    assert all(result.minimum_pack_id == CORE_PACK.id for result in results.values())
    assert all(result.minimum_pack_bitmask == CORE_PACK.bitmask for result in results.values())
    assert "T.Kernel" in results[CORE_PACK.id].used_capabilities
    assert "tilelang.compile" in results[CORE_PACK.id].used_capabilities


def test_schedule_source_requires_sched_or_full_pack() -> None:
    source = CORE_SOURCE.replace("threads=128", "threads=64", 1)
    results = {
        pack.id: validate_tilelang_capability_source(source, pack)
        for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)
    }

    assert not results[CORE_PACK.id].valid
    assert results[SCHED_PACK.id].valid
    assert not results[MAP_PACK.id].valid
    assert results[FULL_PACK.id].valid
    assert results[SCHED_PACK.id].minimum_pack_id == SCHED_PACK.id
    assert "schedule.kernel_threads" in results[SCHED_PACK.id].used_capabilities
    assert results[CORE_PACK.id].error_codes[-1] == "capability_pack_violation"


def test_pipeline_stage_and_gemm_policy_are_schedule_capabilities() -> None:
    source = CORE_SOURCE.replace(
        "for offset in T.Parallel(BLOCK):",
        "for offset in T.Pipelined(BLOCK, num_stages=2):",
        1,
    ).replace(
        "            for offset in T.Pipelined",
        "            T.gemm(values, values, values, policy=T.GemmWarpPolicy.FullRow)\n"
        "            for offset in T.Pipelined",
        1,
    )

    result = validate_tilelang_capability_source(source, SCHED_PACK)

    assert result.valid
    assert result.minimum_pack_id == SCHED_PACK.id
    assert "schedule.pipeline_stages" in result.used_capabilities
    assert "schedule.gemm_policy" in result.used_capabilities


def test_mapping_source_requires_map_or_full_pack() -> None:
    source = _mapping_source()
    results = {
        pack.id: validate_tilelang_capability_source(source, pack)
        for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)
    }

    assert not results[CORE_PACK.id].valid
    assert not results[SCHED_PACK.id].valid
    assert results[MAP_PACK.id].valid
    assert results[FULL_PACK.id].valid
    assert results[MAP_PACK.id].minimum_pack_id == MAP_PACK.id
    assert "T.get_thread_binding" in results[MAP_PACK.id].used_capabilities
    assert "T.vectorized" in results[MAP_PACK.id].used_capabilities


def test_registered_mapping_surface_and_automatic_layout_are_recorded() -> None:
    source = _inside_kernel(
        CORE_SOURCE,
        dedent(
            """
            shared = T.alloc_shared((BLOCK,), T.float16)
            T.annotate_layout({shared: tilelang.layout.make_swizzled_layout(shared)})
            thread = T.get_thread_binding()
            local = T.alloc_local((1,), T.float32)
            for outer in T.serial(4):
                for inner in T.unroll(4, unroll_factor=4):
                    local[0] = thread + outer + inner
            local[0] = T.warp_reduce_sum(local[0])
            T.sync_threads()
            """
        ).strip(),
    )

    result = validate_tilelang_capability_source(source, MAP_PACK)

    assert result.valid
    assert result.minimum_pack_id == MAP_PACK.id
    assert {
        "T.alloc_local",
        "T.annotate_layout",
        "T.get_thread_binding",
        "T.serial",
        "T.sync_threads",
        "T.unroll",
        "T.warp_reduce_sum",
        "tilelang.layout.make_swizzled_layout",
    } <= set(result.used_capabilities)


def test_combined_schedule_and_mapping_source_requires_full_pack() -> None:
    source = _mapping_source(CORE_SOURCE.replace("threads=128", "threads=256", 1))

    results = [
        validate_tilelang_capability_source(source, pack)
        for pack in (CORE_PACK, SCHED_PACK, MAP_PACK, FULL_PACK)
    ]

    assert [result.valid for result in results] == [False, False, False, True]
    assert all(result.minimum_pack_id == FULL_PACK.id for result in results)


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (CORE_SOURCE.replace("threads=128", "threads=512", 1), "capability_argument_domain"),
        (
            CORE_SOURCE.replace(
                "for offset in T.Parallel(BLOCK):",
                "for offset in T.Pipelined(BLOCK, num_stages=4):",
                1,
            ),
            "capability_argument_domain",
        ),
        (
            CORE_SOURCE.replace(
                "for offset in T.Parallel(BLOCK):",
                "for offset in T.Pipelined(BLOCK, num_stages=2, order=[0]):",
                1,
            ),
            "capability_argument_forbidden",
        ),
        (
            CORE_SOURCE.replace(
                "for offset in T.Parallel(BLOCK):",
                "for offset in T.Parallel(BLOCK, coalesced_width=4):",
                1,
            ),
            "capability_argument_forbidden",
        ),
        (
            _inside_kernel(CORE_SOURCE, 'T.gemm(values, values, values, policy="FullRow")'),
            "capability_argument_domain",
        ),
        (
            _inside_kernel(CORE_SOURCE, "thread = T.get_thread_binding(1)"),
            "capability_argument_domain",
        ),
        (
            _inside_kernel(CORE_SOURCE, "for lane in T.vectorized(16):\n    pass"),
            "capability_argument_domain",
        ),
        (
            _inside_kernel(CORE_SOURCE, "for lane in T.unroll(32):\n    pass"),
            "capability_argument_domain",
        ),
        (_inside_kernel(CORE_SOURCE, "T.sync_threads(1)"), "capability_argument_domain"),
    ],
)
def test_argument_domains_fail_closed(source: str, code: str) -> None:
    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert code in result.error_codes


def test_literal_constant_folding_is_allowed_for_schedule_parameters() -> None:
    source = CORE_SOURCE.replace("BLOCK = 128", "BLOCK = 128\nTHREADS = 32 * 2", 1).replace(
        "threads=128", "threads=THREADS", 1
    )

    result = validate_tilelang_capability_source(source, SCHED_PACK)

    assert result.valid
    assert result.minimum_pack_id == SCHED_PACK.id


@pytest.mark.parametrize(
    "extra_binding",
    [
        "THREADS += 64",
        "for THREADS in (64,):\n    pass",
        "if (THREADS := 64):\n    pass",
        "def shadow():\n    THREADS = 64",
    ],
)
def test_module_constants_require_one_simple_binding_in_the_whole_module(
    extra_binding: str,
) -> None:
    source = CORE_SOURCE.replace(
        "BLOCK = 128",
        f"BLOCK = 128\nTHREADS = 64\n{extra_binding}",
        1,
    ).replace("threads=128", "threads=THREADS", 1)

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_argument_dynamic" in result.error_codes


def test_constant_folding_rejects_large_shifts_before_evaluation() -> None:
    source = CORE_SOURCE.replace("BLOCK = 128", "BLOCK = 128\nTHREADS = 1 << 1000000", 1).replace(
        "threads=128", "threads=THREADS", 1
    )

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_argument_dynamic" in result.error_codes


@pytest.mark.parametrize(
    "definition",
    [
        "def helper(op=T.copy):\n    return op",
        "def helper(*, op=T.copy):\n    return op",
        "def helper(op=next(iter(T.copy))):\n    return op",
    ],
)
def test_function_defaults_cannot_capture_or_derive_tilelang_symbols(definition: str) -> None:
    source = CORE_SOURCE.replace("class ModelNew", definition + "\n\nclass ModelNew", 1)

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_symbol_alias" in result.error_codes


def test_allocation_dtype_and_thread_dimension_keyword_forms_are_supported() -> None:
    core_source = CORE_SOURCE.replace(
        "T.alloc_fragment((BLOCK,), T.float32)",
        "T.alloc_fragment((BLOCK,), dtype=T.float32)",
        1,
    )
    core_source = _inside_kernel(
        core_source,
        "shared = T.alloc_shared((BLOCK,), dtype=T.float16)",
    )
    mapping_source = (
        _mapping_source()
        .replace("T.get_thread_binding(0)", "T.get_thread_binding(dim=0)", 1)
        .replace("T.alloc_local((1,), T.float32)", "T.alloc_local((1,), dtype=T.float32)", 1)
    )

    core_result = validate_tilelang_capability_source(core_source, CORE_PACK)
    mapping_result = validate_tilelang_capability_source(mapping_source, MAP_PACK)

    assert core_result.valid
    assert mapping_result.valid


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            CORE_SOURCE.replace("import tilelang.language as T", "import tilelang.language as TL"),
            "capability_import_alias",
        ),
        ("COPY = T.copy\n" + CORE_SOURCE, "capability_symbol_alias"),
        ("COPY = T.copy if True else None\n" + CORE_SOURCE, "capability_symbol_alias"),
        (_inside_kernel(CORE_SOURCE, 'copy = getattr(T, "copy")'), "capability_dynamic_access"),
        (
            _inside_kernel(CORE_SOURCE, "(T.copy if True else T.clear)(values, values)"),
            "capability_dynamic_access",
        ),
        ("CUDA_SOURCE = '__global__ void kernel() {}'\n" + CORE_SOURCE, "capability_raw_cuda"),
        ("import tvm\n" + CORE_SOURCE, "capability_tvm_escape"),
        (_inside_kernel(CORE_SOURCE, "T.assume(True)"), "capability_unknown_symbol"),
        (
            CORE_SOURCE.replace("return self.kernel(x)", "return x + x", 1),
            "framework_compute_fallback",
        ),
    ],
)
def test_alias_dynamic_and_escape_bypasses_are_rejected(source: str, code: str) -> None:
    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert code in result.error_codes


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            _inside_kernel(CORE_SOURCE, '__builtins__["eval"]("1 + 1")'),
            "capability_indirect_call",
        ),
        (
            _inside_kernel(CORE_SOURCE, "(T.copy,)[0](values, values)"),
            "capability_indirect_call",
        ),
        (
            _inside_kernel(CORE_SOURCE, "factory()(values)"),
            "capability_indirect_call",
        ),
        (
            CORE_SOURCE.replace("return self.kernel(x)", "return x.square()", 1),
            "framework_compute_fallback",
        ),
        (
            CORE_SOURCE.replace("return self.kernel(x)", "return x.__add__(x)", 1),
            "framework_compute_fallback",
        ),
        (
            CORE_SOURCE.replace("return self.kernel(x)", "return x", 1),
            "capability_forward_contract",
        ),
    ],
)
def test_non_dsl_calls_and_forward_bypasses_are_default_denied(source: str, code: str) -> None:
    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert code in result.error_codes


def test_python_comparisons_are_forbidden_outside_prim_func() -> None:
    helper = "def compare_helper(x):\n    return x > 0\n\n"
    source = CORE_SOURCE.replace("class ModelNew", helper + "class ModelNew", 1)

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "framework_compute_fallback" in result.error_codes


def test_raw_cuda_words_in_comments_do_not_trigger_source_rejection() -> None:
    source = "# PTX __global__ load_inline RawKernel\n" + CORE_SOURCE

    result = validate_tilelang_capability_source(source, CORE_PACK)

    assert result.valid


def test_actual_raw_cuda_callee_is_rejected_without_a_source_string() -> None:
    source = _inside_kernel(
        CORE_SOURCE,
        'T.CUDASourceCodeKernel(1, source_code_or_path="safe-name")',
    )

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_raw_cuda" in result.error_codes


def test_existing_tilelang_fallback_runs_before_capability_inspection() -> None:
    source = CORE_SOURCE.replace("return self.kernel(x)", "return torch.sum(x)", 1)

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "framework_compute_fallback" in result.error_codes
    assert all(not code.startswith("capability_") for code in result.error_codes)
    assert result.used_capabilities == ()
    assert result.minimum_pack_id is None


def test_framework_arithmetic_cannot_be_hidden_in_a_python_helper() -> None:
    helper = dedent(
        """
        def framework_helper(x):
            return x * x

        """
    )
    source = CORE_SOURCE.replace("class ModelNew", helper + "class ModelNew", 1).replace(
        "return self.kernel(x)", "return framework_helper(x)", 1
    )

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "framework_compute_fallback" in result.error_codes


@pytest.mark.parametrize(
    "framework_kernel",
    (
        "torch.ops.aten.clone.default",
        "nn.functional.gelu",
        "torch.add",
    ),
)
def test_self_kernel_must_bind_to_a_verified_tilelang_builder(
    framework_kernel: str,
) -> None:
    source = CORE_SOURCE.replace(
        "self.kernel = _build()",
        f"self.kernel = {framework_kernel}",
        1,
    )

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_kernel_binding" in result.error_codes


def test_builder_must_directly_return_the_compiled_tilelang_kernel() -> None:
    source = CORE_SOURCE.replace(
        'return tilelang.compile(kernel, out_idx=1, target="cuda")',
        'tilelang.compile(kernel, out_idx=1, target="cuda")\n'
        "    return torch.ops.aten.clone.default",
        1,
    )

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_kernel_binding" in result.error_codes


@pytest.mark.parametrize(
    "model_escape",
    (
        "\n    def __getattribute__(self, name):\n        return torch.add\n",
        "\n    kernel = torch.add\n",
    ),
)
def test_modelnew_cannot_override_kernel_dispatch(model_escape: str) -> None:
    source = CORE_SOURCE.replace(
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
        model_escape + "\n    def forward(self, x: torch.Tensor) -> torch.Tensor:",
        1,
    )

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_model_contract" in result.error_codes


def test_target_card_import_spellings_are_enforced_exactly() -> None:
    source = CORE_SOURCE.replace("from torch import nn", "import torch.nn as nn", 1)

    result = validate_tilelang_capability_source(source, CORE_PACK)

    assert not result.valid
    assert "capability_import_forbidden" in result.error_codes
    assert "capability_missing_import" in result.error_codes


@pytest.mark.parametrize(
    "source",
    (
        CORE_SOURCE.replace(
            "tilelang.compile(kernel,",
            "tilelang.compile(torch.ops.aten.clone.default,",
            1,
        ),
        CORE_SOURCE.replace(
            "    return tilelang.compile",
            "    kernel = torch.add\n    return tilelang.compile",
            1,
        ),
        CORE_SOURCE.replace(
            "    @T.prim_func",
            "    @staticmethod\n    @T.prim_func",
            1,
        ),
    ),
)
def test_compile_input_is_the_unique_canonical_prim_func(source: str) -> None:
    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert {
        "capability_compile_kernel",
        "capability_decorator_forbidden",
    } & set(result.error_codes)


@pytest.mark.parametrize(
    ("captured_name", "expected_code"),
    (
        ("_build", "capability_kernel_binding"),
        ("ModelNew", "capability_symbol_rebinding"),
    ),
)
def test_match_patterns_cannot_rebind_verified_entrypoints(
    captured_name: str,
    expected_code: str,
) -> None:
    source = CORE_SOURCE + f"\nmatch torch.add:\n    case {captured_name}:\n        pass\n"

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert expected_code in result.error_codes


def test_modelnew_must_be_exported_at_module_scope() -> None:
    marker = "class ModelNew(nn.Module):"
    prefix, model_body = CORE_SOURCE.split(marker, 1)
    nested_model = indent(marker + model_body, "    ")
    source = prefix + "class Wrapper:\n" + nested_model

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_model_contract" in result.error_codes


def test_modelnew_cannot_be_conditionally_defined_at_module_scope() -> None:
    marker = "class ModelNew(nn.Module):"
    prefix, model_body = CORE_SOURCE.split(marker, 1)
    conditional_model = indent(marker + model_body, "    ")
    source = prefix + "if False:\n" + conditional_model

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_model_contract" in result.error_codes


@pytest.mark.parametrize(
    "replacement",
    (
        "def ModelNew():\n    return None\n",
        "try:\n    1 / 0\nexcept Exception as ModelNew:\n    pass\n",
    ),
)
def test_modelnew_export_cannot_be_rebound_by_non_name_ast_fields(
    replacement: str,
) -> None:
    source = CORE_SOURCE + "\n" + replacement

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert {
        "capability_model_contract",
        "capability_symbol_rebinding",
    } & set(result.error_codes)


def test_compile_cannot_resolve_a_prim_func_from_class_scope() -> None:
    compile_call = 'tilelang.compile(kernel, out_idx=1, target="cuda")'
    source = CORE_SOURCE.replace(
        "def _build():\n    @T.prim_func",
        "class KernelHolder:\n    @T.prim_func",
        1,
    ).replace(
        f"    return {compile_call}",
        f"\ndef _build():\n    return {compile_call}",
        1,
    )

    result = validate_tilelang_capability_source(source, FULL_PACK)

    assert not result.valid
    assert "capability_compile_kernel" in result.error_codes


def test_compile_can_resolve_a_unique_module_scope_prim_func() -> None:
    compile_call = 'tilelang.compile(kernel, out_idx=1, target="cuda")'
    prefix, remainder = CORE_SOURCE.split("def _build():\n", 1)
    builder_body, model_body = remainder.split("\nclass ModelNew", 1)
    kernel_body, _ = builder_body.rsplit(f"    return {compile_call}", 1)
    source = (
        prefix
        + dedent(kernel_body)
        + f"def _build():\n    return {compile_call}\n\nclass ModelNew"
        + model_body
    )

    result = validate_tilelang_capability_source(source, CORE_PACK)

    assert result.valid
