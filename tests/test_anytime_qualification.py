from __future__ import annotations

from collections.abc import Callable

import pytest

from abstrak.anytime.isolation import (
    AnytimeCandidateInvocation,
    AnytimeOutputChannel,
    AnytimePublicRuntime,
    AnytimePublicTaskABI,
    AnytimeTensorDescriptor,
    build_anytime_candidate_source,
    build_anytime_process_isolation_contract,
)
from abstrak.anytime.qualification import (
    AnytimeCandidateQualificationBinding,
    AnytimeCuteSyntheticLaunchPayload,
    AnytimeSyntheticLaunchAttestation,
    AnytimeSyntheticLaunchPayload,
    AnytimeSyntheticRuntimeObservation,
    AnytimeTileLangSyntheticLaunchPayload,
    AnytimeTritonSyntheticLaunchPayload,
    attest_anytime_synthetic_launch,
    build_anytime_qualification_binding,
    get_anytime_target_static_policy,
    qualify_anytime_candidate_offline,
    validate_anytime_candidate_source,
)

_TARGET_HASH = "1" * 64
_EXECUTION_HASH = "2" * 64
_KERNEL_HASH = "3" * 64
_LOWERED_HASH = "4" * 64
_CHANNEL = "5" * 32

_TRITON_SOURCE = """\
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, output, n_elements: tl.constexpr):
    offsets = tl.program_id(0) * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    left = tl.load(x + offsets, mask=mask)
    right = tl.load(y + offsets, mask=mask)
    tl.store(output + offsets, left + right, mask=mask)

class ModelNew:
    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = x.numel()
        add_kernel[(triton.cdiv(n_elements, 128),)](x, y, output, n_elements)
        return output
"""

_TILELANG_SOURCE = """\
import tilelang
import tilelang.language as T

@T.prim_func
def add_kernel(x, y, output):
    with T.Kernel(1, threads=128) as block:
        T.copy(x, output)

compiled_kernel = tilelang.compile(add_kernel, target="cuda", out_idx=[2])

class ModelNew:
    def forward(self, x, y):
        return compiled_kernel(x, y)
"""

_CUTE_SOURCE = """\
import cutlass.cute as cute

@cute.kernel
def add_kernel(x, y, output):
    cute.copy(x, output)

compiled_kernel = cute.compile(add_kernel)

class ModelNew:
    def forward(self, x, y):
        return compiled_kernel(x, y)
"""

_SOURCES = {
    "triton": _TRITON_SOURCE,
    "tilelang": _TILELANG_SOURCE,
    "cute": _CUTE_SOURCE,
}


def _tensor(name: str) -> AnytimeTensorDescriptor:
    return AnytimeTensorDescriptor(
        name=name,
        shape=(32, 32),
        strides=(32, 1),
        dtype="float16",
    )


def _invocation(backend: str = "triton", source: str | None = None) -> AnytimeCandidateInvocation:
    return AnytimeCandidateInvocation(
        source=build_anytime_candidate_source(source or _SOURCES[backend]),
        public_abi=AnytimePublicTaskABI(
            abi_id="matrix-add",
            abi_version="v1",
            entrypoint="ModelNew.forward",
            input_names=("x", "y"),
            output_count=1,
        ),
        public_runtime=AnytimePublicRuntime(
            target_id=f"{backend}-a100",
            backend=backend,
            runtime_id=f"{backend}-runtime-v1",
            runtime_abi_version="kernelbench-v1",
        ),
        inputs=(_tensor("x"), _tensor("y")),
        output_channel=AnytimeOutputChannel(
            channel_id=_CHANNEL,
            expected_output_count=1,
        ),
    )


def _binding(invocation: AnytimeCandidateInvocation) -> AnytimeCandidateQualificationBinding:
    return build_anytime_qualification_binding(
        invocation=invocation,
        target_stack_sha256=_TARGET_HASH,
        execution_binding_sha256=_EXECUTION_HASH,
    )


def _runtime(
    binding: AnytimeCandidateQualificationBinding,
    *,
    terminal_status: str = "completed",
    outputs_finite: bool = True,
    inputs_unchanged: bool = True,
    observed_output_count: int = 1,
    ipc_envelope_valid: bool = True,
) -> AnytimeSyntheticRuntimeObservation:
    return AnytimeSyntheticRuntimeObservation(
        candidate_invocation_sha256=binding.candidate_invocation_sha256,
        candidate_source_sha256=binding.candidate_source_sha256,
        execution_binding_sha256=binding.execution_binding_sha256,
        terminal_status=terminal_status,
        expected_output_count=1,
        observed_output_count=observed_output_count,
        outputs_finite=outputs_finite,
        inputs_unchanged=inputs_unchanged,
        ipc_envelope_valid=ipc_envelope_valid,
        elapsed_seconds=0.5,
    )


def _payload(
    binding: AnytimeCandidateQualificationBinding,
) -> AnytimeSyntheticLaunchPayload:
    fields = dict(
        target_id=binding.target_id,
        target_stack_sha256=binding.target_stack_sha256,
        candidate_source_sha256=binding.candidate_source_sha256,
        candidate_invocation_sha256=binding.candidate_invocation_sha256,
        execution_binding_sha256=binding.execution_binding_sha256,
        runtime_launch_count=1,
        launched_kernel_sha256=_KERNEL_HASH,
        lowered_code_sha256=_LOWERED_HASH,
        core_operation_attributed=True,
        fallback_detected=False,
        dummy_signature_only=False,
    )
    constructors: dict[str, Callable[..., AnytimeSyntheticLaunchPayload]] = {
        "triton": AnytimeTritonSyntheticLaunchPayload,
        "tilelang": AnytimeTileLangSyntheticLaunchPayload,
        "cute": AnytimeCuteSyntheticLaunchPayload,
    }
    return constructors[binding.backend](**fields)


def _isolation():
    return build_anytime_process_isolation_contract(
        max_wall_seconds=30.0,
        max_memory_bytes=2**30,
    )


@pytest.mark.parametrize("backend", ("triton", "tilelang", "cute"))
def test_each_target_has_an_independent_passing_default_deny_fixture(backend: str) -> None:
    result = validate_anytime_candidate_source(
        _SOURCES[backend],
        target_id=f"{backend}-a100",
        backend=backend,
    )

    assert result.valid is True
    assert result.backend_signature_present is True
    assert result.target_operation_count >= 1
    assert result.error_codes == ()


def test_target_policies_do_not_form_a_cross_dsl_union() -> None:
    policies = {
        backend: get_anytime_target_static_policy(backend)
        for backend in ("triton", "tilelang", "cute")
    }
    assert len({policy.sha256 for policy in policies.values()}) == 3
    assert "tilelang" not in policies["triton"].allowed_imports
    assert "cutlass.cute" not in policies["tilelang"].allowed_imports
    assert "triton.language" not in policies["cute"].allowed_imports

    wrong_target = validate_anytime_candidate_source(
        _TRITON_SOURCE,
        target_id="tilelang-a100",
        backend="tilelang",
    )
    assert wrong_target.valid is False
    assert "import_not_allowed" in wrong_target.error_codes
    assert "missing_target_decorator" in wrong_target.error_codes


@pytest.mark.parametrize(
    ("injected", "expected_code"),
    (
        ("\nimport os\nleak = open('/etc/passwd').read()\n", "filesystem_access_forbidden"),
        ("\nimport inspect\nframe = inspect.currentframe()\n", "frame_inspection_forbidden"),
        ("\nloader = getattr(triton, 'jit')\n", "dynamic_lookup_forbidden"),
        (
            "\nfrom triton.language import __getattr__ as loader\nvalue = loader('load')\n",
            "dynamic_lookup_forbidden",
        ),
        ("\ndef fallback(x, y):\n    return torch.mm(x, y)\n", "framework_fallback_forbidden"),
        ("\ndef fallback(x, y):\n    return x @ y\n", "framework_fallback_forbidden"),
        ("\ndef forward(x, y):\n    x.add_(y)\n", "input_mutation_forbidden"),
        ("\nwhile True:\n    pass\n", "unbounded_control_flow"),
        ("\nasset = '/home/cambricon/AbstraK/benchmarks/task.py'\n", "filesystem_path_forbidden"),
        ("\nsealed_oracle = 'hidden'\n", "private_asset_reference"),
    ),
)
def test_hostile_source_controls_fail_closed(injected: str, expected_code: str) -> None:
    result = validate_anytime_candidate_source(
        _TRITON_SOURCE + injected,
        target_id="triton-a100",
        backend="triton",
    )

    assert result.valid is False
    assert expected_code in result.error_codes


@pytest.mark.parametrize(
    ("backend", "dummy"),
    (
        (
            "triton",
            "import triton\n@triton.jit\ndef kernel(x):\n    pass\n",
        ),
        (
            "tilelang",
            "import tilelang\nimport tilelang.language as T\n"
            "@T.prim_func\ndef kernel(x):\n    pass\n"
            "compiled = tilelang.compile(kernel)\n",
        ),
        (
            "cute",
            "import cutlass.cute as cute\n@cute.kernel\ndef kernel(x):\n    pass\n"
            "compiled = cute.compile(kernel)\n",
        ),
    ),
)
def test_dummy_dsl_signatures_are_not_target_use(backend: str, dummy: str) -> None:
    result = validate_anytime_candidate_source(
        dummy,
        target_id=f"{backend}-a100",
        backend=backend,
    )

    assert result.valid is False
    assert "dummy_target_signature" in result.error_codes
    assert result.backend_signature_present is False


def test_compiled_target_callable_cannot_be_rebound_to_framework_fallback() -> None:
    hostile = _TILELANG_SOURCE.replace(
        "class ModelNew:",
        "compiled_kernel = torch.mm\n\nclass ModelNew:",
    ).replace("import tilelang\n", "import tilelang\nimport torch\n")
    result = validate_anytime_candidate_source(
        hostile,
        target_id="tilelang-a100",
        backend="tilelang",
    )

    assert result.valid is False
    assert "call_not_allowed" in result.error_codes


@pytest.mark.parametrize("backend", ("triton", "tilelang", "cute"))
def test_complete_scripted_attestation_still_only_reaches_pending_m9(backend: str) -> None:
    invocation = _invocation(backend)
    binding = _binding(invocation)
    runtime = _runtime(binding)
    attestation = attest_anytime_synthetic_launch(_payload(binding))

    decision = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=_isolation(),
        runtime_observation=runtime,
        launch_attestation=attestation,
    )

    assert decision.status == "pending-m9"
    assert decision.rejection_codes == ()
    assert decision.formal_target_use_qualified is False
    assert decision.real_os_containment_observed is False
    assert decision.trusted_gpu_launch_observed is False
    assert decision.next_gate == "m9-trusted-gpu-preflight"


@pytest.mark.parametrize(
    "evidence_changes",
    (
        {"evidence_mode": "runtime", "lowered_code_sha256": None},
        {
            "evidence_mode": "lowered",
            "runtime_launch_count": 0,
            "launched_kernel_sha256": None,
        },
    ),
)
def test_runtime_or_lowered_evidence_contracts_are_supported_but_remain_pending(
    evidence_changes: dict[str, object],
) -> None:
    invocation = _invocation()
    binding = _binding(invocation)
    payload = _payload(binding).model_copy(update=evidence_changes)

    decision = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=_isolation(),
        runtime_observation=_runtime(binding),
        launch_attestation=attest_anytime_synthetic_launch(payload),
    )

    assert decision.status == "pending-m9"
    assert decision.formal_target_use_qualified is False


def test_missing_malformed_and_wrong_target_launch_evidence_fail_closed() -> None:
    invocation = _invocation()
    binding = _binding(invocation)
    runtime = _runtime(binding)

    missing = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=_isolation(),
        runtime_observation=runtime,
        launch_attestation=None,
    )
    assert missing.status == "rejected"
    assert "launch_evidence_missing" in missing.rejection_codes

    attestation = attest_anytime_synthetic_launch(_payload(binding))
    malformed = attestation.model_copy(update={"payload_sha256": "f" * 64})
    malformed_decision = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=_isolation(),
        runtime_observation=runtime,
        launch_attestation=malformed,
    )
    assert "launch_evidence_malformed" in malformed_decision.rejection_codes

    wrong_payload = _payload(binding).model_copy(update={"target_id": "triton-other"})
    wrong = attest_anytime_synthetic_launch(wrong_payload)
    wrong_decision = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=_isolation(),
        runtime_observation=runtime,
        launch_attestation=wrong,
    )
    assert "launch_evidence_binding_mismatch" in wrong_decision.rejection_codes


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    (
        ({"runtime_launch_count": 0, "launched_kernel_sha256": None}, "target_launch_not_observed"),
        ({"lowered_code_sha256": None}, "lowered_code_missing"),
        ({"core_operation_attributed": False}, "core_operation_not_attributed"),
        ({"fallback_detected": True}, "framework_fallback_detected"),
        ({"dummy_signature_only": True}, "dummy_target_signature"),
    ),
)
def test_launch_attestation_controls_reject_fallback_and_dummy_evidence(
    changes: dict[str, object],
    expected_code: str,
) -> None:
    invocation = _invocation()
    binding = _binding(invocation)
    payload = _payload(binding).model_copy(update=changes)
    attestation = attest_anytime_synthetic_launch(payload)

    decision = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=_isolation(),
        runtime_observation=_runtime(binding),
        launch_attestation=attestation,
    )

    assert decision.status == "rejected"
    assert expected_code in decision.rejection_codes


@pytest.mark.parametrize(
    ("runtime_changes", "expected_code"),
    (
        ({"terminal_status": "timeout"}, "candidate_timeout"),
        ({"terminal_status": "oom"}, "candidate_oom"),
        ({"terminal_status": "crash"}, "candidate_crash"),
        ({"outputs_finite": False}, "nonfinite_output"),
        ({"inputs_unchanged": False}, "input_mutation_detected"),
        ({"observed_output_count": 0}, "output_count_mismatch"),
        ({"ipc_envelope_valid": False}, "runtime_ipc_invalid"),
    ),
)
def test_supervisor_controls_reject_hang_oom_mutation_and_nonfinite_output(
    runtime_changes: dict[str, object],
    expected_code: str,
) -> None:
    invocation = _invocation()
    binding = _binding(invocation)
    runtime = _runtime(binding).model_copy(update=runtime_changes)

    decision = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=_isolation(),
        runtime_observation=runtime,
        launch_attestation=attest_anytime_synthetic_launch(_payload(binding)),
    )

    assert decision.status == "rejected"
    assert expected_code in decision.rejection_codes


def test_candidate_timing_forgery_and_binding_tampering_are_revalidated() -> None:
    invocation = _invocation()
    binding = _binding(invocation)
    forged_timing = _runtime(binding).model_copy(
        update={"candidate_reported_timing_accepted": True}
    )
    decision = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=_isolation(),
        runtime_observation=forged_timing,
        launch_attestation=attest_anytime_synthetic_launch(_payload(binding)),
    )
    assert "runtime_observation_malformed" in decision.rejection_codes

    forged_attestation = AnytimeSyntheticLaunchAttestation.model_construct(
        payload=_payload(binding).model_copy(update={"execution_binding_sha256": "9" * 64}),
        payload_sha256="8" * 64,
    )
    decision = qualify_anytime_candidate_offline(
        binding=binding,
        invocation=invocation,
        isolation_contract=_isolation(),
        runtime_observation=_runtime(binding),
        launch_attestation=forged_attestation,
    )
    assert "launch_evidence_malformed" in decision.rejection_codes


def test_synthetic_elapsed_negative_zero_is_canonicalized() -> None:
    binding = _binding(_invocation())
    runtime = AnytimeSyntheticRuntimeObservation(
        candidate_invocation_sha256=binding.candidate_invocation_sha256,
        candidate_source_sha256=binding.candidate_source_sha256,
        execution_binding_sha256=binding.execution_binding_sha256,
        terminal_status="completed",
        expected_output_count=1,
        observed_output_count=1,
        outputs_finite=True,
        inputs_unchanged=True,
        ipc_envelope_valid=True,
        elapsed_seconds=-0.0,
    )

    assert runtime.elapsed_seconds == 0.0
    assert str(runtime.elapsed_seconds) == "0.0"
