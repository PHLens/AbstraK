from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from abstrak.anytime.isolation import (
    AnytimeCandidateInvocation,
    AnytimeCandidateOutput,
    AnytimeIsolationError,
    AnytimeOutputChannel,
    AnytimePublicRuntime,
    AnytimePublicTaskABI,
    AnytimeQualifierInvocation,
    AnytimeTensorDescriptor,
    build_anytime_candidate_source,
    build_anytime_process_isolation_contract,
    verify_anytime_candidate_invocation,
    verify_anytime_candidate_output,
    verify_anytime_isolation_contract,
)

_HASH = "a" * 64
_CHANNEL = "b" * 32


def _tensor(name: str, *, shape: tuple[int, ...] = (32, 32)) -> AnytimeTensorDescriptor:
    return AnytimeTensorDescriptor(
        name=name,
        shape=shape,
        strides=(shape[-1], 1),
        dtype="float16",
    )


def _invocation(source: str = "def candidate():\n    return 1\n") -> AnytimeCandidateInvocation:
    return AnytimeCandidateInvocation(
        source=build_anytime_candidate_source(source),
        public_abi=AnytimePublicTaskABI(
            abi_id="matrix-add",
            abi_version="v1",
            entrypoint="ModelNew.forward",
            input_names=("x", "y"),
            output_count=1,
        ),
        public_runtime=AnytimePublicRuntime(
            target_id="triton-a100",
            backend="triton",
            runtime_id="triton-3.7.1",
            runtime_abi_version="kernelbench-v1",
        ),
        inputs=(_tensor("x"), _tensor("y")),
        output_channel=AnytimeOutputChannel(
            channel_id=_CHANNEL,
            expected_output_count=1,
        ),
    )


def test_candidate_invocation_is_a_minimal_public_whitelist() -> None:
    invocation = _invocation()

    assert set(invocation.model_dump()) == {
        "schema_version",
        "source",
        "public_abi",
        "public_runtime",
        "inputs",
        "output_channel",
    }
    assert (
        invocation.source.source_sha256
        == hashlib.sha256(invocation.source.text.encode("utf-8")).hexdigest()
    )
    assert invocation.public_runtime.model_dump().keys() == {
        "schema_version",
        "target_id",
        "backend",
        "runtime_id",
        "runtime_abi_version",
        "accelerator",
    }
    assert invocation.output_channel.candidate_may_report_timing is False
    assert verify_anytime_candidate_invocation(invocation) == invocation


def test_candidate_invocation_rejects_private_fields_paths_and_bad_descriptors() -> None:
    payload = _invocation().model_dump(mode="json")
    payload["sealed_case_path"] = "/home/cambricon/AbstraK/benchmarks/private.json"
    with pytest.raises(AnytimeIsolationError, match="extra_forbidden"):
        verify_anytime_candidate_invocation(payload)

    with pytest.raises(ValidationError, match="pattern"):
        AnytimePublicRuntime(
            target_id="triton-a100",
            backend="triton",
            runtime_id="repo/private",
            runtime_abi_version="v1",
        )
    with pytest.raises(ValidationError, match="private benchmark capability"):
        AnytimePublicTaskABI(
            abi_id="sealed-task",
            abi_version="v1",
            entrypoint="forward",
            input_names=("x",),
            output_count=1,
        )
    with pytest.raises(ValidationError, match="shape and stride rank"):
        AnytimeTensorDescriptor(
            name="x",
            shape=(2, 3),
            strides=(1,),
            dtype="float16",
        )

    mismatched = _invocation().model_dump(mode="json")
    mismatched["inputs"] = list(reversed(mismatched["inputs"]))
    with pytest.raises(AnytimeIsolationError, match="exactly follow"):
        verify_anytime_candidate_invocation(mismatched)


def test_exact_source_and_model_copy_tampering_are_revalidated() -> None:
    invocation = _invocation()
    tampered_source = invocation.source.model_copy(update={"text": invocation.source.text + "\n"})
    tampered = invocation.model_copy(update={"source": tampered_source})

    with pytest.raises(AnytimeIsolationError, match="source digest"):
        verify_anytime_candidate_invocation(tampered)

    valid_with_newline = _invocation(invocation.source.text + "\n")
    assert valid_with_newline.source.source_sha256 != invocation.source.source_sha256


def test_candidate_output_cannot_forge_timing_or_qualification() -> None:
    invocation = _invocation()
    output_descriptor = _tensor("output")
    output = AnytimeCandidateOutput(
        channel_id=_CHANNEL,
        outputs=(output_descriptor,),
    )
    assert verify_anytime_candidate_output(output, invocation=invocation) == output

    payload = output.model_dump(mode="json")
    payload["latency_ms"] = 0.001
    payload["target_use_verified"] = True
    with pytest.raises(AnytimeIsolationError, match="extra_forbidden"):
        verify_anytime_candidate_output(payload, invocation=invocation)

    wrong_channel = output.model_copy(update={"channel_id": "c" * 32})
    with pytest.raises(AnytimeIsolationError, match="wrong controller channel"):
        verify_anytime_candidate_output(wrong_channel, invocation=invocation)

    too_many = output.model_copy(update={"outputs": (output_descriptor, output_descriptor)})
    with pytest.raises(AnytimeIsolationError, match="output count"):
        verify_anytime_candidate_output(too_many, invocation=invocation)


def test_two_role_contract_is_contract_ready_but_live_containment_pending() -> None:
    contract = build_anytime_process_isolation_contract(
        max_wall_seconds=30.0,
        max_memory_bytes=2**30,
    )

    assert contract.topology == "distinct-processes"
    assert contract.candidate.role == "candidate"
    assert contract.qualifier.role == "reference-qualifier"
    assert contract.candidate.private_asset_access is False
    assert contract.candidate.repository_access is False
    assert contract.direct_candidate_to_qualifier_ipc is False
    assert contract.real_os_containment_status == "pending-m9"
    assert contract.real_os_containment_observed is False
    assert contract.candidate_execution_performed is False

    forged = contract.model_copy(update={"real_os_containment_observed": True})
    with pytest.raises(AnytimeIsolationError, match="literal_error"):
        verify_anytime_isolation_contract(forged)


def test_qualifier_gets_hash_capabilities_not_candidate_visible_paths() -> None:
    qualifier = AnytimeQualifierInvocation(
        candidate_invocation_sha256=_HASH,
        candidate_output_sha256="b" * 64,
        private_asset_bundle_sha256="c" * 64,
        reference_source_sha256="d" * 64,
        sealed_case_bundle_sha256="e" * 64,
        execution_binding_sha256="f" * 64,
    )
    assert set(qualifier.model_dump()) == {
        "schema_version",
        "candidate_invocation_sha256",
        "candidate_output_sha256",
        "private_asset_bundle_sha256",
        "reference_source_sha256",
        "sealed_case_bundle_sha256",
        "execution_binding_sha256",
    }

    payload = qualifier.model_dump(mode="json")
    payload["reference_source_path"] = "/repo/reference.py"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AnytimeQualifierInvocation.model_validate(payload)
