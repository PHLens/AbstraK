from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from abstrak.anytime.contracts import (
    SHAKEOUT_CHECKPOINT_CALLS,
    AnytimeAgentSpec,
    AnytimeCheckpointPolicy,
    AnytimeCohortSpec,
    AnytimeGenerationSpec,
    AnytimeLoopPolicy,
    AnytimeReasoningSpec,
    AnytimeResourceBudget,
    AnytimeStudySpec,
)
from abstrak.anytime.manifests import AnytimeManifestError, load_anytime_study_spec
from abstrak.canary.manifests import StudyManifestError, load_study_spec


def _shakeout_study() -> AnytimeStudySpec:
    agent = AnytimeAgentSpec(
        id="deepseek-v4-flash",
        provider_id="deepseek",
        model_ref="deepseek-v4-flash",
        native_protocol="chat_completions",
        generation=AnytimeGenerationSpec(
            max_output_tokens=16384,
            reasoning=AnytimeReasoningSpec(
                requested_reasoning_effort="xhigh",
                conformance_requirement="literal_xhigh",
            ),
        ),
    )
    loop = AnytimeLoopPolicy(
        budget=AnytimeResourceBudget(
            max_scientific_calls=4,
            max_total_output_tokens=4 * 16384,
            max_compile_attempts=4,
            max_evaluation_attempts=4,
            max_gpu_seconds=4 * 600.0,
        ),
        checkpoints=AnytimeCheckpointPolicy(calls=SHAKEOUT_CHECKPOINT_CALLS),
    )
    return AnytimeStudySpec(
        study_id="anytime-shakeout",
        study_kind="shakeout",
        seed=20260803,
        agents=(agent,),
        cohorts=(
            AnytimeCohortSpec(
                id="shakeout",
                agent_id=agent.id,
                task_ids=("kb-l1-2",),
                target_ids=("triton-a100",),
                replicates=(1,),
                scoring=False,
                loop=loop,
            ),
        ),
    )


def _write_manifest(path: Path, study: AnytimeStudySpec) -> bytes:
    payload = json.dumps(
        study.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return payload


def test_loader_binds_validated_spec_to_exact_raw_bytes(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    payload = _write_manifest(path, _shakeout_study())
    digest = hashlib.sha256(payload).hexdigest()

    pinned = load_anytime_study_spec(path, expected_sha256=digest)

    assert pinned.path == path.resolve()
    assert pinned.sha256 == digest
    assert pinned.spec == _shakeout_study()
    assert pinned.spec.schema_version == "abstrak-anytime-study-spec.v1"


def test_loader_rejects_wrong_or_malformed_expected_hash(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    _write_manifest(path, _shakeout_study())

    with pytest.raises(AnytimeManifestError, match="SHA-256 mismatch"):
        load_anytime_study_spec(path, expected_sha256="f" * 64)
    with pytest.raises(AnytimeManifestError, match="SHA-256 is invalid"):
        load_anytime_study_spec(path, expected_sha256="not-a-sha")


def test_anytime_and_canary_loaders_reject_each_others_schema(tmp_path: Path) -> None:
    anytime_path = tmp_path / "anytime.json"
    _write_manifest(anytime_path, _shakeout_study())
    repository = Path(__file__).resolve().parents[1]
    canary_path = repository / "benchmarks" / "capability-gate-a100" / "study.json"

    with pytest.raises(StudyManifestError, match="invalid study manifest"):
        load_study_spec(anytime_path)
    with pytest.raises(AnytimeManifestError, match="invalid anytime study manifest"):
        load_anytime_study_spec(canary_path)


def test_loader_rejects_unknown_fields_and_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    payload = _shakeout_study().model_dump(mode="json")
    payload["unknown"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AnytimeManifestError, match="invalid anytime study manifest"):
        load_anytime_study_spec(path)

    payload.pop("unknown")
    payload["schema_version"] = "abstrak-anytime-study-spec.v2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AnytimeManifestError, match="invalid anytime study manifest"):
        load_anytime_study_spec(path)


def test_loader_rejects_missing_nonregular_and_non_utf8_inputs(tmp_path: Path) -> None:
    with pytest.raises(AnytimeManifestError, match="does not exist"):
        load_anytime_study_spec(tmp_path / "missing.json")
    with pytest.raises(AnytimeManifestError, match="not a regular file"):
        load_anytime_study_spec(tmp_path)

    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(AnytimeManifestError, match="not UTF-8"):
        load_anytime_study_spec(binary)
