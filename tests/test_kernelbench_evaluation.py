from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from abstrak.config import AppConfig, ConfigProfile
from abstrak.evaluation.artifacts import (
    EvaluationArtifactError,
    seal_directory,
    verify_directory_checksums,
    write_derived_json,
)
from abstrak.evaluation.cli import main as evaluation_cli
from abstrak.evaluation.contracts import (
    EvaluationResult,
    KernelBenchNaiveStudy,
    KernelBenchSource,
    KernelBenchTask,
    StudyError,
)
from abstrak.evaluation.evaluator import evaluate_run
from abstrak.evaluation.generation import NaiveGenerationRunner
from abstrak.evaluation.kernelbench import KernelBenchCheckout, extract_first_code
from abstrak.evaluation.summary import summarize_run
from abstrak.providers.manifests import (
    GenerationConfig,
    ManifestBundle,
    ModelManifest,
    ProviderManifest,
)


class FakeUsage(BaseModel):
    input_tokens: int = 100
    output_tokens: int = 200


class FakeResponse(BaseModel):
    text: str
    returned_model: str = "returned-model"
    finish_reason: str = "stop"
    elapsed_ms: float = 12.5
    usage: FakeUsage = FakeUsage()


class FakeClient:
    def __init__(self, bundle: ManifestBundle, output: str) -> None:
        self.bundle = bundle
        self.output = output
        self.requests: list[Any] = []

    @property
    def resolved_manifest_record(self) -> dict[str, Any]:
        return {
            "provider": self.bundle.provider.model_dump(mode="json"),
            "model": self.bundle.model.model_dump(mode="json"),
        }

    def complete(self, request: Any) -> FakeResponse:
        self.requests.append(request)
        return FakeResponse(text=self.output)


def _fake_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "KernelBench"
    prompts = root / "src" / "kernelbench" / "prompts"
    tasks = root / "KernelBench" / "level1"
    prompts.mkdir(parents=True)
    tasks.mkdir(parents=True)
    (prompts / "prompts.toml").write_text(
        """
[shared]
problem_statement = "Write {backend_display}."
instruction = "Create ModelNew with {backend_display} in one codeblock."

[backends.triton]
backend_display = "Triton kernels"
[backends.tilelang]
backend_display = "TileLang kernels"
[backends.cute]
backend_display = "CuTe kernels"

[precision.fp16]
precision_display = "FP16"

[templates.common]
arch_block = "Architecture:\\n{ref_arch_src}"
precision_note = "Precision: {precision_display}."

[options.zero_shot]
components = ["problem_statement", "arch_block", "precision_note", "instruction"]
""".strip(),
        encoding="utf-8",
    )
    (tasks / "1_Add.py").write_text(
        "class Model: pass\n\ndef get_inputs(): return []\n\ndef get_init_inputs(): return []\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=AbstraK Test",
            "-c",
            "user.email=abstrak@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return root, commit


def _study(commit: str, *, profiles: tuple[str, ...] = ("flash", "pro")) -> Any:
    return KernelBenchNaiveStudy(
        id="test-study",
        source=KernelBenchSource(repository="https://example.invalid/kb", commit=commit),
        profiles=profiles,
        targets=("triton", "tilelang", "cute"),
        tasks=(KernelBenchTask(level=1, problem_id=1, stratum="compute"),),
        precision="fp16",
    )


def _app_config(profiles: tuple[str, ...]) -> AppConfig:
    provider = ProviderManifest(id="provider", api_key_env="TEST_API_KEY")
    configured: dict[str, ConfigProfile] = {}
    for profile in profiles:
        model = ModelManifest(
            id=profile,
            provider="provider",
            api_model=f"openai/{profile}",
            model_id_policy="mutable_alias",
            allow_live_probe=True,
            generation=GenerationConfig(max_completion_tokens=128),
            output_contract="plain_json",
        )
        configured[profile] = ConfigProfile(provider=provider, model=model)
    return AppConfig(default_profile=profiles[0], profiles=configured)


def test_checkout_renders_pinned_zero_shot_prompt(tmp_path: Path) -> None:
    root, commit = _fake_checkout(tmp_path)
    study = _study(commit)
    checkout = KernelBenchCheckout(root, study.source)
    material = checkout.load_task(study.tasks[0])

    prompt = checkout.zero_shot_prompt(material, "triton", "fp16")

    assert "Triton kernels" in prompt
    assert "class Model: pass" in prompt
    assert "FP16" in prompt
    assert "example" not in prompt.lower()
    assert "hardware" not in prompt.lower()


def test_checkout_rejects_commit_mismatch(tmp_path: Path) -> None:
    root, _ = _fake_checkout(tmp_path)
    source = KernelBenchSource(repository="https://example.invalid/kb", commit="0" * 40)

    with pytest.raises(StudyError, match="commit mismatch"):
        KernelBenchCheckout(root, source)


def test_study_rejects_fp32_tilelang() -> None:
    with pytest.raises(ValueError, match="TileLang requires fp16 or bf16"):
        KernelBenchNaiveStudy(
            id="invalid-precision",
            source=KernelBenchSource(repository="https://example.invalid/kb", commit="0" * 40),
            profiles=("flash",),
            targets=("tilelang",),
            tasks=(KernelBenchTask(level=1, problem_id=1, stratum="compute"),),
            precision="fp32",
        )


def test_first_code_block_policy() -> None:
    extracted = extract_first_code("before```python\nclass ModelNew: pass\n```after")
    missing = extract_first_code("class ModelNew: pass")
    empty = extract_first_code("before```python\n```after")
    first = extract_first_code("```python\nfirst = 1\n```\n```python\nsecond = 2\n```")

    assert extracted.code == "class ModelNew: pass"
    assert extracted.status == "extracted"
    assert missing.code is None
    assert missing.status == "no_code_block"
    assert empty.code is None
    assert empty.status == "empty_code_block"
    assert first.code == "first = 1"


def test_single_turn_matrix_generation_writes_private_cells(tmp_path: Path) -> None:
    root, commit = _fake_checkout(tmp_path)
    study = _study(commit)
    config = _app_config(study.profiles)
    clients: list[FakeClient] = []

    def client_factory(bundle: ManifestBundle, environment: Any) -> FakeClient:
        assert environment["TEST_API_KEY"] == "unit-secret"
        client = FakeClient(bundle, "```python\nclass ModelNew: pass\n```")
        clients.append(client)
        return client

    runner = NaiveGenerationRunner(
        study=study,
        config=config,
        environment={"TEST_API_KEY": "unit-secret"},
        checkout=KernelBenchCheckout(root, study.source),
        artifact_root=tmp_path / "artifacts",
        run_id="test-run",
        client_factory=client_factory,
    )

    run_directory, counts = runner.run()

    cells = sorted((run_directory / "cells").iterdir())
    assert len(cells) == study.matrix_size == 6
    assert counts == {"generated": 6}
    assert sum(len(client.requests) for client in clients) == 6
    assert all(len(request.messages) == 1 for client in clients for request in client.requests)
    assert all(request.turn_index == 0 for client in clients for request in client.requests)
    assert all(client.bundle.model.generation.max_completion_tokens == 8192 for client in clients)
    assert all(client.bundle.model.output_contract == "plain_text" for client in clients)
    assert all(client.bundle.model.allow_live_probe is False for client in clients)
    assert all(
        config.bundle(profile).model.generation.max_completion_tokens == 128
        for profile in study.profiles
    )
    assert all(
        config.bundle(profile).model.output_contract == "plain_json" for profile in study.profiles
    )
    assert all(config.bundle(profile).model.allow_live_probe is True for profile in study.profiles)
    assert all((cell / "candidate.py").is_file() for cell in cells)
    assert all((cell / "extraction.json").is_file() for cell in cells)
    assert all((cell / "generation.sha256sums").is_file() for cell in cells)
    assert all(cell.stat().st_mode & 0o777 == 0o500 for cell in cells)
    assert all(
        b"unit-secret" not in path.read_bytes()
        for path in run_directory.rglob("*")
        if path.is_file()
    )


def test_no_candidate_evaluation_and_summary_need_no_gpu(tmp_path: Path) -> None:
    root, commit = _fake_checkout(tmp_path)
    study = _study(commit, profiles=("flash",)).model_copy(update={"targets": ("triton",)})
    config = _app_config(study.profiles)

    def client_factory(bundle: ManifestBundle, environment: Any) -> FakeClient:
        return FakeClient(bundle, "plain text without a code block")

    runner = NaiveGenerationRunner(
        study=study,
        config=config,
        environment={"TEST_API_KEY": "unit-secret"},
        checkout=KernelBenchCheckout(root, study.source),
        artifact_root=tmp_path / "artifacts",
        run_id="no-candidate",
        client_factory=client_factory,
    )
    run_directory, _ = runner.run()

    counts, _ = evaluate_run(run_directory, root)
    metrics, _ = summarize_run(run_directory)

    assert counts == {"no_candidate": 1}
    assert metrics["groups"][0]["correctness_rate"] == 0.0
    assert metrics["groups"][0]["performance_coverage"] == 0.0
    cell = next((run_directory / "cells").iterdir())
    evaluation = run_directory / "evaluations" / cell.name
    assert (evaluation / "evaluation.json").is_file()
    assert (evaluation / "generation-ref.json").is_file()
    assert (evaluation / "evaluation.sha256sums").is_file()
    assert not (cell / "evaluation.json").exists()


def test_summary_keeps_correctness_denominator_for_speed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cells = run / "cells"
    evaluations = run / "evaluations"
    cells.mkdir(parents=True)
    evaluations.mkdir()
    now = datetime.now(timezone.utc)
    for index, (correct, ratio) in enumerate(((True, 2.5), (False, None))):
        cell = cells / f"cell-{index}"
        cell.mkdir()
        spec = {
            "schema_version": "kernelbench-naive-cell.v1",
            "cell_id": f"cell-{index}",
            "study_id": "study",
            "study_sha256": "hash",
            "profile": "flash",
            "target": "triton",
            "precision": "fp16",
            "task": {"level": 1, "problem_id": index + 1, "stratum": "compute"},
            "task_name": "task",
            "task_source_sha256": "source",
            "prompt_sha256": "prompt",
            "replicate": 0,
        }
        (cell / "cell.json").write_text(json.dumps(spec), encoding="utf-8")
        seal_directory(cell, "generation.sha256sums")
        result = EvaluationResult(
            cell_id=f"cell-{index}",
            status="evaluated",
            backend="triton",
            precision="fp16",
            compiled=True,
            correctness=correct,
            kernel_runtime_ms=1.0 if correct else None,
            reference_runtime_ms=ratio if correct else None,
            performance_ratio=ratio,
            fast_0=correct,
            fast_1=bool(ratio and ratio > 1),
            fast_2=bool(ratio and ratio > 2),
            started_at_utc=now,
            finished_at_utc=now,
        )
        evaluation = evaluations / f"cell-{index}"
        evaluation.mkdir()
        write_derived_json(evaluation / "evaluation.json", result)
        seal_directory(evaluation, "evaluation.sha256sums")

    metrics, _ = summarize_run(run)
    group = metrics["groups"][0]

    assert group["attempted"] == 2
    assert group["correctness_rate"] == 0.5
    assert group["performance_coverage"] == 0.5
    assert group["geomean_performance_ratio_correct"] == pytest.approx(2.5)


def test_evaluator_rejects_tampered_generation_bundle(tmp_path: Path) -> None:
    root, commit = _fake_checkout(tmp_path)
    study = _study(commit, profiles=("flash",)).model_copy(update={"targets": ("triton",)})
    runner = NaiveGenerationRunner(
        study=study,
        config=_app_config(study.profiles),
        environment={"TEST_API_KEY": "unit-secret"},
        checkout=KernelBenchCheckout(root, study.source),
        artifact_root=tmp_path / "artifacts",
        run_id="tampered",
        client_factory=lambda bundle, environment: FakeClient(
            bundle, "```python\nclass ModelNew: pass\n```"
        ),
    )
    run_directory, _ = runner.run()
    cell = next((run_directory / "cells").iterdir())
    cell.chmod(0o700)
    candidate = cell / "candidate.py"
    candidate.chmod(0o600)
    candidate.write_text("class ModelNew: pass  # changed\n", encoding="utf-8")

    with pytest.raises(EvaluationArtifactError, match="checksum mismatch"):
        evaluate_run(run_directory, root)


def test_checksum_verifier_rejects_extra_bundle_entries(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_derived_json(bundle / "result.json", {"status": "ok"})
    seal_directory(bundle, "sha256sums.txt")
    bundle.chmod(0o700)
    (bundle / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(EvaluationArtifactError, match="do not match"):
        verify_directory_checksums(bundle, "sha256sums.txt")


def test_generate_requires_exact_expected_request_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, commit = _fake_checkout(tmp_path)
    study_path = tmp_path / "study.yaml"
    study_path.write_text(
        "\n".join(
            (
                "schema_version: kernelbench-naive-study.v1",
                "id: cli-study",
                "source:",
                "  repository: https://example.invalid/kb",
                f"  commit: {commit}",
                "profiles: [flash, pro]",
                "targets: [triton, tilelang, cute]",
                "tasks:",
                "  - {level: 1, problem_id: 1, stratum: compute}",
                "precision: fp16",
            )
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version: abstrak-user-config.v1",
                "default_profile: flash",
                "profiles:",
                "  flash:",
                "    provider: {id: provider, api_key_env: TEST_API_KEY}",
                "    model:",
                "      {id: flash, provider: provider, api_model: openai/flash}",
                "  pro:",
                "    provider: {id: provider, api_key_env: TEST_API_KEY}",
                "    model: {id: pro, provider: provider, api_model: openai/pro}",
            )
        ),
        encoding="utf-8",
    )

    status = evaluation_cli(
        (
            "generate",
            "--study",
            str(study_path),
            "--kernelbench-root",
            str(root),
            "--config",
            str(config_path),
            "--live",
            "--expected-requests",
            "5",
        )
    )

    assert status == 2
    assert "must equal the frozen matrix size (6)" in capsys.readouterr().err


def test_evaluate_requires_generated_code_acknowledgement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = evaluation_cli(
        ("evaluate", "--run", str(tmp_path / "run"), "--kernelbench-root", str(tmp_path))
    )

    assert status == 2
    assert "requires --execute-generated-code" in capsys.readouterr().err
