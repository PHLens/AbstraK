"""Single-request generation matrix for the naive KernelBench screen."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from abstrak.config import AppConfig
from abstrak.evaluation.artifacts import StudyRunStore
from abstrak.evaluation.contracts import (
    CellSpec,
    GenerationRecord,
    KernelBenchNaiveStudy,
)
from abstrak.evaluation.kernelbench import (
    KernelBenchCheckout,
    extract_first_code,
    prompt_sha256,
)
from abstrak.providers.client import ProviderClient
from abstrak.providers.contracts import (
    ChatMessage,
    LogicalRequest,
    MessageRole,
    ProviderCallError,
)
from abstrak.providers.manifests import ManifestBundle, ModelManifest, required_environment

ClientFactory = Callable[[ManifestBundle, Mapping[str, str]], ProviderClient]


def default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid4().hex[:10]}"


def cell_id(profile: str, target: str, task_ref: str) -> str:
    return f"{profile}--{target}--{task_ref}--r0"


def _experiment_bundle(bundle: ManifestBundle, study: KernelBenchNaiveStudy) -> ManifestBundle:
    model_payload = bundle.model.model_dump(mode="json")
    generation = model_payload["generation"]
    generation.update(
        {
            "max_completion_tokens": study.generation.max_completion_tokens,
            "temperature": study.generation.temperature,
            "top_p": None,
            "api_seed": None,
            "stop": [],
            "reasoning_effort": None,
        }
    )
    model_payload.update(
        {
            "allow_live_probe": False,
            "output_contract": "plain_text",
            "generation": generation,
        }
    )
    return ManifestBundle(
        provider=bundle.provider,
        model=ModelManifest.model_validate(model_payload),
    )


def _default_client_factory(
    bundle: ManifestBundle, environment: Mapping[str, str]
) -> ProviderClient:
    return ProviderClient(bundle, environment=environment)


class NaiveGenerationRunner:
    def __init__(
        self,
        *,
        study: KernelBenchNaiveStudy,
        config: AppConfig,
        environment: Mapping[str, str],
        checkout: KernelBenchCheckout,
        artifact_root: str | Path,
        run_id: str | None = None,
        client_factory: ClientFactory = _default_client_factory,
    ) -> None:
        self.study = study
        self.config = config
        self.environment = environment
        self.checkout = checkout
        self.client_factory = client_factory
        self.run_id = run_id or default_run_id()
        self._bundles = {
            profile: _experiment_bundle(config.bundle(profile), study) for profile in study.profiles
        }
        secret_values = {
            environment[name]
            for bundle in self._bundles.values()
            for name in required_environment(bundle.provider)
            if environment.get(name)
        }
        self.store = StudyRunStore.create(
            artifact_root,
            study.id,
            self.run_id,
            secrets=tuple(sorted(secret_values)),
        )

    def run(self) -> tuple[Path, dict[str, int]]:
        materials = {task.ref: self.checkout.load_task(task) for task in self.study.tasks}
        self.store.write_json("study.json", self.study)
        self.store.write_json(
            "run.json",
            {
                "schema_version": "kernelbench-naive-run.v1",
                "run_id": self.run_id,
                "study_id": self.study.id,
                "study_sha256": self.study.sha256,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "matrix_size": self.study.matrix_size,
                "single_turn": True,
                "memory": False,
                "workflow": False,
                "repair_loop": False,
            },
        )
        counts: Counter[str] = Counter()
        clients = {
            profile: self.client_factory(bundle, self.environment)
            for profile, bundle in self._bundles.items()
        }
        for profile in self.study.profiles:
            client = clients[profile]
            for target in self.study.targets:
                for task in self.study.tasks:
                    material = materials[task.ref]
                    prompt = self.checkout.zero_shot_prompt(material, target, self.study.precision)
                    identifier = cell_id(profile, target, task.ref)
                    directory = self.store.create_cell(identifier)
                    spec = CellSpec(
                        cell_id=identifier,
                        study_id=self.study.id,
                        study_sha256=self.study.sha256,
                        profile=profile,
                        target=target,
                        precision=self.study.precision,
                        task=task,
                        task_name=material.name,
                        task_source_sha256=material.source_sha256,
                        prompt_sha256=prompt_sha256(prompt),
                    )
                    request = LogicalRequest(
                        model_ref=self._bundles[profile].model.id,
                        messages=(ChatMessage(role=MessageRole.USER, content=prompt),),
                        trajectory_id=identifier,
                        turn_index=0,
                    )
                    prefix = directory.relative_to(self.store.run_directory)
                    self.store.write_json(prefix / "cell.json", spec)
                    self.store.write_text(prefix / "prompt.txt", prompt)
                    self.store.write_text(prefix / "reference.py", material.source)
                    self.store.write_json(prefix / "request.json", request)
                    self.store.write_json(
                        prefix / "resolved-manifest.json", client.resolved_manifest_record
                    )
                    try:
                        response = client.complete(request)
                    except ProviderCallError as error:
                        record = GenerationRecord(
                            cell_id=identifier,
                            status="provider_error",
                            request_id=request.request_id,
                            elapsed_ms=error.record.elapsed_ms,
                        )
                        self.store.write_json(prefix / "error.json", error.record)
                    else:
                        extracted = extract_first_code(response.text)
                        self.store.write_json(prefix / "response.json", response)
                        self.store.write_text(prefix / "model-output.txt", response.text)
                        self.store.write_json(prefix / "extraction.json", extracted)
                        candidate_hash: str | None = None
                        if extracted.code is not None:
                            candidate = f"{extracted.code}\n"
                            self.store.write_text(prefix / "candidate.py", candidate)
                            candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()
                        record = GenerationRecord(
                            cell_id=identifier,
                            status=("generated" if extracted.code is not None else "no_code_block"),
                            request_id=request.request_id,
                            response_model=response.returned_model,
                            finish_reason=response.finish_reason,
                            input_tokens=response.usage.input_tokens,
                            output_tokens=response.usage.output_tokens,
                            elapsed_ms=response.elapsed_ms,
                            candidate_sha256=candidate_hash,
                        )
                    self.store.write_json(prefix / "generation.json", record)
                    self.store.seal_generation_cell(directory)
                    counts[record.status] += 1

        summary: dict[str, Any] = {
            "schema_version": "kernelbench-naive-generation-summary.v1",
            "run_id": self.run_id,
            "study_id": self.study.id,
            "matrix_size": self.study.matrix_size,
            "status_counts": dict(sorted(counts.items())),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.store.write_json("generation-summary.json", summary)
        self.store.verify_no_secrets()
        return self.store.run_directory, dict(counts)
