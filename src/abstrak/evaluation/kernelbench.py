"""Pinned-checkout adapter for KernelBench tasks and zero-shot prompts."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import tomli
from pydantic import BaseModel, ConfigDict

from abstrak.evaluation.contracts import (
    KernelBenchSource,
    KernelBenchTask,
    Precision,
    StudyError,
    TargetName,
)

ZERO_SHOT_COMPONENTS = ["problem_statement", "arch_block", "precision_note", "instruction"]


class TaskMaterial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task: KernelBenchTask
    name: str
    relative_path: str
    source: str
    source_sha256: str


class ExtractedCode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str | None
    status: str


class KernelBenchCheckout:
    def __init__(self, root: str | Path, source: KernelBenchSource) -> None:
        self.root = Path(root).expanduser().resolve()
        self.source = source
        self.prompts_path = self.root / "src" / "kernelbench" / "prompts" / "prompts.toml"
        self._validate()

    def _validate(self) -> None:
        if not self.root.is_dir() or not self.prompts_path.is_file():
            raise StudyError(f"{self.root} is not a KernelBench checkout")
        try:
            commit = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise StudyError(f"cannot inspect KernelBench checkout: {error}") from error
        if commit != self.source.commit:
            raise StudyError(
                f"KernelBench commit mismatch: expected {self.source.commit}, found {commit}"
            )
        if self.source.require_clean_checkout:
            status = subprocess.run(
                ["git", "-C", str(self.root), "status", "--porcelain=v1"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            if status:
                raise StudyError("KernelBench checkout must be clean")

    def load_task(self, task: KernelBenchTask) -> TaskMaterial:
        level_directory = self.root / "KernelBench" / f"level{task.level}"
        matches = sorted(level_directory.glob(f"{task.problem_id}_*.py"))
        if len(matches) != 1:
            raise StudyError(
                f"expected one source for {task.ref}, found {len(matches)} in {level_directory}"
            )
        path = matches[0].resolve()
        if self.root not in path.parents:
            raise StudyError(f"task source escaped checkout: {path}")
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            raise StudyError(f"cannot read task source {path}: {error}") from error
        name = path.stem.partition("_")[2].replace("_", " ").strip()
        return TaskMaterial(
            task=task,
            name=name,
            relative_path=str(path.relative_to(self.root)),
            source=source,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        )

    def zero_shot_prompt(
        self, material: TaskMaterial, target: TargetName, precision: Precision
    ) -> str:
        try:
            with self.prompts_path.open("rb") as handle:
                config = tomli.load(handle)
            configured = config["options"]["zero_shot"]["components"]
            if configured != ZERO_SHOT_COMPONENTS:
                raise StudyError(
                    "pinned KernelBench zero_shot components changed; review before running"
                )
            backend_display = config["backends"][target]["backend_display"]
            precision_display = config["precision"][precision]["precision_display"]
            shared = config["shared"]
            common = config["templates"]["common"]
            context = {
                "backend_display": backend_display,
                "ref_arch_src": material.source,
                "precision_display": precision_display,
            }
            blocks = [
                shared["problem_statement"].format(**context),
                common["arch_block"].format(**context),
                common["precision_note"].format(**context),
                shared["instruction"].format(**context),
            ]
        except StudyError:
            raise
        except (KeyError, OSError, tomli.TOMLDecodeError) as error:
            raise StudyError(f"cannot render pinned KernelBench prompt: {error}") from error
        return "\n".join(blocks).strip() + "\n"


def extract_first_code(response: str) -> ExtractedCode:
    """Mirror KernelBench's first-code-block policy without importing Torch."""

    match = re.search(r"```(.*?)```", response.strip(), re.DOTALL)
    if match is None:
        return ExtractedCode(code=None, status="no_code_block")
    code = match.group(1).strip()
    for language in ("python", "cpp"):
        if code.startswith(language):
            code = code[len(language) :].strip()
            break
    if not code:
        return ExtractedCode(code=None, status="empty_code_block")
    return ExtractedCode(code=code, status="extracted")


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()
