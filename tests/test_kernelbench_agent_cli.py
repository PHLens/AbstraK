from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from abstrak.evaluation import cli
from abstrak.evaluation.agent_contracts import load_agent_study
from abstrak.evaluation.agent_runner import AgentCollectionResult


def test_parser_exposes_split_agent_stages_and_pipeline() -> None:
    parser = cli._parser()
    collect = parser.parse_args(
        [
            "agent-collect",
            "--study",
            "study.yaml",
            "--ssh-host",
            "a100-r1",
            "--worker-root",
            "/srv/AbstraK",
            "--worker-kernelbench-root",
            "/srv/KernelBench",
            "--live",
        ]
    )
    assert collect.command == "agent-collect"
    assert collect.live is True
    assert collect.iterations is None
    assert parser.parse_args(["agent-analyze", "--run", "/tmp/run"]).command == "agent-analyze"
    assert parser.parse_args(["agent-plot", "--run", "/tmp/run"]).command == "agent-plot"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("level1:1", (1, 1)), ("1:40", (1, 40)), ("level2-problem76", (2, 76))],
)
def test_parse_agent_task(value: str, expected: tuple[int, int]) -> None:
    assert cli._parse_agent_task(value) == expected


def test_parse_agent_task_rejects_ambiguous_reference() -> None:
    with pytest.raises(ValueError, match="--task must look like"):
        cli._parse_agent_task("level1/1")


def test_collect_requires_explicit_live_flag() -> None:
    arguments = Namespace(command="agent-collect", live=False)
    with pytest.raises(ValueError, match="requires --live"):
        cli._collect_agent(arguments)


def test_pipeline_chains_collect_and_derived_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    study = load_agent_study(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "studies"
        / "kernelbench-agent-pilot.yaml"
    )
    run_directory = tmp_path / "run"
    collection = AgentCollectionResult(
        run_directory=run_directory,
        attempts=4,
        generation_status_counts={"generated": 4},
        evaluation_status_counts={"evaluated": 4},
    )
    calls: list[str] = []

    def fake_collect(arguments: Namespace) -> tuple[object, AgentCollectionResult]:
        calls.append("collect")
        return study, collection

    def fake_analyze(path: Path) -> tuple[dict[str, str], Path, Path]:
        calls.append("analyze")
        assert path == run_directory
        return {"run_id": "run"}, path / "metrics.json", path / "metrics.csv"

    def fake_plot(path: Path) -> tuple[Path, Path, Path, Path]:
        calls.append("plot")
        assert path == run_directory
        return tuple(path / name for name in ("a.png", "a.pdf", "b.png", "b.pdf"))  # type: ignore[return-value]

    monkeypatch.setattr(cli, "_collect_agent", fake_collect)
    monkeypatch.setattr(cli, "analyze_agent_run", fake_analyze)
    monkeypatch.setattr(cli, "plot_agent_run", fake_plot)
    arguments = Namespace(command="agent-pipeline", live=True)

    status = cli._agent_pipeline(arguments)

    assert status == cli.EXIT_OK
    assert calls == ["collect", "analyze", "plot"]


def test_analyze_command_does_not_require_live(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "analyze_agent_run",
        lambda path: ({"run_id": "offline"}, Path("metrics.json"), Path("metrics.csv")),
    )
    status = cli.main(["agent-analyze", "--run", "/tmp/offline-run"])
    payload = json.loads(capsys.readouterr().out)
    assert status == cli.EXIT_OK
    assert payload["run_id"] == "offline"
