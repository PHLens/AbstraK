from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from abstrak.canary.artifacts import TrajectoryStore, verify_trajectory
from abstrak.canary.contracts import TrajectoryOutcome
from abstrak.canary.schedule import build_r1_schedule
from abstrak.canary.study import (
    FORMAL_REQUEST_CEILING,
    StudyRuntime,
    _cell_command,
    main,
    run_formal_study,
)


def _runtime(tmp_path: Path) -> StudyRuntime:
    return StudyRuntime(
        artifact_root=str(tmp_path),
        study_id="formal-test",
        ssh_host="worker",
        worker_root="/volume/AbstraK",
        worker_timeout=300.0,
        allow_supervised_worker=True,
        revision="1" * 40,
    )


def test_cell_command_contains_frozen_identity_and_guards(tmp_path: Path) -> None:
    cell = build_r1_schedule().formal[0]
    command = _cell_command(cell, _runtime(tmp_path))

    assert command[:3] == [command[0], "-m", "abstrak.canary.cli"]
    assert command[command.index("--trajectory-id") + 1] == cell.trajectory_id
    assert command[command.index("--task") + 1] == cell.task_id
    assert command[command.index("--target") + 1] == cell.target_id
    assert command[command.index("--profile") + 1] == cell.agent_id
    assert command[command.index("--study-id") + 1] == "formal-test"
    assert command[command.index("--expected-max-requests") + 1] == "4"
    assert "--allow-supervised-worker" in command


def test_formal_runner_seals_all_cells_and_resumes_without_rerun(
    tmp_path: Path, capsys
) -> None:
    runtime = _runtime(tmp_path)
    calls: list[list[str]] = []

    def fake_process(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        trajectory_id = command[command.index("--trajectory-id") + 1]
        study_id = command[command.index("--study-id") + 1]
        root = command[command.index("--artifact-root") + 1]
        now = datetime.now(timezone.utc)
        outcome = TrajectoryOutcome(
            trajectory_id=trajectory_id,
            status="no_candidate",
            calls=0,
            usage_complete=True,
            started_at_utc=now,
            finished_at_utc=now,
        )
        store = TrajectoryStore.create(root, study_id, trajectory_id)
        store.write_json("outcome.json", outcome)
        store.seal()
        return subprocess.CompletedProcess(command, 0, "", "")

    first = run_formal_study(runtime, run_process=fake_process)
    second = run_formal_study(
        runtime,
        run_process=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sealed cells must not rerun")
        ),
    )

    assert first.completed_cells == second.completed_cells == 48
    assert first.resumed_cells == 0
    assert second.resumed_cells == 48
    assert len(calls) == 48
    assert len({record.trajectory_id for record in first.records}) == 48
    verify_trajectory(tmp_path / "formal-test" / "study-manifest")
    assert len(capsys.readouterr().out.splitlines()) == 96


def test_study_cli_guards_precede_revision_and_execution(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "abstrak.canary.study._repository_revision",
        lambda: (_ for _ in ()).throw(AssertionError("revision resolved before guard")),
    )

    missing_live = main(["--expected-max-requests", str(FORMAL_REQUEST_CEILING)])
    wrong_ceiling = main(["--live", "--expected-max-requests", "4"])

    assert missing_live == wrong_ceiling == 2
    assert "configuration error" in capsys.readouterr().err
