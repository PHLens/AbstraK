from __future__ import annotations

import json

from abstrak.doctor import check_command, main


def test_missing_command_is_reported() -> None:
    result = check_command("missing", ["akl-command-that-does-not-exist"])

    assert result.name == "missing"
    assert result.available is False
    assert result.detail == "not found"


def test_json_report_is_parseable(capsys) -> None:
    assert main(["--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["python"]
    assert {check["name"] for check in report["checks"]} == {
        "docker",
        "git",
        "gpu",
        "nvcc",
        "uv",
    }
