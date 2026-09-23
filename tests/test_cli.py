"""CLI: коды выхода, JSON-вывод, валидация."""

from __future__ import annotations

import json

from akim_sim.cli import main

from .conftest import WORKED_EXAMPLE_ITEMS


def test_cli_measures_json(capsys) -> None:
    assert main(["--measures", *WORKED_EXAMPLE_ITEMS, "--offline", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["engine"]["score"] == 56.543
    assert report["meta"]["mode"] == "measures"


def test_cli_allocation_console(capsys) -> None:
    assert main(["--alloc", "20", "20", "20", "20", "20", "--offline", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "пять лимитов" in out and "МОДЕЛЬНЫЕ ОЦЕНКИ" in out


def test_cli_invalid_plan_exit_code_and_reasons(capsys) -> None:
    assert main(["--measures", "M1:Нура", "M3:Нура", "M9:Нура", "M12", "M14", "--offline"]) == 2
    err = capsys.readouterr().err
    assert "Score не считается" in err and "M1 и M3" in err


def test_cli_validate(capsys) -> None:
    assert main(["--validate", "--measures", *WORKED_EXAMPLE_ITEMS]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_cli_input_file(tmp_path, capsys) -> None:
    f = tmp_path / "plan.json"
    f.write_text(json.dumps({"mode": "measures", "measures": WORKED_EXAMPLE_ITEMS}), encoding="utf-8")
    assert main(["-i", str(f), "--offline", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["engine"]["cost_total"] == 95


def test_cli_allocation_with_custom_budget(capsys) -> None:
    assert main(["--alloc", "30", "20", "30", "20", "30", "--budget", "150", "--offline", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["engine"]["budget_total"] == 150
    assert len(report["engine"]["plan"]) == 5
    assert report["engine"]["cost_total"] <= 150


def test_cli_infeasible_caps_have_no_score(capsys) -> None:
    assert main(["--alloc", "100", "0", "0", "0", "0", "--offline", "--json"]) == 2
    assert "Score не считается" in capsys.readouterr().err


def test_cli_catalog(capsys) -> None:
    assert main(["--catalog", "--no-color"]) == 0
    assert "M14" in capsys.readouterr().out
