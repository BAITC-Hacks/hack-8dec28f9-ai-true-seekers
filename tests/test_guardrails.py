"""Guardrails: числа в текстах агентов должны иметь источник в движке."""

from __future__ import annotations

import asyncio

import pytest

from akim_sim.engine import analyze, config
from akim_sim.guardrails import build_reference, check_texts, fact_check
from akim_sim.inputs import AllocationDecision, MeasuresDecision
from akim_sim.pipeline import AkimSimulator

from .conftest import WORKED_EXAMPLE_ITEMS


@pytest.fixture(scope="module")
def engine_report() -> dict:
    return analyze(MeasuresDecision(measures=WORKED_EXAMPLE_ITEMS))


def _check(text: str, engine: dict) -> dict:
    return fact_check({"agent": {"text": text}}, engine, {}, config())


def test_engine_numbers_pass_with_provenance(engine_report: dict) -> None:
    e = engine_report
    text = (f"Score {e['score']:.3f}, слабейший район {e['weakest']['district']} — {e['weakest']['score']:.3f}; "
            f"реализовано {e['potential_realized_pct']:.1f}% потенциала.")
    rep = _check(text, e)
    assert rep["status"] == "pass"
    sources = [p["source"] for p in rep["agents"]["agent"]["provenance"]]
    assert "engine.score" in sources and "engine.weakest.score" in sources


def test_hallucinated_numbers_are_flagged(engine_report: dict) -> None:
    rep = _check("Score вырастет до 88.8, а пробки упадут на 42.5%.", engine_report)
    assert rep["status"] == "warn"
    assert {u["value"] for u in rep["unverified"]} == {"88.8", "42.5"}


def test_unknown_measure_codes_are_flagged(engine_report: dict) -> None:
    rep = _check("Добавьте M15 и M3 в Нуре.", engine_report)
    assert [u["code"] for u in rep["unknown_codes"]] == ["M15"]
    assert rep["status"] == "warn"


def test_small_counts_ranges_and_codes_are_not_metrics(engine_report: dict) -> None:
    rep = _check("3 шага; Квартал 2; Дни 30–60; показатель T1 и мера M14.", engine_report)
    assert rep["checked_numbers"] == 0 and rep["status"] == "pass"


def test_percent_form_of_fraction_is_accepted(engine_report: dict) -> None:
    rep = _check("Мера с лагом 1 реализуется на 87.5% за горизонт.", engine_report)  # 0.875 → 87.5%
    assert rep["status"] == "pass"


def test_integer_threshold_narrowed_to_five() -> None:
    values, paths = build_reference({"engine": {"x": 1.0}})
    rep = check_texts({"a": {"t": "осталось 7 пунктов"}}, values, paths)
    assert rep["checked_numbers"] == 1 and rep["status"] == "warn"


@pytest.mark.parametrize("decision", [
    MeasuresDecision(measures=WORKED_EXAMPLE_ITEMS),
    AllocationDecision(allocations={"transport": 25, "green": 15, "social": 25, "safety": 15, "utilities": 20}),
])
def test_rule_based_agents_pass_guardrails(decision) -> None:
    report = asyncio.run(AkimSimulator(offline=True).run(decision))
    g = report["guardrails"]
    assert g["status"] == "pass", g["unverified"]
    assert g["checked_numbers"] > 50
