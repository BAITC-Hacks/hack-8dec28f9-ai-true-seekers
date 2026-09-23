"""Риски — модельные прокси, а не вероятности: подписи в JSON, консоли и текстах агентов."""

from __future__ import annotations

import asyncio

import pytest

from akim_sim.agents import RiskItem, SimulationContext, SocialRiskAgent, SocialRiskOutput
from akim_sim.inputs import MeasuresDecision
from akim_sim.pipeline import AkimSimulator
from akim_sim.report import render
from akim_sim.risk import RISK_DISCLAIMER

from .conftest import WORKED_EXAMPLE_ITEMS


@pytest.fixture(scope="module")
def report() -> dict:
    return asyncio.run(AkimSimulator(offline=True).run(MeasuresDecision(measures=WORKED_EXAMPLE_ITEMS)))


def test_every_risk_is_labelled_model_proxy(report: dict) -> None:
    proxies = report["engine"]["risk_proxies"]
    assert proxies
    for r in proxies.values():
        assert r["kind"] == "model_proxy"
        assert 0 <= r["value"] <= 100
        assert r["basis"]
    assert RISK_DISCLAIMER in report["engine"]["disclaimers"]
    assert "не вероятности" in RISK_DISCLAIMER and "не откалиброваны" in RISK_DISCLAIMER


def test_sensitivity_is_labelled(report: dict) -> None:
    assert all(s["kind"] == "sensitivity_analysis" for s in report["engine"]["sensitivity"])


def test_console_shows_disclaimers(report: dict) -> None:
    text = render(report, color=False)
    assert "МОДЕЛЬНЫЕ ОЦЕНКИ (прокси 0–100, не вероятности)" in text
    assert "не симуляция событий" in text
    assert "вероятност" not in text.replace("не вероятност", "").lower()


def test_social_agent_risk_names_are_labelled(report: dict) -> None:
    risks = report["agents"]["social_risk"]["output"]["risks"]
    assert risks and all("модельная оценка" in r["name"] for r in risks)


def test_llm_risk_names_get_label_in_postprocess(report: dict) -> None:
    agent = SocialRiskAgent()
    ctx = SimulationContext(decision=MeasuresDecision(measures=WORKED_EXAMPLE_ITEMS), engine=report["engine"])
    out = SocialRiskOutput(headline="h", public_mood="m", simulated_posts=[], vulnerable_groups=[],
                           communication_advice="a",
                           risks=[RiskItem(name="Риск аварий ЖКХ", level="high", trigger="t", mitigation="m")])
    fixed = agent.postprocess(out, ctx)
    assert fixed.risks[0].name.endswith("(модельная оценка)")
