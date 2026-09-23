"""Пайплайн: движок → 3 агента параллельно → оркестратор → guardrails → единый JSON-отчёт."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from . import __version__
from .agents import AgentResult, AnalyticsAgent, OrchestratorAgent, SimulationContext, SocialRiskAgent, UrbanExpertAgent
from .engine import analyze, config
from .guardrails import fact_check
from .inputs import Decision
from .llm import LLMClient


class AkimSimulator:
    def __init__(self, *, offline: bool = False, model: str | None = None, lang: str | None = None,
                 llm: LLMClient | None = None) -> None:
        self.offline = offline
        self.model_override = model
        self.lang = lang or os.getenv("AKIM_LANG", "ru")
        self._llm = llm  # для тестов можно передать готовый клиент
        self.analytics = AnalyticsAgent()
        self.urban = UrbanExpertAgent()
        self.social = SocialRiskAgent()
        self.orchestrator = OrchestratorAgent()

    def _client(self) -> LLMClient | None:
        if self.offline:
            return None
        if self._llm is not None:
            return self._llm
        try:
            return LLMClient.from_env(self.model_override)
        except Exception as exc:
            print(f"[warn] AI-клиент недоступен: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    async def run(self, decision: Decision) -> dict[str, Any]:
        t0 = time.perf_counter()
        engine_report = analyze(decision)  # PlanValidationError / ValueError — до вызова LLM
        engine_ms = int((time.perf_counter() - t0) * 1000)
        ctx = SimulationContext(decision=decision, engine=engine_report, lang=self.lang)
        llm = self._client()
        if llm is None and not self.offline:
            print("[info] OPENAI_API_KEY не найден — агенты работают в rule-based режиме.", file=sys.stderr)
        try:
            a, u, s = await asyncio.gather(self.analytics.run(ctx, llm), self.urban.run(ctx, llm),
                                           self.social.run(ctx, llm))
            results: dict[str, AgentResult] = {"analytics": a, "urban_expert": u, "social_risk": s}
            orch = await self.orchestrator.run(ctx, llm, results=results)
        finally:
            if llm is not None and self._llm is None:
                await llm.aclose()
        all_results = {**results, "orchestrator": orch}
        guard = fact_check({k: r.output.model_dump() for k, r in all_results.items()}, engine_report,
                           decision.model_dump(), config())
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for r in all_results.values():
            for k in usage:
                usage[k] += r.usage.get(k, 0)
        return {
            "meta": {
                "app": "Astana Akim Simulator «Аким на 5 часов» — advanced", "version": __version__,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "mode": decision.mode, "agents_mode": "llm" if llm else "offline", "lang": self.lang,
                "model": llm.model if llm else None,
                "orchestrator_model": llm.orchestrator_model if llm else None,
                "latency_ms": {"engine": engine_ms, "total": int((time.perf_counter() - t0) * 1000)},
                "usage": usage,
            },
            "input": decision.model_dump(),
            "engine": engine_report,
            "agents": {k: r.to_dict() for k, r in all_results.items()},
            "final_report": orch.output.model_dump(),
            "guardrails": guard,
        }
