"""Мультиагентный слой: 3 профильных агента + оркестратор.

Принцип датасета: «LLM получает результат расчёта и объясняет его, сравнивает наборы, советует» —
агенты не считают, все числа приходят из engine. У каждого агента есть детерминированный rule-based
fallback: без ключа, без сети или при сбое API пайплайн выдаёт полный отчёт.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import dataset as ds
from .inputs import Decision
from .llm import LLMClient

SectorKey = Literal["transport", "green", "social", "safety", "utilities"]
AreaKey = Literal["transport", "green", "social", "safety", "utilities", "budget", "districts"]
DistrictName = Literal["Есиль", "Алматы", "Сарыарка", "Байконур", "Нура"]
MeasureCode = Literal["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14"]
Severity = Literal["low", "medium", "high", "critical"]
LANG_NAMES = {"ru": "русском", "kk": "казахском", "en": "английском"}


# ════════════════════════════════════════════════════════════════════════════
# Схемы ответов (Structured Outputs)
# ════════════════════════════════════════════════════════════════════════════


class Issue(BaseModel):
    area: AreaKey = Field(description="Сфера, 'districts' для районных проблем или 'budget' для бюджета в целом")
    severity: Severity
    issue: str = Field(description="Суть проблемы с опорой на числа движка, 1–2 предложения")


class AnalyticsOutput(BaseModel):
    headline: str = Field(description="Главный вывод, до 25 слов")
    score_explanation: str = Field(description="Как сложился Score: D_avg, слабейший район, N_crit — 2–3 предложения")
    issues: list[Issue] = Field(description="1–5 выявленных дисбалансов")
    efficiency: str = Field(description="Где деньги работают сильнее/слабее: лаги, вклад мер или ROI сфер")
    synergy: str = Field(description="Какие синергии собраны и какие упущены")
    key_insights: list[str] = Field(description="3–5 коротких тезисов с числами из входных данных")


class DistrictAssessment(BaseModel):
    district: DistrictName
    verdict: Literal["critical", "weak", "ok", "strong"]
    comment: str = Field(description="1–2 предложения: что решение даёт району и чего не хватает")


class SeasonalRisk(BaseModel):
    season: Literal["winter", "spring", "summer", "autumn"]
    risk: str
    impact: str


class ProjectIdea(BaseModel):
    code: MeasureCode = Field(description="Мера только из каталога M1–M14")
    district: str = Field(description="Район для районной меры или 'город' для городской")
    rationale: str


class UrbanExpertOutput(BaseModel):
    headline: str = Field(description="Главный вывод урбаниста, до 25 слов")
    district_assessments: list[DistrictAssessment] = Field(description="Ровно 5 оценок — по одной на район")
    seasonal_risks: list[SeasonalRisk] = Field(description="3–4 сезонных риска")
    priority_projects: list[ProjectIdea] = Field(description="2–4 меры из каталога, которые стоит добавить или перенести")
    local_insight: str = Field(description="Неочевидный инсайт про Астану и формулу, 2–3 предложения")


class SocialPost(BaseModel):
    persona: str = Field(description="Вымышленный житель: имя, роль, район")
    platform: str
    sentiment: Literal["positive", "neutral", "negative"]
    text: str = Field(description="1–2 предложения живым языком")


class RiskItem(BaseModel):
    name: str = Field(description="Название риска с пометкой «модельная оценка»")
    level: Literal["low", "medium", "high"]
    trigger: str
    mitigation: str = Field(description="Что сделать заранее, желательно мерами каталога")


class SocialRiskOutput(BaseModel):
    headline: str
    public_mood: str = Field(description="Настроение горожан (модельно), 2–3 предложения")
    simulated_posts: list[SocialPost] = Field(description="4–5 симулированных постов вымышленных жителей")
    risks: list[RiskItem] = Field(description="3–5 рисков; уровни — из risk_proxies")
    vulnerable_groups: list[str] = Field(description="2–4 группы жителей")
    communication_advice: str


class Disagreement(BaseModel):
    topic: str
    positions: str
    resolution: str


class Recommendation(BaseModel):
    priority: int = Field(description="1 — самый важный")
    area: AreaKey
    action: str
    rationale: str
    expected_effect: str = Field(description="Эффект: числа только из moves или метрик движка")


class OrchestratorOutput(BaseModel):
    title: str
    executive_summary: str = Field(description="3–5 предложений")
    verdict: str = Field(description="Вердикт одной фразой")
    set_comparison: str = Field(description="Сравнение решения с ориентиром/оптимумом и эталоном датасета")
    consensus: list[str] = Field(description="2–4 пункта согласия агентов")
    disagreements: list[Disagreement] = Field(description="1–3 противоречия и их разрешение")
    recommendations: list[Recommendation] = Field(description="3–5 приоритетных рекомендаций")
    roadmap: list[str] = Field(description="4–6 шагов с учётом лагов (кварталы)")
    closing_statement: str


# ════════════════════════════════════════════════════════════════════════════
# Инфраструктура агентов
# ════════════════════════════════════════════════════════════════════════════

COMMON_RULES = """
ПРАВИЛА (обязательны):
1. Score и все метрики уже посчитал детерминированный движок. Числа бери ТОЛЬКО из входных данных — ничего не пересчитывай
   и не придумывай.
2. risk_proxies — модельные оценки (прокси) 0–100, НЕ вероятности: всегда пиши «модельная оценка» и никогда не выдавай
   их за вероятность события.
3. sensitivity — анализ чувствительности (перевзвешивание показателей), а не симуляция бурана/паводка/миграции.
4. Меры — только из каталога M1–M14 (код и название), районы — Есиль, Алматы, Сарыарка, Байконур, Нура.
5. Не упоминай реальных политиков и чиновников. Пиши конкретно, без воды.
6. Язык ответа: {lang}. Ответ — строго JSON по заданной схеме.
"""


def _rules(lang: str) -> str:
    return COMMON_RULES.format(lang=LANG_NAMES.get(lang, "русском"))


@dataclass
class SimulationContext:
    decision: Decision
    engine: dict[str, Any]
    lang: str = "ru"


@dataclass
class AgentResult:
    key: str
    title: str
    role: str
    source: str  # llm | rules | rules_fallback
    output: BaseModel
    model: str | None = None
    latency_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    response_mode: str | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"agent": self.title, "role": self.role, "source": self.source, "model": self.model,
                "response_mode": self.response_mode, "attempts": self.attempts, "latency_ms": self.latency_ms,
                "usage": self.usage, "error": self.error, "output": self.output.model_dump()}


class BaseAgent:
    key = "base"
    title = "Base"
    role = ""
    output_model: type[BaseModel] = BaseModel

    def system_prompt(self, lang: str) -> str:
        raise NotImplementedError

    def payload(self, ctx: SimulationContext, **kw: Any) -> dict[str, Any]:
        raise NotImplementedError

    def fallback(self, ctx: SimulationContext, **kw: Any) -> BaseModel:
        raise NotImplementedError

    def postprocess(self, out: BaseModel, ctx: SimulationContext, **kw: Any) -> BaseModel:
        return out

    def model_for(self, llm: LLMClient) -> str:
        return llm.model

    async def run(self, ctx: SimulationContext, llm: LLMClient | None, **kw: Any) -> AgentResult:
        t0 = time.perf_counter()
        if llm is None:
            out = self.fallback(ctx, **kw)
            return AgentResult(self.key, self.title, self.role, "rules", out,
                               latency_ms=int((time.perf_counter() - t0) * 1000))
        try:
            out, meta = await llm.structured(system=self.system_prompt(ctx.lang), payload=self.payload(ctx, **kw),
                                             schema=self.output_model, schema_name=self.key, model=self.model_for(llm))
            out = self.postprocess(out, ctx, **kw)
            return AgentResult(self.key, self.title, self.role, "llm", out, model=meta["model"],
                               latency_ms=int((time.perf_counter() - t0) * 1000), usage=meta["usage"],
                               response_mode=meta["mode"], attempts=meta["attempts"])
        except Exception as exc:  # любой сбой LLM → детерминированный fallback
            out = self.fallback(ctx, **kw)
            err = f"{type(exc).__name__}: {exc}"[:400]
            print(f"[warn] {self.title}: LLM недоступен → rule-based fallback ({err})", file=sys.stderr)
            return AgentResult(self.key, self.title, self.role, "rules_fallback", out, model=self.model_for(llm),
                               latency_ms=int((time.perf_counter() - t0) * 1000), error=err)


# ── утилиты для текстов fallback ────────────────────────────────────────────


def _mname(code: str) -> str:
    return f"{code} «{ds.MEASURES[code].name}»"


def _where(district: str) -> str:
    return "на весь город" if district == ds.CITY else f"в районе {district}"


def _cells(cells: list[dict[str, Any]]) -> str:
    return ", ".join(f"{c['district']} {c['indicator']} ({c['value']:.2f})" for c in cells)


def _engine_core(e: dict[str, Any]) -> dict[str, Any]:
    keys = ("mode", "score", "base_score", "delta_vs_base", "potential_realized_pct", "grade", "reference",
            "d_avg", "d_avg_base", "weakest", "n_crit", "n_crit_base", "critical_cells", "critical_cells_fixed",
            "budget_total")
    core = {k: e[k] for k in keys if k in e}
    core.update({k: e[k] for k in ("cost_total", "unspent", "plan", "comparison", "top_plans")})
    if e["mode"] == "allocation":
        core.update({k: e[k] for k in ("allocations", "sectors", "feasible_portfolios_checked", "selection_policy")})
    core["reference"] = {k: v for k, v in e["reference"].items() if k != "plan"}
    return core


def _chosen_codes(e: dict[str, Any]) -> list[tuple[str, str]]:
    return [(p["code"], p["district"]) for p in e["plan"]]


# ════════════════════════════════════════════════════════════════════════════
# Agent 1 · Analytics & Math Engine
# ════════════════════════════════════════════════════════════════════════════


class AnalyticsAgent(BaseAgent):
    key = "analytics"
    title = "Agent 1 · Analytics & Math Engine"
    role = "Объясняет Score: D_avg, слабейший район, N_crit, вклад мер, лаги, синергии, резервы"
    output_model = AnalyticsOutput

    def system_prompt(self, lang: str) -> str:
        return f"""Ты — Agent 1 «Analytics & Math Engine» симулятора «Аким на 5 часов» (Астана).
Движок посчитал официальный Score датасета: {ds.W_AVG}·D_avg + {ds.W_MIN}·min_d D_d − {ds.CRIT_PENALTY:g}·N_crit.
Объясни как главный аналитик: откуда взялся Score (D_avg — 70%, слабейший район — 30%, штраф за пары ниже 40),
где деньги работают слабо (лаги: за 8 кварталов реализуется (8−L)/8 эффекта; contribution — вклад меры в Score;
в режиме сумм — выбранные целые меры при лимитах сфер), какие синергии собраны/упущены, чем решение уступает reference и какие ходы из
moves (проверены движком) вернут баллы.
{_rules(lang)}"""

    def payload(self, ctx: SimulationContext, **kw: Any) -> dict[str, Any]:
        e = ctx.engine
        return {**_engine_core(e), "districts": e["districts"], "synergies": e["synergies"], "moves": e["moves"]}

    def postprocess(self, out: BaseModel, ctx: SimulationContext, **kw: Any) -> BaseModel:
        assert isinstance(out, AnalyticsOutput)
        out.issues = out.issues[:6]
        out.key_insights = out.key_insights[:6]
        return out

    def fallback(self, ctx: SimulationContext, **kw: Any) -> AnalyticsOutput:
        e = ctx.engine
        w = e["weakest"]
        headline = (f"Score {e['score']:.3f} (+{e['delta_vs_base']:.3f} к базе {e['base_score']:.3f}): "
                    f"реализовано {e['potential_realized_pct']:.1f}% потенциала — «{e['grade']['label']}».")
        expl = (f"Среднее по городу D_avg: {e['d_avg_base']:.3f} → {e['d_avg']:.3f} (70% Score). Слабейший район — "
                f"{w['district']} с баллом {w['score']:.3f} (в базе {w['base_district']}, {w['base_score']:.3f}); "
                f"он даёт 30% Score. Критических пар ниже 40: {e['n_crit']} (в базе {e['n_crit_base']}), "
                "каждая стоит −1 балл.")
        if e["critical_cells_fixed"]:
            expl += " Сняты из критических: " + _cells(e["critical_cells_fixed"]) + "."
        issues: list[Issue] = []
        if e["critical_cells"]:
            issues.append(Issue(area="districts", severity="critical",
                                issue="Остались пары ниже 40 (−1 балл каждая): " + _cells(e["critical_cells"]) + "."))
        if "plan" in e:
            targeted = {p["district"] for p in e["plan"] if p["district"] != ds.CITY}
            if w["district"] not in targeted:
                issues.append(Issue(area="districts", severity="high", issue=(
                    f"Ни одна районная мера не направлена в {w['district']} — слабейший район определяет 30% Score.")))
            for p in e["plan"]:
                if p["lag"] >= 4:
                    issues.append(Issue(area=p["sector"], severity="medium", issue=(
                        f"{_mname(p['code'])}: лаг {p['lag']} кв. — за горизонт реализуется только "
                        f"{100 * p['realized_share']:.1f}% эффекта.")))
            worst = min(e["plan"], key=lambda p: p["contribution_per_unit"])
            issues.append(Issue(area=worst["sector"], severity="medium", issue=(
                f"Минимальная отдача на единицу — {worst['code']} ({worst['district']}): вклад "
                f"{worst['contribution']:.3f} балла за {worst['cost']:g} {ds.UNIT}")))
            if e["unspent"] >= 5:
                issues.append(Issue(area="budget", severity="low",
                                    issue=f"Не использовано {e['unspent']:g} {ds.UNIT} из {e['budget_total']:g}."))
        for syn in e["synergies"]:
            if syn["status"] == "missed":
                issues.append(Issue(area="budget", severity="low",
                                    issue=f"Синергия {syn['pair']} не собрана: {syn['hint']}."))
        if not issues:
            issues.append(Issue(area="budget", severity="low", issue="Критических перекосов нет."))

        if "plan" in e:
            best = max(e["plan"], key=lambda p: p["contribution_per_unit"])
            worst = min(e["plan"], key=lambda p: p["contribution_per_unit"])
            efficiency = (f"Лучшая отдача на единицу — {best['code']} ({best['district']}): вклад {best['contribution']:.3f} "
                          f"за {best['cost']:g} {ds.UNIT}; худшая — {worst['code']} ({worst['district']}): "
                          f"{worst['contribution']:.3f} за {worst['cost']:g}. Средневзвешенная реализация эффекта "
                          f"за 8 кварталов — {e['realized_share_weighted_pct']:.1f}%: меры с лагом 4 дают лишь половину эффекта.")
        active = [s["pair"] for s in e["synergies"] if s["status"] == "active"]
        missed = [s for s in e["synergies"] if s["status"] == "missed"]
        synergy = ("Собраны синергии: " + ", ".join(active) + "." if active else "Ни одна синергия не собрана.")
        if missed:
            synergy += " Упущены: " + "; ".join(f"{s['pair']} ({s['hint']})" for s in missed) + "."
        insights = [f"Score {e['score']:.3f} против {e['reference']['score']:.3f} у ориентира "
                    f"({e['reference']['label']})."]
        if e["moves"]:
            m = e["moves"][0]
            insights.append(f"Лучший проверенный ход: {m['text']} → +{m['gain']:.3f} (Score {m['score_after']:.3f}).")
        sens = min(e["sensitivity"], key=lambda s: s["gain_change"])
        insights.append(f"Сильнее всего решение теряет в сценарии чувствительности «{sens['name']}»: прирост к базе "
                        f"{sens['gain_vs_base']:.3f} вместо {e['delta_vs_base']:.3f}.")
        top_risk = max(e["risk_proxies"].values(), key=lambda r: r["value"])
        insights.append(f"Наибольший риск-индекс (модельная оценка): {top_risk['label']} — {top_risk['value']:.1f}/100, "
                        f"хуже всего в районе {top_risk['worst_district']}.")
        return AnalyticsOutput(headline=headline, score_explanation=expl, issues=issues[:6], efficiency=efficiency,
                               synergy=synergy, key_insights=insights)


# ════════════════════════════════════════════════════════════════════════════
# Agent 2 · Astana Urban Expert
# ════════════════════════════════════════════════════════════════════════════

CITY_PROFILE = [
    "Резко континентальный климат: морозы до −35…−40 °C, метели; жаркое сухое лето; сильные степные ветры.",
    "Река Есиль делит город на берега: мосты — узкие места в часы пик.",
    "Весной — паводки и подтопления при снеготаянии; ливни перегружают ливнёвки.",
    "Частный сектор с печным отоплением даёт зимний смог; изношенные тепло- и водосети.",
    "Быстрый рост населения: дефицит мест в школах и поликлиниках в новых районах.",
]


class UrbanExpertAgent(BaseAgent):
    key = "urban_expert"
    title = "Agent 2 · Astana Urban Expert"
    role = "Оценивает решение по районам Астаны: мосты, смог частного сектора, сети ЖКХ, школы, сезонность"
    output_model = UrbanExpertOutput

    def system_prompt(self, lang: str) -> str:
        return f"""Ты — Agent 2 «Astana Urban Expert», урбанист с 15-летним опытом в Астане.
Оцени решение Акима по каждому из 5 районов (ровно 5 district_assessments) с учётом профиля районов из датасета
(district_notes) и специфики города (city_profile): зима, смог частного сектора, мосты через Есиль, паводки,
износ сетей, рост населения. Опирайся на показатели районов после мер, критические пары (<40), лаги мер
(8 кварталов горизонта) и каталог. priority_projects — только меры каталога M1–M14 с корректным районом
('город' для городских мер). Если по опыту видишь, что модель что-то недооценивает, скажи — это ценно.
{_rules(lang)}"""

    def payload(self, ctx: SimulationContext, **kw: Any) -> dict[str, Any]:
        e = ctx.engine
        return {
            **_engine_core(e), "akim_statement": ctx.decision.akim_statement,
            "districts": e["districts"], "indicators": e["indicators"], "risk_proxies": e["risk_proxies"],
            "sensitivity": e["sensitivity"], "synergies": e["synergies"],
            "city_profile": CITY_PROFILE, "district_notes": ds.DISTRICT_NOTES,
            "catalog": [{"code": m.code, "name": m.name, "sector": m.sector, "cost": m.cost, "lag": m.lag,
                         "scope": m.scope, "effects": dict(m.effects)} for m in ds.MEASURES.values()],
        }

    def postprocess(self, out: BaseModel, ctx: SimulationContext, **kw: Any) -> BaseModel:
        assert isinstance(out, UrbanExpertOutput)
        seen: dict[str, DistrictAssessment] = {}
        for a in out.district_assessments:
            seen.setdefault(a.district, a)
        if len(seen) < len(ds.DISTRICTS):
            for a in self.fallback(ctx).district_assessments:
                seen.setdefault(a.district, a)
        out.district_assessments = [seen[d] for d in ds.DISTRICTS]
        projects = []
        for p in out.priority_projects:  # отбрасываем меры с неверным масштабом/районом
            m = ds.MEASURES[p.code]
            d = ds.normalize_district(p.district)
            if m.citywide:
                projects.append(ProjectIdea(code=p.code, district=ds.CITY, rationale=p.rationale))
            elif d in ds.DISTRICTS:
                projects.append(ProjectIdea(code=p.code, district=d, rationale=p.rationale))
        out.priority_projects = projects[:5] or self.fallback(ctx).priority_projects
        out.seasonal_risks = out.seasonal_risks[:4]
        return out

    def fallback(self, ctx: SimulationContext, **kw: Any) -> UrbanExpertOutput:
        e = ctx.engine
        rp = e["risk_proxies"]
        chosen = _chosen_codes(e)
        crit = e["critical_cells"]
        assessments = []
        for d in e["districts"]:
            name = d["name"]
            crit_here = [c for c in crit if c["district"] == name]
            verdict = ("critical" if crit_here else "weak" if d["is_weakest"] or d["score"] < 55
                       else "strong" if d["delta"] >= 1.5 else "ok")
            here = [c for c, dist in chosen if dist == name]
            comment = f"{d['note'][0].upper() + d['note'][1:]}. Балл района {d['base_score']:.3f} → {d['score']:.3f} ({d['delta']:+.3f})."
            comment += (f" Районные меры: {', '.join(here)}." if here
                        else " Районных мер нет — район получает только общегородские.")
            if crit_here:
                comment += " Ниже 40: " + _cells(crit_here) + "."
            assessments.append(DistrictAssessment(district=name, verdict=verdict, comment=comment))
        ind = {i["code"]: i for i in e["indicators"]}
        seasonal = [
            SeasonalRisk(season="winter", risk=(
                f"Отопительный сезон и смог: риск-индекс аварий ЖКХ {rp['utilities_failure']['value']:.1f}/100, "
                f"смога {rp['winter_smog']['value']:.1f}/100 (модельные оценки); хуже всего — "
                + " и ".join(dict.fromkeys([rp['utilities_failure']['worst_district'], rp['winter_smog']['worst_district']]))
                + "."),
                impact="Аварии на сетях в морозы и смог от печного отопления частного сектора."),
            SeasonalRisk(season="spring", risk=(
                f"Паводок и ливни нагружают сети и службы реагирования: риск-индекс накопления обращений "
                f"{rp['service_backlog']['value']:.1f}/100 (модельная оценка)."),
                impact="Подтопления низин и дворов, очереди заявок жителей."),
            SeasonalRisk(season="summer", risk=(
                f"Жара и пыль: озеленение (E1) хуже всего в районе {ind['E1']['worst_district']} — "
                f"{ind['E1']['worst_value']:.2f}."), impact="Перегрев дворов, нехватка тени и мест для прогулок."),
            SeasonalRisk(season="autumn", risk=(
                f"Подготовка к зиме: надёжность ЖКХ (C1) хуже всего в районе {ind['C1']['worst_district']} — "
                f"{ind['C1']['worst_value']:.2f}."), impact="От осенних работ зависит, пройдёт ли зима без аварий."),
        ]
        projects: list[ProjectIdea] = []
        chosen_set = set(chosen)
        ref_plan = e["reference"].get("plan") or []
        if not ref_plan and e["mode"] == "allocation":
            from .optimizer import best_measure_plans
            best, _ = best_measure_plans(top=1)
            ref_plan = [{"code": ch.code, "district": ch.district} for ch in best[0].choices]
        for p in ref_plan:
            if (p["code"], p["district"]) not in chosen_set:
                m = ds.MEASURES[p["code"]]
                note = ds.DISTRICT_NOTES.get(p["district"], "эффект во всех районах")
                effects = ", ".join(f"{k} +{v * m.realized_share:.2f}" for k, v in m.effects.items() if v > 0)
                projects.append(ProjectIdea(code=p["code"], district=p["district"],
                                            rationale=f"{effects} с учётом лага; {note}."))
        for syn in e["synergies"]:
            if syn["status"] == "missed" and len(projects) < 4:
                code = syn["pair"].split("+")[1] if syn["anchor_districts"] else syn["pair"].split("+")[0]
                m = ds.MEASURES[code]
                if not any(pp.code == code for pp in projects):
                    dist = ds.CITY if m.citywide else (syn["anchor_districts"] or [e["weakest"]["district"]])[0]
                    projects.append(ProjectIdea(code=code, district=dist, rationale=f"Собирает синергию {syn['pair']}: {syn['hint']}."))
        if not projects:
            projects.append(ProjectIdea(code="M14", district=ds.CITY,
                                        rationale="Аварийные бригады ЖКХ страхуют город зимой и работают с лагом 1."))
        w = e["weakest"]
        headline = (f"Главный рычаг — {w['district']} ({ds.DISTRICT_NOTES[w['district']]}): слабейший район даёт 30% Score"
                    + (f", а пар ниже 40 осталось {e['n_crit']}." if e["n_crit"] else ", и критических пар ниже 40 не осталось."))
        insight = ("В официальной формуле 30% Score — это слабейший район, а каждая пара ниже 40 стоит целый балл. "
                   "Поэтому для Астаны важнее адресно закрыть провалы Нуры (школы, поликлиники, транспорт) и смог "
                   "Сарыарки, чем равномерно улучшать благополучный Есиль. Долгие проекты (ЛРТ, сети, лаг 4) "
                   "стратегически нужны городу, но за 8 кварталов дают лишь половину эффекта.")
        return UrbanExpertOutput(headline=headline, district_assessments=assessments, seasonal_risks=seasonal,
                                 priority_projects=projects[:4], local_insight=insight)


# ════════════════════════════════════════════════════════════════════════════
# Agent 3 · Social & Risk
# ════════════════════════════════════════════════════════════════════════════

_POSITIVE_POSTS: dict[str, tuple[str, str, str]] = {
    "M1": ("Мадина, IT-специалист", "Threads", "Выделенные полосы для автобусов реально работают: до работы теперь быстрее, чем на машине."),
    "M2": ("Ерлан, таксист", "Telegram", "Умные светофоры — меньше стоим на перекрёстках, особенно у мостов."),
    "M3": ("Мадина, IT-специалист", "Threads", "Ветку ЛРТ ведут к нашему району — ждём как праздника."),
    "M4": ("Айгуль, мама в декрете", "Instagram", "Новый сквер у дома — тень, лавочки, дорожки для колясок."),
    "M5": ("Данияр, житель частного сектора", "Instagram", "Соседи перешли на чистое топливо — зимой вечером наконец можно открыть окно."),
    "M6": ("Асель, эколог-волонтёр", "Instagram", "Ветрозащитные полосы и новые деревья по всему городу — это надолго."),
    "M7": ("Айгерим, мама двоих школьников", "Instagram", "Строят школу с детсадом рядом с домом — наконец-то без второй смены!"),
    "M8": ("Гульнара, медсестра", "Telegram", "Центр семейного здоровья рядом — не нужно ехать через весь город."),
    "M9": ("Арман, тренер", "Instagram", "Во дворах появились спорт-хабы: подростки теперь во дворе, а не в подъезде."),
    "M10": ("Жанна, фитнес-тренер", "Instagram", "Во дворах поставили свет и камеры — вечерние пробежки снова в радость."),
    "M11": ("Сауле, учитель", "Facebook", "У школы сделали безопасный переход — спокойнее за детей."),
    "M12": ("Нурлан, пенсионер", "Telegram", "Подал заявку через городскую платформу — закрыли быстро и с фотоотчётом."),
    "M13": ("Нурлан, пенсионер", "Telegram", "После замены труб батареи горячие, а воду больше не отключают."),
    "M14": ("Бауыржан, инженер", "Telegram", "Аварийная бригада приехала быстро, жителей предупредили заранее."),
}
_NEGATIVE_POSTS: dict[str, tuple[str, str, str]] = {
    "S1": ("Айгерим, мама двоих школьников", "Telegram", "Сын учится во вторую смену, в садик очередь. Район растёт, а школа одна."),
    "S2": ("Гульнара, медсестра", "Telegram", "К терапевту запись на недели вперёд, поликлиника одна на весь район."),
    "T1": ("Ерлан, таксист", "Telegram", "Мосты в час пик стоят намертво — полтора часа с берега на берег."),
    "T2": ("Дана, студентка", "Instagram", "До остановки идти далеко, автобус раз в полчаса. Без машины тут никак."),
    "E1": ("Айгуль, мама в декрете", "Instagram", "Во дворе ни одного дерева: летом асфальт плавится, зимой ветер сбивает с ног."),
    "E2": ("Данияр, житель частного сектора", "Instagram", "Зимой вечером не открыть окно — весь квартал топит углём."),
    "B1": ("Сергей, пенсионер", "Facebook", "Во дворе темно с шести вечера, фонари не горят. Страшно выходить."),
    "B2": ("Сауле, учитель", "Facebook", "У школы нет нормального перехода — каждый день переживаю за детей."),
    "C1": ("Нурлан, пенсионер", "Telegram", "Опять отключили горячую воду, батареи еле тёплые — сети старые."),
    "C2": ("Асель, экономист", "Threads", "Заявку в ЖКХ подала месяц назад — тишина."),
}
_MITIGATION = {
    "utilities_failure": "M13 (модернизация сетей) в худшем районе и M14 (аварийные бригады) на весь город.",
    "traffic_gridlock": "M1 или M3 в районе с худшим T1 и M2 (умные светофоры) на весь город.",
    "winter_smog": "M5 (чистое топливо) в районе со смогом, вместе с M6 для синергии E2.",
    "road_accidents": "M11 (безопасные переходы) и M2 (умные светофоры).",
    "service_backlog": "M12 (единая платформа обращений) и M14.",
    "social_tension": "Адресные меры в слабейшем районе и открытая отчётность по кварталам.",
}


class SocialRiskAgent(BaseAgent):
    key = "social_risk"
    title = "Agent 3 · Social & Risk"
    role = "Моделирует реакцию жителей (вымышленные посты) и риски по модельным прокси"
    output_model = SocialRiskOutput

    def system_prompt(self, lang: str) -> str:
        return f"""Ты — Agent 3 «Social & Risk», аналитик общественных настроений и городских рисков Астаны.
1) Смоделируй реакцию горожан: 4–5 постов ВЫМЫШЛЕННЫХ жителей (имя, роль, район) — это симуляция для игры,
   данных соцсетей нет. Реакции должны соответствовать мерам решения и провалам районов.
2) Риски бери из risk_proxies: это модельные оценки 0–100 (прокси), а не вероятности. Уровень level — из прокси.
   В названии каждого риска пиши «модельная оценка». Меры снижения — из каталога.
3) Уязвимые группы — по худшим показателям районов; совет по коммуникации для Акима.
{_rules(lang)}"""

    def payload(self, ctx: SimulationContext, **kw: Any) -> dict[str, Any]:
        e = ctx.engine
        return {**_engine_core(e), "akim_statement": ctx.decision.akim_statement, "districts": e["districts"],
                "risk_proxies": e["risk_proxies"], "disclaimers": e["disclaimers"]}

    def postprocess(self, out: BaseModel, ctx: SimulationContext, **kw: Any) -> BaseModel:
        assert isinstance(out, SocialRiskOutput)
        out.simulated_posts = out.simulated_posts[:6]
        out.risks = out.risks[:6]
        for r in out.risks:  # гарантируем явную пометку прокси
            if "модельн" not in r.name.lower() and "proxy" not in r.name.lower() and "прокси" not in r.name.lower():
                r.name = f"{r.name} (модельная оценка)"
        out.vulnerable_groups = out.vulnerable_groups[:5]
        return out

    def fallback(self, ctx: SimulationContext, **kw: Any) -> SocialRiskOutput:
        e = ctx.engine
        rp = e["risk_proxies"]
        disc = rp["public_discontent"]
        w = e["weakest"]["district"]
        mood = ("Раздражение заметно" if disc["level"] == "high" else "Сдержанный оптимизм" if disc["level"] == "medium"
                else "Горожане в целом довольны")
        public_mood = (f"{mood}: индекс недовольства {disc['value']:.1f}/100, риск-индекс социальной напряжённости "
                       f"{rp['social_tension']['value']:.1f}/100 (модельные оценки, не данные соцсетей). "
                       f"Больше всего недовольных — в районе {w}.")
        chosen = _chosen_codes(e)
        posts: list[SocialPost] = []
        # негатив — самые низкие показатели районов после мер; позитив — реализованные меры
        values = e["district_indicators"]
        lows = sorted(((dist, k, v) for dist in ds.DISTRICTS for k, v in values[dist].items()), key=lambda t: t[2])
        used_k: set[str] = set()
        for dist, k, v in lows:
            if len(posts) >= 2:
                break
            if k in used_k or k not in _NEGATIVE_POSTS or v >= 50:
                continue
            persona, platform, text = _NEGATIVE_POSTS[k]
            posts.append(SocialPost(persona=f"{persona}, {dist}", platform=platform, sentiment="negative", text=text))
            used_k.add(k)
        used_codes: set[str] = set()
        for code, dist in chosen:
            if len(posts) >= 5:
                break
            if code in _POSITIVE_POSTS and code not in used_codes:
                persona, platform, text = _POSITIVE_POSTS[code]
                where = "" if dist == ds.CITY else f", {dist}"
                posts.append(SocialPost(persona=f"{persona}{where}", platform=platform, sentiment="positive", text=text))
                used_codes.add(code)
        risks = []
        for key in ("utilities_failure", "traffic_gridlock", "winter_smog", "road_accidents", "social_tension"):
            r = rp[key]
            risks.append(RiskItem(name=f"{r['label']} (модельная оценка)", level=r["level"],
                                  trigger=f"Индекс {r['value']:.1f}/100 (прокси, не вероятность); хуже всего — {r['worst_district']}",
                                  mitigation=_MITIGATION[key]))
        groups_map = {
            "S1": "семьи с детьми в районе {d}: нехватка мест в школах и детсадах",
            "S2": "пожилые и хронические пациенты в районе {d}: далеко до первичной помощи",
            "T2": "жители района {d} без удобного общественного транспорта",
            "T1": "те, кто ежедневно ездит через мосты из района {d}",
            "E2": "жители частного сектора района {d}: зимний смог",
            "E1": "дети и пожилые в районе {d}: мало зелени и мест для прогулок",
            "C1": "жители старого жилфонда района {d}: изношенные сети",
            "C2": "жители района {d}: долгое закрытие обращений",
            "B1": "жители неосвещённых дворов района {d}",
            "B2": "школьники района {d}: опасные переходы",
        }
        groups: list[str] = []
        for dist, k, _ in lows:
            g = groups_map[k].format(d=dist)
            if g not in groups:
                groups.append(g)
            if len(groups) >= 3:
                break
        advice = (f"Начать с признания проблем района {w} и показать адресный план по кварталам с учётом лагов мер: "
                  "что заработает уже через квартал, а что — через год. Публиковать открытый отчёт о ходе работ.")
        headline = (f"Настроения: {mood.lower()}; главный источник недовольства — район {w} "
                    f"(индекс недовольства {disc['value']:.1f}/100, модельная оценка).")
        return SocialRiskOutput(headline=headline, public_mood=public_mood, simulated_posts=posts[:5], risks=risks,
                                vulnerable_groups=groups, communication_advice=advice)


# ════════════════════════════════════════════════════════════════════════════
# Orchestrator · Akim Chief Advisor
# ════════════════════════════════════════════════════════════════════════════


class OrchestratorAgent(BaseAgent):
    key = "orchestrator"
    title = "Orchestrator · Akim Chief Advisor"
    role = "Сводит выводы агентов, сравнивает наборы, разрешает противоречия и даёт рекомендации"
    output_model = OrchestratorOutput

    def model_for(self, llm: LLMClient) -> str:
        return llm.orchestrator_model

    def system_prompt(self, lang: str) -> str:
        return f"""Ты — «Akim Chief Advisor», главный советник Акима Астаны и оркестратор мультиагентной системы.
На входе — отчёт движка и выводы трёх агентов. Синтезируй финальный отчёт:
- executive_summary: Score, доля реализованного потенциала, слабейший район, главный резерв;
- set_comparison: сравни решение с ориентиром (reference) и, в режиме мер, с эталонным примером датасета (comparison);
- consensus и disagreements (где агенты расходятся, например адресность против охвата всего города, долгие проекты
  против быстрого эффекта) и как ты их разрешаешь;
- recommendations: 3–5 действий; эффект в баллах указывай ТОЛЬКО из moves (ходы проверены движком);
- roadmap: шаги по кварталам с учётом лагов мер (горизонт 8 кварталов);
- тон уважительный и прямой, без лести.
{_rules(lang)}"""

    def payload(self, ctx: SimulationContext, **kw: Any) -> dict[str, Any]:
        e = ctx.engine
        results: dict[str, AgentResult] = kw["results"]
        return {"akim_name": ctx.decision.akim_name, "akim_statement": ctx.decision.akim_statement,
                "engine": {**_engine_core(e), "moves": e["moves"], "synergies": e["synergies"],
                           "risk_proxies": e["risk_proxies"], "sensitivity": e["sensitivity"]},
                "agents": {k: r.output.model_dump() for k, r in results.items()}}

    def postprocess(self, out: BaseModel, ctx: SimulationContext, **kw: Any) -> BaseModel:
        assert isinstance(out, OrchestratorOutput)
        recs = sorted(out.recommendations, key=lambda r: r.priority)[:6]
        for i, r in enumerate(recs, 1):
            r.priority = i
        out.recommendations = recs
        out.consensus = out.consensus[:5]
        out.disagreements = out.disagreements[:3]
        out.roadmap = out.roadmap[:7]
        return out

    def fallback(self, ctx: SimulationContext, **kw: Any) -> OrchestratorOutput:
        e = ctx.engine
        results: dict[str, AgentResult] = kw.get("results", {})
        w = e["weakest"]
        ref = e["reference"]
        summary = (f"Ваше решение даёт Score {e['score']:.3f} против базы {e['base_score']:.3f} "
                   f"(+{e['delta_vs_base']:.3f}); это {e['potential_realized_pct']:.1f}% потенциала — "
                   f"оценка {e['grade']['grade']} «{e['grade']['label']}»; {ref['short_label']} — {ref['score']:.3f}. "
                   f"Слабейший район — {w['district']} ({w['score']:.3f}); критических пар ниже 40: {e['n_crit']}.")
        if e["moves"]:
            m = e["moves"][0]
            summary += f" Главный резерв: {m['text'].lower()[0] + m['text'][1:]} → +{m['gain']:.3f}."
        pct = e["potential_realized_pct"]
        verdict = ("Решение близко к оптимальному — остаётся точная настройка." if pct >= 90
                   else "Рабочее решение с понятным резервом роста." if pct >= 75
                   else "Решение работает, но заметная часть бюджета тратится неэффективно." if pct >= 50
                   else "Решение требует пересмотра: бюджет почти не улучшает город.")
        if "plan" in e:
            cmp = {c["label"]: c for c in e["comparison"]}
            mine, best, ex = cmp["Ваш набор"], cmp["Официальный оптимум"], cmp["Эталонный пример датасета"]
            comparison = (("При заданных пяти лимитах выбран лучший допустимый набор. "
                           if e["mode"] == "allocation" else "") + f"Ваш набор ({mine['plan_label']}) — {mine['score']:.3f} за {mine['cost']:g} {ds.UNIT}; "
                          f"Официальный оптимум ({best['plan_label']}) — {best['score']:.3f} за {best['cost']:g}. "
                          f"Эталонный пример датасета — {ex['score']:.3f}. Разрыв с оптимумом — "
                          f"{e['gap_to_reference']:.3f} балла.")
        consensus = [f"Все агенты сходятся: главный рычаг — {w['district']}, слабейший район даёт 30% Score."]
        if e["critical_cells"]:
            consensus.append("Критические пары ниже 40 нужно закрыть в первую очередь: " + _cells(e["critical_cells"]) + ".")
        elif e["critical_cells_fixed"]:
            consensus.append("Критические пары базы закрыты: " + _cells(e["critical_cells_fixed"]) + ".")
        if e["moves"]:
            mv = e["moves"][0]["text"]
            consensus.append(f"Резерв без роста бюджета: {mv[0].lower() + mv[1:]} (+{e['moves'][0]['gain']:.3f}).")
        consensus.append("Долгие проекты с лагом 4 дают за 8 кварталов лишь половину эффекта — их стоит сочетать с быстрыми мерами.")
        disagreements: list[Disagreement] = []
        chosen = _chosen_codes(e)
        lag4 = [c for c, _ in chosen if ds.MEASURES[c].lag >= 4]
        if lag4:
            disagreements.append(Disagreement(
                topic="Долгий проект против быстрого эффекта",
                positions=(f"Urban Expert: {', '.join(sorted(set(lag4)))} стратегически нужны Астане. Analytics: за 8 "
                           "кварталов такие меры реализуются лишь на 50%."),
                resolution="Оставить один долгий проект в слабейшем районе и добавить быстрые меры с лагом 1."))
        citywide = [c for c, d in chosen if d == ds.CITY]
        if citywide:
            disagreements.append(Disagreement(
                topic="Охват всего города против адресности",
                positions=(f"Social & Risk: городские меры ({', '.join(sorted(set(citywide)))}) заметны всем жителям. "
                           f"Analytics: формула сильнее вознаграждает рост слабейшего района ({w['district']})."),
                resolution="Одна недорогая городская мера плюс адресный пакет для слабейшего района."))
        if not disagreements:
            disagreements.append(Disagreement(
                topic="Смог Сарыарки против соцсферы Нуры",
                positions="Urban Expert: зимний смог — самая заметная боль. Analytics: провалы Нуры сильнее бьют по Score.",
                resolution="Сначала закрыть пары ниже 40 в Нуре, смог — следующим шагом с синергией M5+M6."))
        recs: list[Recommendation] = []
        for m in e["moves"]:
            area = (ds.MEASURES[m["detail"]["with"]["code"]].sector if m["kind"] == "swap" else m["to"])
            recs.append(Recommendation(priority=len(recs) + 1, area=area, action=m["text"],
                                       rationale="Ход проверен движком на том же бюджете и правилах датасета.",
                                       expected_effect=f"+{m['gain']:.3f} балла, Score после шага {m['score_after']:.3f}"))
        urban = results.get("urban_expert")
        if urban is not None:
            uo = urban.output
            assert isinstance(uo, UrbanExpertOutput)
            for p in uo.priority_projects:
                if len(recs) >= 5:
                    break
                if not any(p.code in r.action for r in recs):
                    recs.append(Recommendation(priority=len(recs) + 1, area=ds.MEASURES[p.code].sector,
                                               action=f"Рассмотреть {_mname(p.code)} {_where(p.district)}",
                                               rationale=p.rationale,
                                               expected_effect="Качественная оценка Urban Expert; эффект проверить в движке"))
        if not recs:
            recs.append(Recommendation(priority=1, area="budget", action="Сохранить решение и сфокусироваться на исполнении",
                                       rationale="Решение совпадает с ориентиром движка.",
                                       expected_effect=f"Удержание Score {e['score']:.3f}"))
        roadmap = _roadmap(e)
        who = ctx.decision.akim_name or "Господин Аким"
        closing = (f"{who}, в этой модели город оценивается по самому слабому району. Закройте провалы "
                   f"района {w['district']} — и средний балл потянется следом.")
        return OrchestratorOutput(title=f"Отчёт главного советника: Score {e['score']:.3f} — {e['grade']['label']}",
                                  executive_summary=summary, verdict=verdict, set_comparison=comparison,
                                  consensus=consensus[:4], disagreements=disagreements[:3], recommendations=recs[:5],
                                  roadmap=roadmap, closing_statement=closing)


def _roadmap(e: dict[str, Any]) -> list[str]:
    chosen = _chosen_codes(e)
    by_lag: dict[int, list[str]] = {}
    for code, dist in chosen:
        by_lag.setdefault(ds.MEASURES[code].lag, []).append(f"{code} ({dist})")
    steps = ["Квартал 1: утвердить решение, KPI по районам и открытый отчёт о расходах."]
    for lag in sorted(by_lag):
        steps.append(f"Лаг {lag} кв.: {', '.join(by_lag[lag])} — эффект с квартала {lag + 1}, "
                     f"за горизонт реализуется {100 * (ds.HORIZON_QUARTERS - lag) / ds.HORIZON_QUARTERS:.1f}%.")
    steps.append("Каждый квартал: замер показателей районов и сверка с прогнозом движка.")
    return steps[:6]
