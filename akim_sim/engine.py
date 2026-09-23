"""Детерминированный движок: собирает единый отчёт по решению Акима для агентов, UI и API.

Все числа отчёта считает этот модуль. LLM-агенты их только объясняют.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import dataset as ds
from .inputs import AllocationDecision, Decision, MeasuresDecision, PlanValidationError
from .model import Choice, Evaluation, Placement, base_evaluation, evaluate_plan, plan_fractions
from .optimizer import best_measure_plans, suggest_swaps
from .risk import DISCLAIMERS, risk_proxies, sensitivity

FORMULA = ("Score = 0.7·D_avg + 0.3·min_d D_d − 1.0·N_crit;  D_d = Σ_k w_k·I'_{d,k};  "
           "I'_{d,k} = clip(I_{d,k} + Σ_m эффект_{m,k}·(8 − L_m)/8 + синергии, 0, 100)")


def grade_for(pct: float) -> dict[str, str]:
    """Оценка по доле реализованного потенциала: (Score − база) / (ориентир − база)."""
    if pct >= 99.0:
        return {"grade": "S", "label": "Оптимальное решение"}
    if pct >= 90.0:
        return {"grade": "A", "label": "Сильный управленец"}
    if pct >= 75.0:
        return {"grade": "B", "label": "Уверенный хозяйственник"}
    if pct >= 50.0:
        return {"grade": "C", "label": "Есть заметные резервы"}
    if pct >= 25.0:
        return {"grade": "D", "label": "Слабый эффект бюджета"}
    return {"grade": "F", "label": "Бюджет почти не работает"}


def _choice_dict(ch: Choice) -> dict[str, Any]:
    m = ch.measure
    return {
        "code": m.code, "name": m.name, "sector": m.sector, "sector_name": ds.SECTOR_NAMES[m.sector],
        "district": ch.district, "scope": m.scope, "cost": m.cost, "lag": m.lag,
        "realized_share": m.realized_share,
        "effects_full": dict(m.effects),
        "effects_realized": {k: round(v * m.realized_share, 3) for k, v in m.effects.items()},
    }


def _plan_label(choices: tuple[Choice, ...] | list[Choice]) -> str:
    return ", ".join(ch.label() for ch in choices)


# ── общие блоки отчёта ──────────────────────────────────────────────────────


def _common(ev: Evaluation, fractions: Mapping[Placement, float]) -> dict[str, Any]:
    base = base_evaluation()
    districts = [{
        "name": d, "population": ds.POPULATION[d], "note": ds.DISTRICT_NOTES[d],
        "base_score": round(base.district_scores[d], 3), "score": round(ev.district_scores[d], 3),
        "delta": round(ev.district_scores[d] - base.district_scores[d], 3),
        "is_weakest": d == ev.weakest_district,
    } for d in ds.DISTRICTS]
    indicators = []
    for k in ds.INDICATORS:
        b = sum(ds.POPULATION[d] * base.indicators[d][k] for d in ds.DISTRICTS)
        v = sum(ds.POPULATION[d] * ev.indicators[d][k] for d in ds.DISTRICTS)
        worst = min(ds.DISTRICTS, key=lambda d: ev.indicators[d][k])
        indicators.append({
            "code": k, "name": ds.INDICATOR_NAMES[k], "sector": ds.INDICATOR_SECTOR[k], "weight": ds.WEIGHTS[k],
            "base_city": round(b, 2), "city": round(v, 2), "delta": round(v - b, 2),
            "worst_district": worst, "worst_value": round(ev.indicators[worst][k], 2),
        })
    crit = [{"district": d, "indicator": k, "name": ds.INDICATOR_NAMES[k], "value": round(v, 2)}
            for d, k, v in ev.critical_cells]
    crit_base = [{"district": d, "indicator": k, "name": ds.INDICATOR_NAMES[k], "value": round(v, 2)}
                 for d, k, v in base.critical_cells]
    fixed = [c for c in crit_base if (c["district"], c["indicator"]) not in {(x["district"], x["indicator"]) for x in crit}]
    synergies = []
    present = {c for (c, _), f in fractions.items() if f > 1e-9}
    for syn in ds.SYNERGIES:
        fa = sum(f for (c, _), f in fractions.items() if c == syn.a)
        fb = sum(f for (c, _), f in fractions.items() if c == syn.b)
        status = "active" if fa > 1e-9 and fb > 1e-9 else "missed" if (fa > 1e-9) != (fb > 1e-9) else "unused"
        anchor = [d for (c, d), f in fractions.items() if c == syn.anchor and f > 1e-9]
        hint = ""
        if status == "missed":
            lacking = syn.b if fa > 1e-9 else syn.a
            blockers = [a if b == lacking else b for a, b, _ in ds.INCOMPATIBLE_GLOBAL
                        if lacking in (a, b) and (a if b == lacking else b) in present]
            if blockers:
                status = "blocked"
                hint = f"{lacking} несовместима с выбранной {blockers[0]} — синергию собрать нельзя"
            else:
                hint = f"добавьте {lacking}, чтобы получить +{syn.bonus:g} к {syn.indicator}"
        synergies.append({
            "pair": f"{syn.a}+{syn.b}", "indicator": syn.indicator, "bonus": syn.bonus, "status": status,
            "anchor_districts": anchor, "hint": hint,
        })
    return {
        "base_score": round(base.score, 3),
        "d_avg": round(ev.d_avg, 3), "d_avg_base": round(base.d_avg, 3),
        "weakest": {"district": ev.weakest_district, "score": round(ev.weakest_score, 3),
                    "base_district": base.weakest_district, "base_score": round(base.weakest_score, 3)},
        "n_crit": ev.n_crit, "n_crit_base": base.n_crit,
        "critical_cells": crit, "critical_cells_base": crit_base, "critical_cells_fixed": fixed,
        "districts": districts, "indicators": indicators, "synergies": synergies,
        "district_indicators": {d: {k: round(ev.indicators[d][k], 3) for k in ds.INDICATORS} for d in ds.DISTRICTS},
        "risk_proxies": risk_proxies(ev), "sensitivity": sensitivity(fractions),
        "disclaimers": DISCLAIMERS, "formula": FORMULA,
    }


# ── режим мероприятий ───────────────────────────────────────────────────────


def analyze_measures(decision: MeasuresDecision) -> dict[str, Any]:
    choices = decision.choices()  # PlanValidationError, если набор невалиден
    ev = evaluate_plan(choices)
    top, n_valid = best_measure_plans(top=5, budget=decision.budget_total)
    best = top[0]
    common = _common(ev, plan_fractions(choices))
    base_score = common["base_score"]
    potential = best.score - base_score
    pct = 100.0 * (ev.score - base_score) / potential if potential > 1e-9 else 100.0
    example = evaluate_plan([Choice(c, d) for c, d in ds.WORKED_EXAMPLE])
    cost = sum(ch.measure.cost for ch in choices)
    plan_rows = []
    for i, ch in enumerate(choices):  # вклад меры: насколько упадёт Score, если её убрать (leave-one-out)
        contribution = ev.score - evaluate_plan(choices[:i] + choices[i + 1:]).score
        row = _choice_dict(ch)
        row["contribution"] = round(contribution, 3)
        row["contribution_per_unit"] = round(contribution / ch.measure.cost, 4)
        plan_rows.append(row)
    swaps = suggest_swaps(choices, top=3, budget=decision.budget_total)
    moves = [{
        "kind": "swap",
        "text": (f"Заменить {s['replace']['code']} ({s['replace']['district']}) на "
                 f"{s['with']['code']} ({s['with']['district']})"),
        "gain": s["gain"], "score_after": s["score_after"], "detail": s,
    } for s in swaps]
    per_sector: dict[str, dict[str, Any]] = {s: {"key": s, "name": ds.SECTOR_NAMES[s], "cost": 0.0, "measures": []}
                                              for s in ds.SECTORS}
    for ch in choices:
        per_sector[ch.measure.sector]["cost"] += ch.measure.cost
        per_sector[ch.measure.sector]["measures"].append(ch.code)
    sectors = []
    for s in ds.SECTORS:
        row = per_sector[s]
        row["share_pct"] = round(100.0 * row["cost"] / cost, 1) if cost else 0.0
        row["count"] = len(row["measures"])
        sectors.append(row)
    return {
        "mode": "measures", "valid": True,
        "budget_total": decision.budget_total, "cost_total": cost, "unspent": round(decision.budget_total - cost, 2),
        "score": round(ev.score, 3), "delta_vs_base": round(ev.score - base_score, 3),
        "reference": {
            "kind": "exact_optimum", "label": f"Официальный оптимум (полный перебор {n_valid} валидных наборов)",
            "short_label": "официальный оптимум",
            "score": round(best.score, 3), "cost": best.cost,
            "plan": [_choice_dict(ch) for ch in best.choices], "plan_label": _plan_label(best.choices),
            "valid_plans": n_valid,
        },
        "potential_realized_pct": round(pct, 1),
        "gap_to_reference": round(best.score - ev.score, 3),
        "grade": grade_for(pct),
        "realized_share_weighted_pct": round(100.0 * sum(ch.measure.realized_share * ch.measure.cost
                                                           for ch in choices) / cost, 1) if cost else 0.0,
        "plan": plan_rows,
        "sectors": sectors,
        "moves": moves,
        "top_plans": [{"score": round(p.score, 3), "cost": p.cost, "plan_label": _plan_label(p.choices)} for p in top],
        "comparison": [
            {"label": "Ваш набор", "score": round(ev.score, 3), "cost": cost, "weakest": ev.weakest_district,
             "n_crit": ev.n_crit, "plan_label": _plan_label(choices)},
            {"label": "Официальный оптимум", "score": round(best.score, 3), "cost": best.cost,
             "weakest": evaluate_plan(list(best.choices)).weakest_district,
             "n_crit": evaluate_plan(list(best.choices)).n_crit, "plan_label": _plan_label(best.choices)},
            {"label": "Эталонный пример датасета", "score": round(example.score, 3),
             "cost": sum(ds.MEASURES[c].cost for c, _ in ds.WORKED_EXAMPLE), "weakest": example.weakest_district,
             "n_crit": example.n_crit, "plan_label": ", ".join(f"{c} ({d})" for c, d in ds.WORKED_EXAMPLE)},
        ],
        **common,
    }


# ── режим пяти бюджетных лимитов ───────────────────────────────────────────
def analyze_allocation(decision: AllocationDecision) -> dict[str, Any]:
    caps = dict(decision.allocations)
    ranked, checked = best_measure_plans(top=1, budget=decision.budget_total, caps=caps)
    if not ranked:
        raise PlanValidationError(["Эти лимиты по сферам не позволяют выбрать ровно 5 совместимых мер из каталога. "
                                   "Увеличьте лимиты хотя бы в одной другой сфере."])
    choices = list(ranked[0].choices)
    result = analyze_measures(MeasuresDecision(
        measures=[{"code": ch.code, "district": ch.district} for ch in choices],
        budget_total=decision.budget_total, akim_name=decision.akim_name,
        akim_statement=decision.akim_statement))
    spent = {s: sum(ch.measure.cost for ch in choices if ch.measure.sector == s) for s in ds.SECTORS}
    reference = {s: sum(p["cost"] for p in result["reference"]["plan"] if p["sector"] == s)
                 for s in ds.SECTORS}
    result.update({
        "mode": "allocation", "allocations": caps,
        "allocated_total": round(sum(caps.values()), 2),
        "unallocated": round(decision.budget_total - sum(caps.values()), 2),
        "feasible_portfolios_checked": checked,
        "sectors": [{"key": s, "name": ds.SECTOR_NAMES[s], "cap": caps[s], "spent": spent[s],
                     "unused_cap": round(caps[s] - spent[s], 2), "global_optimum_spent": reference[s]}
                    for s in ds.SECTORS],
        "moves": [],  # максимум в пределах этих лимитов; замены из режима мер могли бы нарушить их
        "selection_policy": "Пять сумм — верхние лимиты; выбран лучший допустимый набор ровно из пяти целых мер.",
    })
    return result


def analyze(decision: Decision) -> dict[str, Any]:
    if isinstance(decision, MeasuresDecision):
        return analyze_measures(decision)
    return analyze_allocation(decision)


def config() -> dict[str, Any]:
    """Датасет и правила для UI/API."""
    return {
        "dataset": ds.DATASET_NAME, "unit": ds.UNIT, "budget": ds.BUDGET, "n_decisions": ds.N_DECISIONS,
        "max_per_direction": ds.MAX_PER_DIRECTION, "horizon_quarters": ds.HORIZON_QUARTERS,
        "crit_threshold": ds.CRIT_THRESHOLD, "formula": FORMULA,
        "sectors": [{"key": s, "name": ds.SECTOR_NAMES[s]} for s in ds.SECTORS],
        "indicators": [{"code": k, "name": ds.INDICATOR_NAMES[k], "sector": ds.INDICATOR_SECTOR[k],
                        "weight": ds.WEIGHTS[k]} for k in ds.INDICATORS],
        "districts": [{"name": d, "population": ds.POPULATION[d], "note": ds.DISTRICT_NOTES[d],
                       "baseline": ds.BASELINE[d]} for d in ds.DISTRICTS],
        "measures": [{"code": m.code, "name": m.name, "sector": m.sector, "cost": m.cost, "lag": m.lag,
                      "scope": m.scope, "realized_share": m.realized_share, "effects": dict(m.effects)}
                     for m in ds.MEASURES.values()],
        "synergies": [{"pair": f"{s.a}+{s.b}", "anchor": s.anchor, "indicator": s.indicator, "bonus": s.bonus}
                      for s in ds.SYNERGIES],
        "incompatible": [{"pair": f"{a}+{b}", "scope": "global", "reason": r} for a, b, r in ds.INCOMPATIBLE_GLOBAL]
        + [{"pair": f"{a}+{b}", "scope": "same_district", "reason": r} for a, b, r in ds.INCOMPATIBLE_SAME_DISTRICT],
        "worked_example": [{"code": c, "district": d} for c, d in ds.WORKED_EXAMPLE],
        "disclaimers": DISCLAIMERS,
    }
