"""Модельные прокси рисков и анализ чувствительности.

ВАЖНО: всё в этом модуле — модельные оценки (прокси), а не вероятности. Индексы вычисляются по формулам
из показателей датасета и НЕ откалиброваны на эмпирических данных (статистике аварий, ДТП, обращений,
социологии). Коэффициенты формул ниже — допущения проекта, а не данные организаторов.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import dataset as ds
from .model import Evaluation, Placement, base_evaluation, evaluate_fractions, normalized_weights

RISK_DISCLAIMER = (
    "Модельные оценки (прокси): индексы риска 0–100 рассчитаны по формулам из показателей датасета. "
    "Это не вероятности и не прогноз событий; они не откалиброваны на эмпирических данных об авариях, "
    "заторах, ДТП, обращениях граждан или протестах."
)
SENSITIVITY_DISCLAIMER = (
    "Анализ чувствительности: в той же официальной формуле меняются только веса показателей. "
    "Сами события (буран, паводок, миграционный рост) не моделируются."
)
SOCIAL_DISCLAIMER = (
    "Реакции жителей — симуляция вымышленных персонажей для игры; данные соцсетей не подключены."
)
DISCLAIMERS = [RISK_DISCLAIMER, SENSITIVITY_DISCLAIMER, SOCIAL_DISCLAIMER]

# (ключ, подпись, показатель): риск = 100 − (0.5·среднее по населению + 0.5·минимум по районам)
_EXPOSURE_PROXIES = (
    ("utilities_failure", "Риск аварий тепло- и водоснабжения", "C1"),
    ("traffic_gridlock", "Риск транспортного коллапса в час пик", "T1"),
    ("winter_smog", "Риск эпизодов смога зимой", "E2"),
    ("road_accidents", "Риск ДТП с пострадавшими", "B2"),
    ("service_backlog", "Риск накопления необработанных обращений", "C2"),
)


def _level(value: float) -> str:
    return "high" if value >= 50 else "medium" if value >= 35 else "low"


def risk_proxies(ev: Evaluation) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, label, k in _EXPOSURE_PROXIES:
        avg = sum(ds.POPULATION[d] * ev.indicators[d][k] for d in ds.DISTRICTS)
        worst = min(ds.DISTRICTS, key=lambda d: ev.indicators[d][k])
        value = 100.0 - (0.5 * avg + 0.5 * ev.indicators[worst][k])
        out[key] = {
            "label": label, "value": round(value, 1), "level": _level(value), "worst_district": worst,
            "indicator": k, "basis": f"100 − (0.5·среднее + 0.5·минимум по районам) показателя {k}",
            "kind": "model_proxy",
        }
    tension = min(100.0, max(0.0, 2.5 * max(0.0, 60.0 - ev.weakest_score) + 8.0 * ev.n_crit))
    out["social_tension"] = {
        "label": "Риск социальной напряжённости (жалобы, петиции, коллективные обращения)",
        "value": round(tension, 1), "level": _level(tension), "worst_district": ev.weakest_district,
        "indicator": "D_min, N_crit", "basis": "2.5·max(0, 60 − балл слабейшего района) + 8·N_crit",
        "kind": "model_proxy",
    }
    discontent = 100.0 - ev.d_avg
    out["public_discontent"] = {
        "label": "Индекс недовольства жителей", "value": round(discontent, 1), "level": _level(discontent),
        "worst_district": ev.weakest_district, "indicator": "D_avg", "basis": "100 − D_avg",
        "kind": "model_proxy",
    }
    return out


SENSITIVITY_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"key": "harsh_winter", "name": "Суровая зима", "focus": "надёжность ЖКХ, воздух, дороги",
     "multipliers": {"C1": 1.8, "E2": 1.5, "T1": 1.3}},
    {"key": "flood_season", "name": "Паводок и ливни", "focus": "сети, реагирование, безопасность дорог",
     "multipliers": {"C1": 1.6, "C2": 1.4, "B2": 1.2}},
    {"key": "hot_summer", "name": "Жаркое лето", "focus": "зелень, воздух, здоровье",
     "multipliers": {"E1": 1.8, "E2": 1.2, "S2": 1.2}},
    {"key": "population_growth", "name": "Рост населения", "focus": "школы, поликлиники, транспорт",
     "multipliers": {"S1": 1.7, "S2": 1.3, "T2": 1.3}},
)


def sensitivity(fractions: Mapping[Placement, float]) -> list[dict[str, Any]]:
    """Прирост к базе при перевзвешивании показателей. Чем меньше gain — тем уязвимее решение к сценарию."""
    default_gain = evaluate_fractions(fractions).score - base_evaluation().score
    rows = []
    for sc in SENSITIVITY_SCENARIOS:
        w = normalized_weights(sc["multipliers"])
        base = base_evaluation(w).score
        score = evaluate_fractions(fractions, w).score
        rows.append({
            "key": sc["key"], "name": sc["name"], "focus": sc["focus"],
            "score": round(score, 3), "base_score": round(base, 3), "gain_vs_base": round(score - base, 3),
            "gain_change": round((score - base) - default_gain, 3), "kind": "sensitivity_analysis",
        })
    return rows
