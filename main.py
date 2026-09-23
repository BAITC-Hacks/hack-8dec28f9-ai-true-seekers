"""Аким на 5 часов: детерминированная модель + объясняющие AI-агенты.

Python 3.10+. Запуск: python main.py --input example_allocation.json --offline
Без --offline при наличии OPENAI_API_KEY вызываются три AI-роли.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


BUDGET = 100.0  # Условные единицы; в демо можно называть млрд тенге.
SECTORS = ("transport", "greening", "social", "safety", "services")
SECTOR_RU = {
    "transport": "транспорт", "greening": "озеленение",
    "social": "соцсфера", "safety": "безопасность",
    "services": "городские сервисы",
}
SECTOR_KEYS = {
    "transport": ("T1", "T2"), "greening": ("E1", "E2"),
    "social": ("S1", "S2"), "safety": ("B1", "B2"),
    "services": ("C1", "C2"),
}
WEIGHTS = {
    "T1": .10, "T2": .10, "E1": .09, "E2": .11,
    "S1": .11, "S2": .11, "B1": .09, "B2": .09,
    "C1": .10, "C2": .10,
}
DISTRICTS = {
    "Есиль": {"pop": .27, "T1": 45, "T2": 62, "E1": 68, "E2": 72,
              "S1": 48, "S2": 55, "B1": 78, "B2": 60, "C1": 75, "C2": 70},
    "Алматы": {"pop": .24, "T1": 40, "T2": 75, "E1": 50, "E2": 55,
                "S1": 60, "S2": 65, "B1": 62, "B2": 52, "C1": 50, "C2": 60},
    "Сарыарка": {"pop": .20, "T1": 50, "T2": 70, "E1": 42, "E2": 40,
                  "S1": 62, "S2": 68, "B1": 58, "B2": 55, "C1": 45, "C2": 55},
    "Байконур": {"pop": .13, "T1": 52, "T2": 68, "E1": 55, "E2": 50,
                 "S1": 58, "S2": 60, "B1": 52, "B2": 58, "C1": 55, "C2": 58},
    "Нура": {"pop": .16, "T1": 55, "T2": 40, "E1": 45, "E2": 65,
             "S1": 38, "S2": 35, "B1": 55, "B2": 50, "C1": 60, "C2": 50},
}

# A: максимальный прирост индикаторов; tau: масштаб насыщения;
# target: доля сбалансированного бюджета. Коэффициенты учебные.
ALLOC_PARAMS = {
    "transport": {"A": 24.0, "tau": 24.0, "target": .24},
    "greening": {"A": 20.0, "tau": 20.0, "target": .16},
    "social": {"A": 24.0, "tau": 22.0, "target": .23},
    "safety": {"A": 19.0, "tau": 18.0, "target": .16},
    "services": {"A": 23.0, "tau": 24.0, "target": .21},
}

# id: (название, направление, стоимость, область, лаг в кварталах, эффекты)
MEASURES = {
    "M1": ("Выделенные полосы для автобусов", "transport", 18, "district", 2, {"T1": 6, "T2": 9}),
    "M2": ("Умные светофоры", "transport", 22, "city", 2, {"T1": 4, "B2": 3}),
    "M3": ("Линия ЛРТ / расширение", "transport", 30, "district", 4, {"T1": 16, "T2": 20, "E2": 4}),
    "M4": ("Парк / сквер", "greening", 15, "district", 2, {"E1": 12, "E2": 3, "B1": 2}),
    "M5": ("Чистое топливо", "greening", 25, "district", 3, {"E2": 14, "C1": 4}),
    "M6": ("Озеленение и ветрозащитные полосы", "greening", 20, "city", 4, {"E1": 5, "E2": 3}),
    "M7": ("Школа + детсад", "social", 24, "district", 3, {"S1": 16}),
    "M8": ("Центр семейного здоровья", "social", 20, "district", 3, {"S2": 14}),
    "M9": ("Дворовые спорт-хабы", "social", 10, "district", 1, {"S1": 3, "S2": 3, "B1": 3}),
    "M10": ("Освещение и камеры", "safety", 12, "district", 1, {"B1": 12, "B2": 2}),
    "M11": ("Переходы и школьные зоны", "safety", 10, "district", 1, {"B2": 12, "T1": -2}),
    "M12": ("Платформа обращений", "services", 14, "city", 1, {"C2": 5}),
    "M13": ("Модернизация тепло- и водосетей", "services", 28, "district", 4, {"C1": 18, "E2": 2}),
    "M14": ("Аварийные бригады ЖКХ", "services", 16, "city", 1, {"C1": 5, "C2": 2}),
}


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: ожидается число")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label}: нужно конечное неотрицательное число")
    return number


def validate_allocation(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict) or set(raw) != set(SECTORS):
        raise ValueError(f"allocation должен содержать ровно: {', '.join(SECTORS)}")
    amounts = {key: finite_number(raw[key], key) for key in SECTORS}
    if sum(amounts.values()) > BUDGET + 1e-9:
        raise ValueError(f"Превышен бюджет {BUDGET:g}: запрошено {sum(amounts.values()):g}")
    return amounts


def validate_measures(raw: Any) -> list[dict[str, str | None]]:
    if not isinstance(raw, list) or len(raw) != 5:
        raise ValueError("Нужно принять ровно 5 решений-мероприятий")
    selected: list[dict[str, str | None]] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict) or "id" not in item:
            raise ValueError(f"Решение {index}: нужен объект с id")
        mid = item["id"]
        if not isinstance(mid, str) or mid not in MEASURES:
            raise ValueError(f"Решение {index}: неизвестный id {mid!r}")
        scope = MEASURES[mid][3]
        district = item.get("district")
        if scope == "district" and district not in DISTRICTS:
            raise ValueError(f"{mid}: укажите район из {', '.join(DISTRICTS)}")
        if scope == "city" and district is not None:
            raise ValueError(f"{mid}: для городской меры район не указывается")
        selected.append({"id": mid, "district": district})
    ids = [item["id"] for item in selected]
    if len(set(ids)) != 5:
        raise ValueError("Повторять мероприятие нельзя")
    cost = sum(MEASURES[mid][2] for mid in ids)
    if cost > BUDGET:
        raise ValueError(f"Превышен бюджет {BUDGET:g}: стоимость {cost:g}")
    counts = Counter(MEASURES[mid][1] for mid in ids)
    if any(count > 2 for count in counts.values()):
        raise ValueError("Нельзя выбирать больше 2 мероприятий одного направления")
    by_id = {item["id"]: item["district"] for item in selected}
    if "M1" in by_id and "M3" in by_id:
        raise ValueError("M1 и M3 несовместимы в любых районах")
    for left, right in (("M4", "M7"), ("M5", "M13")):
        if left in by_id and right in by_id and by_id[left] == by_id[right]:
            raise ValueError(f"{left} и {right} несовместимы в одном районе")
    return selected


def empty_state() -> dict[str, dict[str, float]]:
    return {name: {key: float(row[key]) for key in WEIGHTS}
            for name, row in DISTRICTS.items()}


def clip_state(state: dict[str, dict[str, float]]) -> None:
    for values in state.values():
        for key in WEIGHTS:
            values[key] = min(100.0, max(0.0, values[key]))


def apply_measures(items: list[dict[str, str | None]]) -> dict[str, dict[str, float]]:
    state = empty_state()
    by_id = {str(item["id"]): item["district"] for item in items}
    for item in items:
        mid = str(item["id"])
        _, _, _, scope, lag, effects = MEASURES[mid]
        targets = state if scope == "city" else [str(item["district"])]
        for name in targets:
            for key, effect in effects.items():
                state[name][key] += effect * (8 - lag) / 8
    if "M1" in by_id and "M2" in by_id:
        state[str(by_id["M1"])]["T1"] += 2
    if "M10" in by_id and "M12" in by_id:
        state[str(by_id["M10"])]["B1"] += 2
    if "M5" in by_id and "M6" in by_id:
        state[str(by_id["M5"])]["E2"] += 2
    clip_state(state)
    return state


def apply_allocation(amounts: dict[str, float]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    state = empty_state()
    utilization: dict[str, float] = {}
    for sector in SECTORS:
        p = ALLOC_PARAMS[sector]
        u = 1 - math.exp(-amounts[sector] / p["tau"])
        utilization[sector] = u
        keys = SECTOR_KEYS[sector]
        deficit = {
            name: 100 - sum(DISTRICTS[name][key] for key in keys) / len(keys)
            for name in DISTRICTS
        }
        mean_deficit = sum(DISTRICTS[name]["pop"] * value
                           for name, value in deficit.items())
        for name in DISTRICTS:
            gain = p["A"] * u * deficit[name] / mean_deficit
            for key in keys:
                state[name][key] += gain
    # Комплементарность: бонус зависит от двух вложений и равен нулю,
    # если хотя бы одно из них равно нулю.
    synergies = (
        ("transport", "services", "T1", 2.0),
        ("transport", "greening", "E2", 1.5),
        ("social", "safety", "B1", 2.0),
        ("services", "safety", "C1", 1.5),
    )
    for left, right, indicator, coefficient in synergies:
        bonus = coefficient * math.sqrt(utilization[left] * utilization[right])
        for name in DISTRICTS:
            state[name][indicator] += bonus
    clip_state(state)
    return state, utilization


def balance_penalties(amounts: dict[str, float]) -> dict[str, float]:
    spent = sum(amounts.values())
    if spent == 0:
        return {"concentration": 0.0, "underfunding": 0.0}
    shares = {key: amounts[key] / spent for key in SECTORS}
    concentration = 6 * (spent / BUDGET) * sum(
        (shares[key] - ALLOC_PARAMS[key]["target"]) ** 2 for key in SECTORS
    )
    underfunding = 1.5 * (spent / BUDGET) * sum(
        max(0.0, 0.5 * ALLOC_PARAMS[key]["target"] - shares[key])
        / (0.5 * ALLOC_PARAMS[key]["target"])
        for key in SECTORS
    )
    return {"concentration": concentration, "underfunding": underfunding}


def score_state(state: dict[str, dict[str, float]],
                extra_penalty: float = 0.0) -> dict[str, Any]:
    district_scores = {
        name: sum(WEIGHTS[key] * row[key] for key in WEIGHTS)
        for name, row in state.items()
    }
    city_average = sum(DISTRICTS[name]["pop"] * score
                       for name, score in district_scores.items())
    weakest = min(district_scores, key=district_scores.get)
    critical = [{"district": name, "indicator": key, "value": round(row[key], 2)}
                for name, row in state.items() for key in WEIGHTS if row[key] < 40]
    raw_score = .7 * city_average + .3 * district_scores[weakest]
    final_score = min(100.0, max(0.0, raw_score - len(critical) - extra_penalty))
    return {
        "score": round(final_score, 2),
        "city_average": round(city_average, 2),
        "weakest_district": weakest,
        "weakest_score": round(district_scores[weakest], 2),
        "critical": critical,
        "critical_penalty": len(critical),
        "district_scores": {name: round(value, 2) for name, value in district_scores.items()},
    }


def evaluate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Входные данные должны быть JSON-объектом")
    mode = payload.get("mode", "allocations")
    baseline = score_state(empty_state())
    if mode == "allocations":
        amounts = validate_allocation(payload.get("allocation"))
        state, utilization = apply_allocation(amounts)
        penalties = balance_penalties(amounts)
        computed = score_state(state, sum(penalties.values()))
        spent = sum(amounts.values())
        details: dict[str, Any] = {
            "allocation": amounts,
            "utilization": {key: round(value, 4) for key, value in utilization.items()},
            "balance_penalties": {key: round(value, 2) for key, value in penalties.items()},
        }
    elif mode == "measures":
        items = validate_measures(payload.get("decisions"))
        state = apply_measures(items)
        computed = score_state(state)
        spent = sum(MEASURES[str(item["id"])][2] for item in items)
        details = {
            "decisions": [
                {"id": item["id"], "name": MEASURES[str(item["id"])][0],
                 "sector": MEASURES[str(item["id"])][1], "district": item["district"],
                 "cost": MEASURES[str(item["id"])][2]}
                for item in items
            ],
            "marginal_score_if_removed": {
                str(item["id"]): round(
                    computed["score"] - score_state(apply_measures(
                        [other for other in items if other is not item]))["score"], 2)
                for item in items
            },
        }
    else:
        raise ValueError("mode должен быть allocations или measures")
    deltas = {
        name: {key: round(state[name][key] - DISTRICTS[name][key], 2)
               for key in WEIGHTS if abs(state[name][key] - DISTRICTS[name][key]) > 1e-9}
        for name in DISTRICTS
    }
    return {
        "mode": mode, "budget": BUDGET, "spent": round(spent, 2),
        "remaining": round(BUDGET - spent, 2),
        "baseline_score": baseline["score"],
        "score_delta": round(computed["score"] - baseline["score"], 2),
        **computed, "details": details, "indicator_deltas": deltas,
        "model_note": "Синтетическая сценарная модель; прогноз не является наблюдаемым фактом.",
    }


def analytics_agent(result: dict[str, Any]) -> dict[str, Any]:
    scores = result["district_scores"]
    top = max(scores, key=scores.get)
    return {
        "role": "Analytics & Math Engine", "source": "deterministic",
        "summary": (f"Индекс {result['score']:.2f}/100; изменение к базе "
                    f"{result['score_delta']:+.2f}. Слабейший район — "
                    f"{result['weakest_district']} ({result['weakest_score']:.2f})."),
        "strengths": [f"Максимальный районный балл: {top} — {scores[top]:.2f}."],
        "risks": [f"Критических пар район × показатель (<40): {len(result['critical'])}."],
        "recommendations": ["Проверить эффект перераспределения на самый слабый район."],
    }


def fallback_urban(result: dict[str, Any]) -> dict[str, Any]:
    weakest = result["weakest_district"]
    risks = []
    if result["mode"] == "allocations":
        a = result["details"]["allocation"]
        if a["services"] < 10:
            risks.append("Слабое финансирование сервисов повышает модельный зимний риск для ЖКХ и уборки снега.")
        if a["transport"] < 10:
            risks.append("Недостаток вложений в транспорт оставляет проблему мостов и общественного транспорта.")
        if a["greening"] > 0 and a["services"] < 10:
            risks.append("Озеленение требует проверки обеспеченности водой и ухода в местном климате.")
    else:
        chosen = {item["id"] for item in result["details"]["decisions"]}
        if "M13" not in chosen and "M14" not in chosen:
            risks.append("В сценарии нет отдельной меры против аварий тепло- и водосетей.")
        if "M1" not in chosen and "M2" not in chosen and "M3" not in chosen:
            risks.append("Нет отдельной транспортной меры против пробок.")
    if not risks:
        risks.append("Зимняя эксплуатация, мосты и водоснабжение требуют отдельной проверки на местных данных.")
    return {
        "role": "Astana Urban Expert", "source": "rules",
        "summary": f"Приоритет проверки на местности — район {weakest}.",
        "strengths": ["В расчёте учтены разные стартовые условия пяти районов."],
        "risks": risks,
        "recommendations": ["Проверить сроки эффекта, зимнюю эксплуатацию и доступность воды перед внедрением."],
    }


def fallback_social(result: dict[str, Any]) -> dict[str, Any]:
    proxy = min(100, round(100 - result["weakest_score"] + 5 * len(result["critical"])))
    return {
        "role": "Social & Risk Agent", "source": "rules",
        "summary": f"Прокси социального риска: {proxy}/100; данные соцсетей не подключены.",
        "strengths": ["Оценка отдельно учитывает самый слабый район."],
        "risks": ["Низкие значения инфраструктурных показателей могут повышать недовольство; это сценарный риск, не прогноз протестов."],
        "recommendations": ["Перед реальным решением добавить обезличенные обращения жителей и данные об авариях ЖКХ."],
    }


def fallback_chief(result: dict[str, Any], urban: dict[str, Any],
                   social: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "Akim Chief Advisor", "source": "rules",
        "headline": f"Astana Quality of Life Score: {result['score']:.2f}/100",
        "summary": (f"Сценарий меняет индекс на {result['score_delta']:+.2f} пункта "
                    f"при расходе {result['spent']:.0f} из {BUDGET:.0f}."),
        "strengths": [f"Лучший район: {max(result['district_scores'], key=result['district_scores'].get)}."],
        "risks": [*urban["risks"], *social["risks"]],
        "recommendations": [
            f"Перепроверить эффект для района {result['weakest_district']}.",
            "Сравнить альтернативные наборы решений на тех же исходных данных.",
        ],
    }


async def ai_report(result: dict[str, Any], model: str) -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("Для AI-режима установите зависимости: pip install -r requirements.txt") from exc

    class ExpertNote(BaseModel):
        summary: str
        strengths: list[str]
        risks: list[str]
        recommendations: list[str]

    class ChiefNote(BaseModel):
        headline: str
        summary: str
        strengths: list[str]
        risks: list[str]
        recommendations: list[str]

    facts = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    common = ("Пиши по-русски кратко и предметно. Вход — результаты синтетической "
              "модели. Не считай новый score, не выдумывай показатели, цены, "
              "статистику соцсетей или реальные прогнозы. Не добавляй числа, "
              "которых нет во входном JSON. Формулируй последствия как сценарные риски.")

    async with AsyncOpenAI(timeout=35.0, max_retries=1) as client:
        async def expert(role: str, instruction: str) -> dict[str, Any]:
            response = await client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": f"Ты {role}. {instruction} {common}"},
                    {"role": "user", "content": facts},
                ],
                text_format=ExpertNote,
            )
            if response.output_parsed is None:
                raise RuntimeError(f"{role}: модель не вернула структурированный ответ")
            return {"role": role, "source": "openai", **response.output_parsed.model_dump()}

        urban, social = await asyncio.gather(
            expert("Astana Urban Expert",
                   "Учитывай зимний снег, мосты, воду, застройку и профиль районов; "
                   "не утверждай, что эффекты доказаны наблюдениями."),
            expert("Social & Risk Agent",
                   "Оцени социальные и инфраструктурные риски как гипотезы. "
                   "Данные соцсетей отсутствуют; явно обозначь это."),
        )
        chief_input = json.dumps({"facts": result, "analytics": analytics_agent(result),
                                  "urban": urban, "social": social}, ensure_ascii=False)
        response = await client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": "Ты Akim Chief Advisor. Сведи выводы трёх ролей в доклад акиму. " + common},
                {"role": "user", "content": chief_input},
            ],
            text_format=ChiefNote,
        )
        if response.output_parsed is None:
            raise RuntimeError("Akim Chief Advisor: нет структурированного ответа")
        chief = {"role": "Akim Chief Advisor", "source": "openai",
                 **response.output_parsed.model_dump()}
    return {"analytics": analytics_agent(result), "urban": urban,
            "social": social, "chief": chief}


def offline_report(result: dict[str, Any]) -> dict[str, Any]:
    urban = fallback_urban(result)
    social = fallback_social(result)
    return {"analytics": analytics_agent(result), "urban": urban,
            "social": social, "chief": fallback_chief(result, urban, social)}


def main() -> int:
    # На Windows перенаправленный stdout нередко имеет системную кодировку,
    # которая не умеет печатать русские имена районов и знак ×.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Путь к JSON; без аргумента читается stdin")
    parser.add_argument("--offline", action="store_true", help="Детерминированный отчёт без API")
    parser.add_argument("--model", help="Модель OpenAI; по умолчанию OPENAI_MODEL или gpt-4o-mini")
    args = parser.parse_args()
    try:
        if args.input:
            raw = Path(args.input).read_text(encoding="utf-8-sig")
        elif sys.stdin.isatty():
            raise ValueError("Укажите --input path.json или передайте JSON в stdin")
        else:
            raw = sys.stdin.read()
        payload = json.loads(raw)
        result = evaluate(payload)
        if not args.offline:
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).with_name(".env"))
            except ImportError:
                pass
        if not args.offline and os.getenv("OPENAI_API_KEY"):
            result["ai_mode"] = "openai"
            result["agents"] = asyncio.run(ai_report(
                result, args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
        else:
            result["ai_mode"] = "offline_rules"
            result["agents"] = offline_report(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"error": f"Ошибка AI API: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

