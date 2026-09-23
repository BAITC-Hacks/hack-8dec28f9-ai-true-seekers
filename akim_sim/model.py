"""Официальная модель оценки (формула датасета), валидатор наборов мер и режим сумм.

Score = 0.7 · D_avg + 0.3 · min_d D_d − 1.0 · N_crit
    D_d     = Σ_k w_k · I'_{d,k}                       — балл района
    D_avg   = Σ_d pop_d · D_d                          — среднее по городу (веса — доли населения)
    N_crit  = число пар (район, показатель) с I' < 40
    I'_{d,k} = clip(I_{d,k} + Σ_m эффект_{m,k} · (8 − L_m)/8 + синергии, 0, 100)

Режим сумм задаёт верхние лимиты на пять направлений. Точный оптимизатор
выбирает из каталога пять целых мер с общей формулой Score.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from . import dataset as ds

Placement = tuple[str, str]  # (код меры, район или ds.CITY)
EPS = 1e-9


# ════════════════════════════════════════════════════════════════════════════
# Расчёт показателей и Score
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Evaluation:
    indicators: dict[str, dict[str, float]]
    district_scores: dict[str, float]
    d_avg: float
    weakest_district: str
    weakest_score: float
    critical_cells: list[tuple[str, str, float]]
    score: float

    @property
    def n_crit(self) -> int:
        return len(self.critical_cells)


def baseline_indicators() -> dict[str, dict[str, float]]:
    return {d: dict(row) for d, row in ds.BASELINE.items()}


def _targets(code: str, district: str) -> tuple[str, ...]:
    return ds.DISTRICTS if ds.MEASURES[code].citywide else (district,)


def realize(fractions: Mapping[Placement, float]) -> dict[str, dict[str, float]]:
    """Показатели после мер. fractions[(код, район)] — доля реализации меры (1.0 в режиме мер)."""
    ind = baseline_indicators()
    for (code, district), frac in fractions.items():
        if frac <= EPS:
            continue
        m = ds.MEASURES[code]
        share = m.realized_share * frac
        for d in _targets(code, district):
            for k, v in m.effects.items():
                ind[d][k] += v * share
    for syn in ds.SYNERGIES:  # фиксированный бонус, без масштабирования лагом
        partner = syn.b if syn.anchor == syn.a else syn.a
        f_partner = min(1.0, sum(f for (c, _), f in fractions.items() if c == partner))
        if f_partner <= EPS:
            continue
        for (code, district), frac in fractions.items():
            if code == syn.anchor and frac > EPS:
                for d in _targets(code, district):
                    ind[d][syn.indicator] += syn.bonus * min(frac, f_partner)
    for d in ds.DISTRICTS:
        for k in ds.INDICATORS:
            ind[d][k] = min(100.0, max(0.0, ind[d][k]))
    return ind


def evaluate(indicators: Mapping[str, Mapping[str, float]], weights: Mapping[str, float] | None = None) -> Evaluation:
    w = weights or ds.WEIGHTS
    scores = {d: sum(w[k] * indicators[d][k] for k in ds.INDICATORS) for d in ds.DISTRICTS}
    d_avg = sum(ds.POPULATION[d] * scores[d] for d in ds.DISTRICTS)
    weakest = min(ds.DISTRICTS, key=lambda d: scores[d])
    crit = [(d, k, indicators[d][k]) for d in ds.DISTRICTS for k in ds.INDICATORS
            if indicators[d][k] < ds.CRIT_THRESHOLD]
    score = max(0.0, min(100.0, ds.W_AVG * d_avg + ds.W_MIN * scores[weakest] - ds.CRIT_PENALTY * len(crit)))
    return Evaluation({d: dict(indicators[d]) for d in ds.DISTRICTS}, scores, d_avg, weakest, scores[weakest], crit, score)


def evaluate_fractions(fractions: Mapping[Placement, float], weights: Mapping[str, float] | None = None) -> Evaluation:
    return evaluate(realize(fractions), weights)


def base_evaluation(weights: Mapping[str, float] | None = None) -> Evaluation:
    return evaluate(baseline_indicators(), weights)


def normalized_weights(multipliers: Mapping[str, float]) -> dict[str, float]:
    raw = {k: ds.WEIGHTS[k] * multipliers.get(k, 1.0) for k in ds.INDICATORS}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# ════════════════════════════════════════════════════════════════════════════
# Режим мероприятий: разбор и валидатор (7 правил датасета)
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Choice:
    code: str
    district: str  # ds.CITY для мер масштаба города

    @property
    def measure(self) -> ds.Measure:
        return ds.MEASURES[self.code]

    def label(self) -> str:
        return f"{self.code} ({self.district})"


def parse_choice(item: Any) -> tuple[Choice | None, str | None]:
    """Принимает "M7:Нура", "M7@Нура", "M12", {"code": "M7", "district": "Нура"}, ["M7", "Нура"]."""
    code: Any
    district: Any = None
    if isinstance(item, str):
        text = item.strip()
        for sep in (":", "@", "/"):
            if sep in text:
                code, district = text.split(sep, 1)
                break
        else:
            code = text
    elif isinstance(item, Mapping):
        code = item.get("code") or item.get("measure") or item.get("id")
        district = item.get("district") or item.get("район")
    elif isinstance(item, (list, tuple)) and 1 <= len(item) <= 2:
        code = item[0]
        district = item[1] if len(item) == 2 else None
    else:
        return None, f"Не удалось разобрать решение: {item!r}"
    if not code:
        return None, f"У решения не указан код меры: {item!r}"
    code = ds.normalize_code(code)
    if code not in ds.MEASURES:
        return None, f"Неизвестная мера «{code}». Допустимо: {', '.join(ds.MEASURE_CODES)}"
    m = ds.MEASURES[code]
    district = ds.normalize_district(district)
    if m.citywide:
        if district not in (None, ds.CITY):
            return None, f"{code} — мера масштаба города, район указывать нельзя (получено «{district}»)"
        return Choice(code, ds.CITY), None
    if district is None or district == ds.CITY:
        return None, f"{code} — районная мера: нужно указать район ({', '.join(ds.DISTRICTS)})"
    if district not in ds.DISTRICTS:
        return None, f"{code}: неизвестный район «{district}». Допустимо: {', '.join(ds.DISTRICTS)}"
    return Choice(code, district), None


def check_rules(choices: list[Choice], budget: float = ds.BUDGET, check_count: bool = True) -> list[str]:
    """Правила набора: ровно 5, без повторов, бюджет, ≤2 на направление, несовместимости."""
    reasons: list[str] = []
    if check_count and len(choices) != ds.N_DECISIONS:
        reasons.append(f"Нужно ровно {ds.N_DECISIONS} решений, получено {len(choices)}")
    dup = [c for c, n in Counter(ch.code for ch in choices).items() if n > 1]
    if dup:
        reasons.append("Повторы запрещены — каждая мера не более одного раза: " + ", ".join(sorted(dup)))
    cost = sum(ch.measure.cost for ch in choices)
    if cost > budget + EPS:
        reasons.append(f"Бюджет превышен: {cost:g} {ds.UNIT} при лимите {budget:g}")
    per_dir = Counter(ch.measure.sector for ch in choices)
    for sector, n in per_dir.items():
        if n > ds.MAX_PER_DIRECTION:
            reasons.append(f"Не более {ds.MAX_PER_DIRECTION} мер из одного направления: "
                           f"«{ds.SECTOR_NAMES[sector]}» — {n}")
    codes = {ch.code for ch in choices}
    for a, b, why in ds.INCOMPATIBLE_GLOBAL:
        if a in codes and b in codes:
            reasons.append(f"Несовместимы {a} и {b}: {why}")
    for a, b, why in ds.INCOMPATIBLE_SAME_DISTRICT:
        da = {ch.district for ch in choices if ch.code == a}
        db = {ch.district for ch in choices if ch.code == b}
        for d in sorted(da & db):
            reasons.append(f"Несовместимы {a} и {b} в районе {d}: {why}")
    return reasons


def validate_plan(items: Iterable[Any], budget: float = ds.BUDGET) -> tuple[list[Choice], list[str]]:
    """Разбирает и проверяет набор. Возвращает (решения, причины невалидности). Пустой список причин — набор валиден."""
    choices: list[Choice] = []
    reasons: list[str] = []
    items = list(items)
    if len(items) != ds.N_DECISIONS:
        reasons.append(f"Нужно ровно {ds.N_DECISIONS} решений, получено {len(items)}")
    for item in items:
        choice, err = parse_choice(item)
        if err:
            reasons.append(err)
        else:
            choices.append(choice)  # type: ignore[arg-type]
    return choices, reasons + check_rules(choices, budget, check_count=False)


def plan_fractions(choices: Iterable[Choice]) -> dict[Placement, float]:
    return {(ch.code, ch.district): 1.0 for ch in choices}


def evaluate_plan(choices: Iterable[Choice], weights: Mapping[str, float] | None = None) -> Evaluation:
    return evaluate_fractions(plan_fractions(choices), weights)


