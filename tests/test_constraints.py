"""Жёсткие ограничения: бюджет, ровно 5 мер, повторы, ≤2 на направление, несовместимости, районы."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from akim_sim.inputs import AllocationDecision, MeasuresDecision, PlanValidationError, parse_decision
from akim_sim.model import validate_plan

from .conftest import OPTIMUM_ITEMS, WORKED_EXAMPLE_ITEMS


def reasons(items: list) -> list[str]:
    return validate_plan(items)[1]


def test_valid_sets() -> None:
    assert reasons(WORKED_EXAMPLE_ITEMS) == []
    assert reasons(OPTIMUM_ITEMS) == []


def test_budget_exactly_100_is_allowed() -> None:
    assert reasons(["M3:Нура", "M7:Нура", "M8:Нура", "M10:Нура", "M12"]) == []  # 30+24+20+12+14 = 100


def test_budget_overspend_is_rejected() -> None:
    r = reasons(["M3:Нура", "M13:Алматы", "M7:Нура", "M5:Сарыарка", "M9:Нура"])  # 30+28+24+25+10 = 117
    assert any("Бюджет превышен" in x and "117" in x for x in r)


def test_custom_lower_budget_is_enforced() -> None:
    _, r = validate_plan(WORKED_EXAMPLE_ITEMS, budget=90)
    assert any("Бюджет превышен" in x for x in r)


@pytest.mark.parametrize("n", [4, 6])
def test_exactly_five_decisions(n: int) -> None:
    items = ["M9:Нура", "M10:Нура", "M12", "M14", "M4:Сарыарка", "M11:Есиль"][:n]
    assert any("ровно 5" in x for x in reasons(items))


def test_repeats_are_rejected_even_in_other_district() -> None:
    assert any("Повторы" in x for x in reasons(["M9:Нура", "M9:Есиль", "M10:Нура", "M12", "M14"]))


def test_max_two_per_direction() -> None:
    r = reasons(["M7:Нура", "M8:Нура", "M9:Нура", "M12", "M10:Нура"])
    assert any("Не более 2" in x and "Социальная" in x for x in r)


def test_m1_m3_incompatible_in_any_district() -> None:
    r = reasons(["M1:Есиль", "M3:Нура", "M9:Нура", "M12", "M10:Нура"])
    assert any("M1 и M3" in x for x in r)


def test_same_district_incompatibilities() -> None:
    assert any("M4 и M7 в районе Нура" in x for x in reasons(["M4:Нура", "M7:Нура", "M12", "M10:Нура", "M11:Есиль"]))
    assert reasons(["M4:Есиль", "M7:Нура", "M12", "M10:Нура", "M11:Есиль"]) == []
    assert any("M5 и M13" in x for x in reasons(["M5:Сарыарка", "M13:Сарыарка", "M12", "M10:Нура", "M9:Нура"]))


def test_district_rules() -> None:
    r = reasons(["M7", "M12:Нура", "M8:Москва", "M9:Нура", "M10:Нура"])
    assert any("M7 — районная мера" in x for x in r)
    assert any("M12 — мера масштаба города" in x for x in r)
    assert any("неизвестный район «Москва»" in x for x in r)


def test_all_reasons_are_returned_at_once() -> None:
    r = reasons(["M1:Нура", "M3:Нура", "M99", "M7", "M8:Нура", "M9:Нура"])
    assert len(r) >= 4  # количество, неизвестная мера, район, несовместимость


def test_input_normalization() -> None:
    assert reasons(["m9@нура", {"code": "М10", "district": "Nura"}, ["M12", "город"], "M14", "M4/Saryarka"]) == []


def test_measures_decision_raises_with_reasons() -> None:
    with pytest.raises(PlanValidationError) as exc:
        MeasuresDecision(measures=["M1:Нура", "M3:Нура", "M9:Нура", "M12", "M14"]).choices()
    assert exc.value.reasons


def test_measures_mapping_input() -> None:
    d = MeasuresDecision(measures={"M2": "город", "M3": "Нура", "M8": "Нура", "M9": "Нура", "M14": "город"})
    assert len(d.choices()) == 5


# ── режим сумм ──────────────────────────────────────────────────────────────


def test_allocation_overspend_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Бюджет превышен"):
        AllocationDecision(allocations={"transport": 40, "green": 20, "social": 30, "safety": 20, "utilities": 10})


def test_allocation_exact_budget_ok_and_aliases() -> None:
    d = AllocationDecision.model_validate({"Транспорт": 25, "greening": 15, "social": 25, "security": 15, "ЖКХ": 20})
    assert sum(d.allocations.values()) == 100


@pytest.mark.parametrize("alloc,msg", [
    ({"transport": -1, "green": 0, "social": 0, "safety": 0, "utilities": 0}, "неотрицательным"),
    ({"transport": 10, "green": 10, "social": 10}, "не хватает"),
    ({"transport": 10, "green": 10, "social": 10, "safety": 10, "utilities": 10, "space": 5}, "Неизвестная сфера"),
    ({"transport": "много", "green": 0, "social": 0, "safety": 0, "utilities": 0}, "числом"),
])
def test_allocation_invalid(alloc: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        AllocationDecision(allocations=alloc)


def test_parse_decision_dispatch() -> None:
    assert parse_decision(OPTIMUM_ITEMS).mode == "measures"
    assert parse_decision({"measures": OPTIMUM_ITEMS}).mode == "measures"
    assert parse_decision({"transport": 20, "green": 20, "social": 20, "safety": 20, "utilities": 20}).mode == "allocation"
    legacy = {"mode": "allocations", "allocation": {"transport": 25, "greening": 15, "social": 25,
                                                       "safety": 15, "services": 20}}
    assert parse_decision(legacy).allocations == {"transport": 25, "green": 15, "social": 25,
                                                  "safety": 15, "utilities": 20}
    with pytest.raises(ValueError):
        parse_decision({"mode": "lottery"})
