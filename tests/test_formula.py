"""Формула QoL/Score: совпадение с опорными числами датасета и свойства формулы."""

from __future__ import annotations

import pytest

from akim_sim import dataset as ds
from akim_sim.model import (Choice, base_evaluation, evaluate, evaluate_plan, realize, validate_plan)

from .conftest import WORKED_EXAMPLE_ITEMS


def test_weights_and_population_sum_to_one() -> None:
    assert sum(ds.WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(ds.POPULATION.values()) == pytest.approx(1.0)


def test_baseline_matches_dataset_reference() -> None:
    ev = base_evaluation()
    assert ev.score == pytest.approx(ds.REFERENCE_BASE_SCORE, abs=0.005)  # 52.558 ≈ 52.56
    assert ev.n_crit == 2
    assert {(d, k) for d, k, _ in ev.critical_cells} == {("Нура", "S1"), ("Нура", "S2")}
    assert ev.weakest_district == "Нура"
    assert ev.weakest_score == pytest.approx(49.18, abs=1e-9)


def test_worked_example_matches_dataset_reference() -> None:
    choices, reasons = validate_plan(WORKED_EXAMPLE_ITEMS)
    assert reasons == []
    assert sum(ch.measure.cost for ch in choices) == 95
    ev = evaluate_plan(choices)
    assert ev.score == pytest.approx(56.543, abs=0.001)
    assert ev.score == pytest.approx(ds.REFERENCE_WORKED_EXAMPLE_SCORE, abs=0.05)
    assert ev.n_crit == 0


def test_score_is_official_combination_of_components() -> None:
    ev = evaluate_plan([Choice("M3", "Нура"), Choice("M14", ds.CITY)])
    expected = ds.W_AVG * ev.d_avg + ds.W_MIN * ev.weakest_score - ds.CRIT_PENALTY * ev.n_crit
    assert ev.score == pytest.approx(expected)
    for d in ds.DISTRICTS:
        assert ev.district_scores[d] == pytest.approx(sum(ds.WEIGHTS[k] * ev.indicators[d][k] for k in ds.INDICATORS))
    assert ev.d_avg == pytest.approx(sum(ds.POPULATION[d] * ev.district_scores[d] for d in ds.DISTRICTS))


@pytest.mark.parametrize("code,share", [("M3", 0.5), ("M13", 0.5), ("M7", 0.625), ("M1", 0.75), ("M10", 0.875)])
def test_lag_scaling(code: str, share: float) -> None:
    assert ds.MEASURES[code].realized_share == pytest.approx(share)


def test_district_measure_affects_only_its_district() -> None:
    ind = realize({("M3", "Нура"): 1.0})
    assert ind["Нура"]["T1"] == pytest.approx(55 + 16 * 0.5)
    assert ind["Нура"]["T2"] == pytest.approx(40 + 20 * 0.5)
    assert ind["Есиль"]["T1"] == ds.BASELINE["Есиль"]["T1"]


def test_citywide_measure_affects_all_districts() -> None:
    ind = realize({("M14", ds.CITY): 1.0})
    for d in ds.DISTRICTS:
        assert ind[d]["C1"] == pytest.approx(ds.BASELINE[d]["C1"] + 5 * 0.875)
        assert ind[d]["C2"] == pytest.approx(ds.BASELINE[d]["C2"] + 2 * 0.875)


def test_synergy_bonus_is_fixed_and_only_in_anchor_district() -> None:
    ind = realize({("M10", "Байконур"): 1.0, ("M12", ds.CITY): 1.0})
    assert ind["Байконур"]["B1"] == pytest.approx(52 + 12 * 0.875 + 2.0)  # +2 без масштабирования лагом
    assert ind["Нура"]["B1"] == ds.BASELINE["Нура"]["B1"]
    without = realize({("M10", "Байконур"): 1.0})
    assert without["Байконур"]["B1"] == pytest.approx(52 + 12 * 0.875)


def test_negative_effect_can_create_new_critical_pair() -> None:
    # M11 даёт T1 −2·0.875; в Алматы T1 = 40 → 38.25 < 40 — новая критическая пара
    ev = evaluate_plan([Choice("M11", "Алматы")])
    assert ("Алматы", "T1") in {(d, k) for d, k, _ in ev.critical_cells}
    assert ev.n_crit == 3


def test_indicators_are_clipped_to_0_100() -> None:
    ind = realize({("M10", "Есиль"): 10.0})  # искусственно завышенная доля
    assert ind["Есиль"]["B1"] == 100.0
    ev = evaluate(ind)
    assert all(0.0 <= v <= 100.0 for d in ds.DISTRICTS for v in ev.indicators[d].values())


def test_crit_threshold_is_strict() -> None:
    # значение ровно 40 не критическое (правило «< 40»)
    assert ds.BASELINE["Сарыарка"]["E2"] == 40
    assert ("Сарыарка", "E2") not in {(d, k) for d, k, _ in base_evaluation().critical_cells}
