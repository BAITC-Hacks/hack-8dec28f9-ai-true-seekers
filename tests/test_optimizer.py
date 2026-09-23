"""Оптимизаторы: точный перебор режима мер, подсказки замен, режим сумм."""

from __future__ import annotations

import itertools
import random

import pytest

from akim_sim import dataset as ds
from akim_sim.model import Choice, check_rules, evaluate_plan, validate_plan
from akim_sim.optimizer import ExactMeasuresOptimizer, best_measure_plans, suggest_swaps

from .conftest import WORKED_EXAMPLE_ITEMS


def test_official_optimum() -> None:
    ranked, n_valid = best_measure_plans(top=5)
    assert n_valid == 694_395
    best = ranked[0]
    assert best.score == pytest.approx(57.237, abs=0.001)
    assert {(c.code, c.district) for c in best.choices} == {
        ("M2", ds.CITY), ("M3", "Нура"), ("M8", "Нура"), ("M9", "Нура"), ("M14", ds.CITY)}
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_ranked_plans_are_valid_and_match_reference_scoring() -> None:
    ranked, _ = best_measure_plans(top=5)
    for r in ranked:
        assert check_rules(list(r.choices)) == []
        assert evaluate_plan(list(r.choices)).score == pytest.approx(r.score, abs=1e-9)


def test_optimum_beats_worked_example() -> None:
    ranked, _ = best_measure_plans(top=1)
    choices, _ = validate_plan(WORKED_EXAMPLE_ITEMS)
    assert ranked[0].score > evaluate_plan(choices).score


def test_fast_scoring_matches_reference_on_random_valid_plans() -> None:
    opt = ExactMeasuresOptimizer()
    rng = random.Random(42)
    checked = 0
    while checked < 300:
        codes = rng.sample(ds.MEASURE_CODES, 5)
        plan = [Choice(c, ds.CITY if ds.MEASURES[c].citywide else rng.choice(ds.DISTRICTS)) for c in codes]
        if check_rules(plan):
            continue
        assert opt.score_choices(plan) == pytest.approx(evaluate_plan(plan).score, abs=1e-9)
        checked += 1


def test_exact_search_equals_brute_force_on_reduced_catalog() -> None:
    codes = ("M1", "M2", "M6", "M9", "M10", "M11", "M12", "M14")
    ranked, n = ExactMeasuresOptimizer(codes=codes).search(top=1)
    best_score, n_ref = float("-inf"), 0
    for combo in itertools.combinations(codes, 5):
        local = [c for c in combo if not ds.MEASURES[c].citywide]
        for dists in itertools.product(ds.DISTRICTS, repeat=len(local)):
            where = dict(zip(local, dists))
            plan = [Choice(c, where.get(c, ds.CITY)) for c in combo]
            if check_rules(plan):
                continue
            n_ref += 1
            best_score = max(best_score, evaluate_plan(plan).score)
    assert n == n_ref
    assert ranked[0].score == pytest.approx(best_score, abs=1e-9)


def test_swaps_are_valid_and_verified() -> None:
    choices, _ = validate_plan(WORKED_EXAMPLE_ITEMS)
    base = evaluate_plan(choices).score
    swaps = suggest_swaps(choices, top=3)
    assert swaps and swaps[0]["with"] == {"code": "M3", "district": "Нура"}
    for s in swaps:
        plan = [Choice(s["with"]["code"], s["with"]["district"]) if (c.code, c.district) ==
                (s["replace"]["code"], s["replace"]["district"]) else c for c in choices]
        assert check_rules(plan) == []
        assert evaluate_plan(plan).score == pytest.approx(s["score_after"], abs=0.001)
        assert s["gain"] > 0 and s["score_after"] > base


def test_no_swaps_for_optimum() -> None:
    ranked, _ = best_measure_plans(top=1)
    assert suggest_swaps(list(ranked[0].choices)) == []


# ── режим сумм ──────────────────────────────────────────────────────────────


def _alloc(*v: float) -> dict[str, float]:
    return dict(zip(ds.SECTORS, v))


def test_caps_choose_five_whole_measures_and_match_reference_score() -> None:
    caps = _alloc(25, 15, 25, 15, 20)
    ranked, checked = best_measure_plans(top=1, caps=caps)
    assert checked > 0
    best = ranked[0]
    assert len(best.choices) == 5 and check_rules(list(best.choices)) == []
    assert best.score == pytest.approx(56.015, abs=0.001)
    assert evaluate_plan(best.choices).score == pytest.approx(best.score)
    for s in ds.SECTORS:
        assert sum(ch.measure.cost for ch in best.choices if ch.measure.sector == s) <= caps[s]


def test_caps_can_be_infeasible_and_global_optimum_is_reachable() -> None:
    assert best_measure_plans(top=1, caps=_alloc(100, 0, 0, 0, 0))[0] == []
    ranked, _ = best_measure_plans(top=1, caps=_alloc(52, 0, 30, 0, 16))
    assert ranked[0].score == pytest.approx(57.237, abs=0.001)
    assert ranked[0].cost == 98


def test_caps_search_equals_brute_force_on_small_catalog() -> None:
    codes = ("M1", "M2", "M6", "M9", "M10", "M11", "M12", "M14")
    caps = _alloc(22, 20, 10, 12, 30)
    ranked, n = ExactMeasuresOptimizer(codes=codes).search(top=1, caps=caps)
    scores = []
    for combo in itertools.combinations(codes, 5):
        local = [c for c in combo if not ds.MEASURES[c].citywide]
        for districts in itertools.product(ds.DISTRICTS, repeat=len(local)):
            by_code = dict(zip(local, districts))
            choices = [Choice(c, by_code.get(c, ds.CITY)) for c in combo]
            if check_rules(choices) or any(sum(ch.measure.cost for ch in choices if ch.measure.sector == s) > caps[s]
                                           for s in ds.SECTORS):
                continue
            scores.append(evaluate_plan(choices).score)
    assert n == len(scores) and ranked[0].score == pytest.approx(max(scores))
