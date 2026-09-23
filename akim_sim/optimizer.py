"""Оптимизаторы.

ExactMeasuresOptimizer — полный перебор всех валидных наборов из 5 мер (≈694 тыс. при официальных правилах)
на предвычисленных векторах: вклад каждой меры в баллы районов линеен, нелинейны только min по районам и
N_crit, а для N_crit достаточно следить за «чувствительными» клетками (тем, что могут оказаться ниже 40).
Корректность упрощений проверяется при построении (клиппинг 0..100 не достигается) и тестами
(совпадение с эталонной функцией model.evaluate_plan).
"""

from __future__ import annotations

import heapq
import itertools
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping

from . import dataset as ds
from .model import Choice, check_rules, evaluate_plan

_DIDX = {d: i for i, d in enumerate(ds.DISTRICTS)}


@dataclass(frozen=True)
class RankedPlan:
    score: float
    choices: tuple[Choice, ...]

    @property
    def cost(self) -> float:
        return sum(ch.measure.cost for ch in self.choices)


class ExactMeasuresOptimizer:
    def __init__(self, weights: Mapping[str, float] | None = None, codes: Iterable[str] | None = None) -> None:
        self.weights = dict(weights or ds.WEIGHTS)
        self.codes = tuple(codes or ds.MEASURE_CODES)
        w = self.weights
        n = len(ds.DISTRICTS)
        self.pop = [ds.POPULATION[d] for d in ds.DISTRICTS]
        self.base_d = [sum(w[k] * ds.BASELINE[d][k] for k in ds.INDICATORS) for d in ds.DISTRICTS]

        # вклад каждой постановки (мера, район) в клетки и в баллы районов
        self.cell_delta: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
        for code in self.codes:
            m = ds.MEASURES[code]
            for dist in ((ds.CITY,) if m.citywide else ds.DISTRICTS):
                cells: dict[tuple[str, str], float] = {}
                for t in (ds.DISTRICTS if m.citywide else (dist,)):
                    for k, v in m.effects.items():
                        cells[(t, k)] = cells.get((t, k), 0.0) + v * m.realized_share
                self.cell_delta[(code, dist)] = cells
        self.syn_cells: dict[tuple[int, str], dict[tuple[str, str], float]] = {}
        for si, syn in enumerate(ds.SYNERGIES):
            for dist in ds.DISTRICTS:
                self.syn_cells[(si, dist)] = {(dist, syn.indicator): syn.bonus}

        # чувствительные клетки: те, что в каком-то наборе могут оказаться ниже порога
        neg = Counter()
        pos = Counter()
        for cells in list(self.cell_delta.values()) + list(self.syn_cells.values()):
            for cell, v in cells.items():
                (neg if v < 0 else pos)[cell] += v
        self.sensitive = [(d, k) for d in ds.DISTRICTS for k in ds.INDICATORS
                          if ds.BASELINE[d][k] + neg[(d, k)] < ds.CRIT_THRESHOLD]
        for d in ds.DISTRICTS:
            for k in ds.INDICATORS:
                hi = ds.BASELINE[d][k] + pos[(d, k)]
                lo = ds.BASELINE[d][k] + neg[(d, k)]
                if hi > 100.0 or lo < 0.0:
                    raise AssertionError("Клиппинг 0..100 достижим — быстрый перебор неприменим")
        sidx = {c: i for i, c in enumerate(self.sensitive)}
        self.sens_base = [ds.BASELINE[d][k] for d, k in self.sensitive]

        def compile_(cells: dict[tuple[str, str], float]) -> tuple[tuple[float, ...], tuple[tuple[int, float], ...]]:
            dv = [0.0] * n
            for (t, k), v in cells.items():
                dv[_DIDX[t]] += w[k] * v
            sv = tuple((sidx[c], v) for c, v in cells.items() if c in sidx)
            return tuple(dv), sv

        self.vec = {p: compile_(c) for p, c in self.cell_delta.items()}
        self.syn_vec = {p: compile_(c) for p, c in self.syn_cells.items()}

    # ── оценка одного набора на векторах (используется и перебором) ─────────
    def _score(self, placements: list[tuple[str, str]], syn_ids: list[tuple[int, str]]) -> float:
        dv = list(self.base_d)
        sv = list(self.sens_base)
        for p in placements:
            v, s = self.vec[p]
            for i in range(len(dv)):
                dv[i] += v[i]
            for j, x in s:
                sv[j] += x
        for key in syn_ids:
            v, s = self.syn_vec[key]
            for i in range(len(dv)):
                dv[i] += v[i]
            for j, x in s:
                sv[j] += x
        ncrit = sum(1 for x in sv if x < ds.CRIT_THRESHOLD)
        avg = sum(p * x for p, x in zip(self.pop, dv))
        return ds.W_AVG * avg + ds.W_MIN * min(dv) - ds.CRIT_PENALTY * ncrit

    def score_choices(self, choices: Iterable[Choice]) -> float:
        """Быстрая оценка набора (для сверки с эталонной функцией в тестах)."""
        choices = list(choices)
        codes = {c.code for c in choices}
        where = {c.code: c.district for c in choices}
        syn_ids = [(si, where[syn.anchor]) for si, syn in enumerate(ds.SYNERGIES) if syn.a in codes and syn.b in codes]
        return self._score([(c.code, c.district) for c in choices], syn_ids)

    def search(self, top: int = 5, budget: float = ds.BUDGET,
               caps: Mapping[str, float] | None = None) -> tuple[list[RankedPlan], int]:
        """Полный перебор. Возвращает (лучшие `top` наборов, число валидных наборов)."""
        heap: list[tuple[float, int, tuple[Choice, ...]]] = []
        counter = itertools.count()
        n_valid = 0
        n = len(ds.DISTRICTS)
        pop = self.pop
        crit = ds.CRIT_THRESHOLD
        glob = [(a, b) for a, b, _ in ds.INCOMPATIBLE_GLOBAL]
        same = [(a, b) for a, b, _ in ds.INCOMPATIBLE_SAME_DISTRICT]
        for combo in itertools.combinations(self.codes, ds.N_DECISIONS):
            ms = [ds.MEASURES[c] for c in combo]
            if sum(m.cost for m in ms) > budget + 1e-9:
                continue
            if caps is not None and any(sum(m.cost for m in ms if m.sector == s) > caps[s] + 1e-9
                                        for s in ds.SECTORS):
                continue
            if max(Counter(m.sector for m in ms).values()) > ds.MAX_PER_DIRECTION:
                continue
            cs = set(combo)
            if any(a in cs and b in cs for a, b in glob):
                continue
            city = [c for c in combo if ds.MEASURES[c].citywide]
            local = [c for c in combo if not ds.MEASURES[c].citywide]
            dv0 = list(self.base_d)
            sv0 = list(self.sens_base)
            for c in city:
                v, s = self.vec[(c, ds.CITY)]
                for i in range(n):
                    dv0[i] += v[i]
                for j, x in s:
                    sv0[j] += x
            same_pairs = [(local.index(a), local.index(b)) for a, b in same if a in cs and b in cs]
            active_syn = [(si, syn) for si, syn in enumerate(ds.SYNERGIES) if syn.a in cs and syn.b in cs]
            local_vecs = [[self.vec[(c, d)] for d in ds.DISTRICTS] for c in local]
            for didx in itertools.product(range(n), repeat=len(local)):
                if any(didx[i] == didx[j] for i, j in same_pairs):
                    continue
                n_valid += 1
                dv = dv0[:]
                sv = sv0[:]
                for li, di in enumerate(didx):
                    v, s = local_vecs[li][di]
                    for i in range(n):
                        dv[i] += v[i]
                    for j, x in s:
                        sv[j] += x
                for si, syn in active_syn:
                    anchor_d = ds.DISTRICTS[didx[local.index(syn.anchor)]]
                    v, s = self.syn_vec[(si, anchor_d)]
                    for i in range(n):
                        dv[i] += v[i]
                    for j, x in s:
                        sv[j] += x
                ncrit = 0
                for x in sv:
                    if x < crit:
                        ncrit += 1
                score = ds.W_AVG * (pop[0] * dv[0] + pop[1] * dv[1] + pop[2] * dv[2] + pop[3] * dv[3] + pop[4] * dv[4]) \
                    + ds.W_MIN * min(dv) - ds.CRIT_PENALTY * ncrit
                if len(heap) < top or score > heap[0][0]:
                    choices = tuple(Choice(c, ds.CITY) for c in city) + tuple(
                        Choice(c, ds.DISTRICTS[di]) for c, di in zip(local, didx))
                    item = (score, next(counter), choices)
                    if len(heap) < top:
                        heapq.heappush(heap, item)
                    else:
                        heapq.heapreplace(heap, item)
        ranked = [RankedPlan(s, _canonical(ch)) for s, _, ch in sorted(heap, key=lambda t: -t[0])]
        return ranked, n_valid


def _canonical(choices: Iterable[Choice]) -> tuple[Choice, ...]:
    order = {c: i for i, c in enumerate(ds.MEASURE_CODES)}
    return tuple(sorted(choices, key=lambda ch: order[ch.code]))


@lru_cache(maxsize=16)
def _cached_search(weights_key: tuple[tuple[str, float], ...], budget: float, top: int,
                   caps_key: tuple[float, ...] | None) -> tuple[tuple[RankedPlan, ...], int]:
    opt = ExactMeasuresOptimizer(dict(weights_key) if weights_key else None)
    caps = dict(zip(ds.SECTORS, caps_key)) if caps_key is not None else None
    ranked, n = opt.search(top=top, budget=budget, caps=caps)
    return tuple(ranked), n


def best_measure_plans(top: int = 5, budget: float = ds.BUDGET,
                       weights: Mapping[str, float] | None = None,
                       caps: Mapping[str, float] | None = None) -> tuple[list[RankedPlan], int]:
    key = tuple(sorted(weights.items())) if weights else ()
    caps_key = tuple(float(caps[s]) for s in ds.SECTORS) if caps is not None else None
    ranked, n = _cached_search(key, float(budget), max(top, 5), caps_key)
    return list(ranked[:top]), n


def suggest_swaps(choices: list[Choice], top: int = 3, budget: float = ds.BUDGET) -> list[dict]:
    """Одиночные замены (другая мера или другой район), которые сохраняют валидность и повышают Score."""
    base = evaluate_plan(choices).score
    found: dict[tuple[str, str, str], dict] = {}
    for i, old in enumerate(choices):
        rest = choices[:i] + choices[i + 1:]
        used = {c.code for c in rest}
        for code in ds.MEASURE_CODES:
            if code in used:
                continue
            m = ds.MEASURES[code]
            for dist in ((ds.CITY,) if m.citywide else ds.DISTRICTS):
                new = Choice(code, dist)
                if new == old:
                    continue
                plan = rest[:i] + [new] + rest[i:]
                if check_rules(plan, budget):
                    continue
                score = evaluate_plan(plan).score
                gain = score - base
                if gain > 0.005:
                    key = (old.code, new.code, new.district)
                    found[key] = {"replace": {"code": old.code, "district": old.district},
                                  "with": {"code": new.code, "district": new.district},
                                  "gain": round(gain, 3), "score_after": round(score, 3),
                                  "cost_after": sum(c.measure.cost for c in plan)}
    ranked = sorted(found.values(), key=lambda r: -r["gain"])
    out, seen_old = [], set()
    for r in ranked:  # разнообразие: по одной лучшей замене на каждую исходную меру
        if r["replace"]["code"] in seen_old:
            continue
        seen_old.add(r["replace"]["code"])
        out.append(r)
        if len(out) >= top:
            break
    return out
