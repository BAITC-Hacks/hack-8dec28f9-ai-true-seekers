"""Guardrails: каждое число в тексте агента должно иметь источник в отчёте движка, входе или датасете.

Для каждого подтверждённого числа возвращается путь к ключу-источнику (provenance), например
"engine.weakest.score". Числа без источника попадают в unverified; коды мер вне каталога — в unknown_codes.
"""

from __future__ import annotations

import bisect
import re
from typing import Any, Iterable, Mapping

from . import dataset as ds

NUM_RE = re.compile(r"(?<![\w.,])[−-]?\d+(?:[.,]\d+)?")
CODE_RE = re.compile(r"(?<![\w])[MМ](\d{1,3})(?!\d)")
# диапазоны кварталов/дней плана («Квартал 1», «Дни 30–60», «Q2») — не метрики
_RANGE_RE = re.compile(
    r"(?i)(?:дн(?:и|я|ей)|день|days?|кварт\w*|квартал\w*|q)\s*\d+(?:\s*[–—-]\s*\d+)?"
    r"|\d+\s*[–—-]\s*\d+\s*(?:дн\w*|days?|кварт\w*)")
SKIP_INT_MAX = 5  # целые 0..5 — счётные слова («3 шага», «5 мер»), не метрики
ABS_TOL = 0.051
REL_TOL = 0.005


def _variants(v: float) -> set[float]:
    out = {v, abs(v), round(v, 1), round(v, 2), round(v, 3), float(round(v))}
    if abs(v) <= 1.0:
        out |= {round(100 * v, 1), round(100 * abs(v), 1), round(100 * v, 2)}
    return {abs(x) for x in out}


def build_reference(sources: Mapping[str, Any]) -> tuple[list[float], list[str]]:
    """Плоский отсортированный индекс «значение → путь». sources: {"engine": {...}, "input": {...}, ...}."""
    pairs: list[tuple[float, str]] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            for x in _variants(float(obj)):
                pairs.append((x, path))
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                walk(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            for m in NUM_RE.findall(obj):
                try:
                    pairs.append((abs(float(m.replace(",", ".").replace("−", "-"))), f"{path}~text"))
                except ValueError:
                    pass

    for name, obj in sources.items():
        walk(obj, name)
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def lookup(value: float, values: list[float], paths: list[str]) -> str | None:
    tol = max(ABS_TOL, REL_TOL * abs(value))
    lo = bisect.bisect_left(values, value - tol)
    hi = bisect.bisect_right(values, value + tol)
    if lo >= hi:
        return None
    # при равной точности — самый канонический источник: не из текста, с самым коротким путём
    best = min(range(lo, hi), key=lambda i: (round(abs(values[i] - value), 9), "~text" in paths[i], len(paths[i])))
    return paths[best]


def iter_strings(obj: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from iter_strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_strings(v, f"{path}[{i}]")


def check_texts(outputs: Mapping[str, Any], values: list[float], paths: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"checked_numbers": 0, "unverified": [], "unknown_codes": [], "agents": {}}
    for agent, output in outputs.items():
        checked, provenance, unverified, unknown = 0, [], [], []
        for field_path, text in iter_strings(output):
            for code in CODE_RE.findall(text):
                if f"M{int(code)}" not in ds.MEASURES:
                    unknown.append({"code": f"M{code}", "field": field_path})
            clean = _RANGE_RE.sub(" ", CODE_RE.sub(" ", text))
            for m in NUM_RE.findall(clean):
                try:
                    val = abs(float(m.replace(",", ".").replace("−", "-")))
                except ValueError:
                    continue
                if val.is_integer() and val <= SKIP_INT_MAX:
                    continue
                checked += 1
                src = lookup(val, values, paths)
                if src is None:
                    unverified.append({"value": m, "field": field_path})
                else:
                    provenance.append({"value": m, "field": field_path, "source": src})
        report["agents"][agent] = {"checked": checked, "unverified": unverified, "unknown_codes": unknown,
                                   "provenance": provenance}
        report["checked_numbers"] += checked
        report["unverified"] += [{"agent": agent, **u} for u in unverified]
        report["unknown_codes"] += [{"agent": agent, **u} for u in unknown]
    report["status"] = "pass" if not report["unverified"] and not report["unknown_codes"] else "warn"
    return report


def fact_check(outputs: Mapping[str, Any], engine: Mapping[str, Any], decision: Mapping[str, Any],
               dataset_config: Mapping[str, Any]) -> dict[str, Any]:
    values, paths = build_reference({"engine": engine, "input": decision, "dataset": dataset_config})
    return check_texts(outputs, values, paths)
