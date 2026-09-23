"""Консольный отчёт (CLI). Веб-интерфейса в этой версии нет — только CLI и JSON API."""

from __future__ import annotations

import textwrap
from typing import Any

from . import dataset as ds

WIDTH = 80
LEVEL_RU = {"low": "низкий", "medium": "средний", "high": "ВЫСОКИЙ", "critical": "КРИТИЧНО"}
VERDICT_RU = {"critical": "КРИТИЧНО", "weak": "слабо", "ok": "норма", "strong": "сильно",
              "underfunded": "недофин.", "balanced": "баланс", "overfunded": "избыток"}
SEASON_RU = {"winter": "Зима", "spring": "Весна", "summer": "Лето", "autumn": "Осень"}
MARK = {"positive": "(+)", "neutral": "(=)", "negative": "(−)"}


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _c(self, code: str, t: str) -> str:
        return f"\033[{code}m{t}\033[0m" if self.enabled else t

    def bold(self, t: str) -> str:
        return self._c("1", t)

    def dim(self, t: str) -> str:
        return self._c("2", t)

    def red(self, t: str) -> str:
        return self._c("31", t)

    def green(self, t: str) -> str:
        return self._c("32", t)

    def yellow(self, t: str) -> str:
        return self._c("33", t)

    def cyan(self, t: str) -> str:
        return self._c("36", t)


def _wrap(text: str, indent: int = 2, first: str = "") -> list[str]:
    pad = " " * indent
    return textwrap.wrap(text, width=WIDTH, initial_indent=pad + first,
                         subsequent_indent=pad + ("  " if first else "")) or [pad + first]


def _bar(value: float, lo: float, hi: float, width: int = 30) -> str:
    frac = 0.0 if hi <= lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = int(round(width * frac))
    return "█" * n + "░" * (width - n)


def render(report: dict[str, Any], color: bool = True) -> str:
    c = Palette(color)
    e = report["engine"]
    L: list[str] = []

    def section(title: str) -> None:
        L.append("")
        L.append(c.bold(c.cyan(f"── {title} " + "─" * max(0, WIDTH - len(title) - 4))))

    def lvl(level: str, text: str) -> str:
        return {"high": c.red, "critical": c.red, "medium": c.yellow}.get(level, c.green)(text)

    mode_ru = "режим мер (официальный, ровно 5 мер)" if e["mode"] == "measures" else "пять лимитов, подбор пяти целых мер"
    L.append(c.bold("╔" + "═" * (WIDTH - 2) + "╗"))
    L.append(c.bold("║ " + "ASTANA CITY SCORE · «Аким на 5 часов» · официальная формула датасета".ljust(WIDTH - 4) + " ║"))
    L.append(c.bold("╚" + "═" * (WIDTH - 2) + "╝"))
    ref = e["reference"]
    score_txt = "%.3f" % e["score"]
    L.append(f"  Score: {c.bold(score_txt)}  база {e['base_score']:.3f} → {e['delta_vs_base']:+.3f}   "
             f"{c.bold(e['grade']['grade'])} — {e['grade']['label']}")
    L.append(f"  Потенциал: {_bar(e['score'], e['base_score'], ref['score'], 34)} {e['potential_realized_pct']:.1f}%")
    L += _wrap(c.dim(f"Ориентир: {ref['label']} — {ref['score']:.3f}. Режим: {mode_ru}."))

    if e["mode"] == "allocation":
        section("Пять бюджетных лимитов и фактический расход")
        L.append(c.dim(f"  {'Сфера':<27}{'Лимит':>8}{'Расход':>9}{'Остаток':>9}"))
        for sector in e["sectors"]:
            L.append(f"  {sector['name']:<27}{sector['cap']:>8g}{sector['spent']:>9g}{sector['unused_cap']:>9g}")
        L.append(f"  Допустимых наборов в этих лимитах: {e['feasible_portfolios_checked']}")
        L += _wrap(c.dim(e["selection_policy"]))
    section("Выбранные пять мер")
    L.append(c.dim(f"  {'Мера':<44}{'Район':<10}{'Цена':>5}{'Лаг':>4}{'Реализ.':>8}{'Вклад':>8}"))
    for p in e["plan"]:
        name = f"{p['code']} {p['name']}"[:43]
        L.append(f"  {name:<44}{p['district']:<10}{p['cost']:>5g}{p['lag']:>4}"
                 f"{100 * p['realized_share']:>7.1f}%{p['contribution']:>8.3f}")
    L.append(f"  Стоимость: {e['cost_total']:g} из {e['budget_total']:g} {ds.UNIT}"
             + (c.yellow(f" · не использовано {e['unspent']:g}") if e["unspent"] > 0 else ""))
    L.append(c.dim("  Вклад — насколько упадёт Score, если убрать меру; реализация — (8 − лаг)/8 за горизонт."))

    section("Районы (балл D_d по официальной формуле)")
    for d in e["districts"]:
        mark = c.red(" ← слабейший") if d["is_weakest"] else ""
        L.append(f"  {d['name']:<10}{d['population']:>5.2f}  {d['base_score']:7.3f} → {d['score']:7.3f}  "
                 f"({d['delta']:+.3f}){mark}")
    crit = e["critical_cells"]
    L.append(f"  Пары ниже 40 (N_crit): {e['n_crit']} (в базе {e['n_crit_base']})"
             + (": " + ", ".join(f"{x['district']} {x['indicator']} {x['value']:.2f}" for x in crit) if crit else ""))
    syn = [f"{s['pair']}: " + {"active": c.green("собрана"), "missed": c.yellow("упущена"), "blocked": c.dim("заблокирована"),
                                  "unused": "—"}[s["status"]]
           for s in e["synergies"]]
    L.append("  Синергии: " + " · ".join(syn))
    if e["moves"]:
        L.append(c.bold("  Проверенные движком ходы:"))
        for m in e["moves"]:
            L += _wrap(f"{m['text']} → +{m['gain']:.3f} (Score {m['score_after']:.3f})", indent=4, first="• ")

    section("Риски — МОДЕЛЬНЫЕ ОЦЕНКИ (прокси 0–100, не вероятности)")
    for r in e["risk_proxies"].values():
        L.append(f"  {r['label'][:52]:<53}{lvl(r['level'], '%5.1f' % r['value'])}/100  {r['worst_district']}")
    L.append(c.dim("  Анализ чувствительности (перевзвешивание показателей, не симуляция событий):"))
    for s in e["sensitivity"]:
        L.append(f"    {s['name']:<18} прирост к базе {s['gain_vs_base']:+.3f} ({s['gain_change']:+.3f} к штатным весам)")
    for d in e["disclaimers"]:
        L += _wrap(c.dim("⚠ " + d))

    agents = report["agents"]

    def header(key: str) -> None:
        a = agents[key]
        tag = (f"LLM {a['model']} · {a['latency_ms'] / 1000:.1f} с" if a["source"] == "llm"
               else "rule-based" if a["source"] == "rules" else "rule-based fallback")
        section(f"{a['agent']}  [{tag}]")

    ao = agents["analytics"]["output"]
    header("analytics")
    L += _wrap(c.bold(ao["headline"]))
    L += _wrap(ao["score_explanation"])
    for it in ao["issues"]:
        L += _wrap(it["issue"], indent=4, first=lvl(it["severity"], f"[{LEVEL_RU[it['severity']]}] "))
    L += _wrap("Эффективность: " + ao["efficiency"])
    L += _wrap("Синергии: " + ao["synergy"])

    uo = agents["urban_expert"]["output"]
    header("urban_expert")
    L += _wrap(c.bold(uo["headline"]))
    for a in uo["district_assessments"]:
        L += _wrap(a["comment"], indent=4, first=lvl({"critical": "critical", "weak": "medium"}.get(a["verdict"], "low"),
                                                      f"{a['district']} [{VERDICT_RU[a['verdict']]}]: "))
    for sr in uo["seasonal_risks"]:
        L += _wrap(f"{sr['risk']} → {sr['impact']}", indent=4, first=f"{SEASON_RU[sr['season']]}: ")
    L.append("  Меры каталога к рассмотрению:")
    for p in uo["priority_projects"]:
        name = ds.MEASURES[p["code"]].name
        L += _wrap(f"{name} — {p['rationale']}", indent=4, first=f"• {p['code']} ({p['district']}): ")
    L += _wrap(c.dim(uo["local_insight"]))

    so = agents["social_risk"]["output"]
    header("social_risk")
    L += _wrap(c.bold(so["headline"]))
    L += _wrap(so["public_mood"])
    L.append(c.dim("  Симулированные реакции (вымышленные жители, данных соцсетей нет):"))
    for p in so["simulated_posts"]:
        mark = {"positive": c.green, "negative": c.red}.get(p["sentiment"], c.yellow)(MARK[p["sentiment"]])
        L += _wrap(f"«{p['text']}» — {p['persona']}, {p['platform']}", indent=4, first=f"{mark} ")
    for r in so["risks"]:
        L += _wrap(f"{r['name']}: {r['trigger']}. Мера: {r['mitigation']}", indent=4,
                   first=lvl(r["level"], f"[{LEVEL_RU[r['level']]}] "))
    L += _wrap("Уязвимые группы: " + "; ".join(so["vulnerable_groups"]))
    L += _wrap("Коммуникация: " + so["communication_advice"])

    fo = report["final_report"]
    header("orchestrator")
    L += _wrap(c.bold(fo["title"]))
    L += _wrap(fo["executive_summary"])
    L += _wrap(c.bold("Вердикт: ") + fo["verdict"])
    L += _wrap(c.bold("Сравнение наборов: ") + fo["set_comparison"])
    L.append("  Консенсус агентов:")
    for x in fo["consensus"]:
        L += _wrap(x, indent=4, first="✓ ")
    L.append("  Противоречия и их разрешение:")
    for d in fo["disagreements"]:
        L += _wrap(f"{d['topic']}. {d['positions']} → Решение: {d['resolution']}", indent=4, first="⚖ ")
    L.append(c.bold("  Рекомендации:"))
    for r in fo["recommendations"]:
        L += _wrap(f"{r['action']}. {r['rationale']} Эффект: {c.green(r['expected_effect'])}", indent=4,
                   first=f"{r['priority']}. ")
    L.append("  Дорожная карта (с учётом лагов):")
    for step in fo["roadmap"]:
        L += _wrap(step, indent=4, first="→ ")
    L += _wrap(c.bold(fo["closing_statement"]))

    g = report["guardrails"]
    meta = report["meta"]
    section("Трассировка")
    L += _wrap(" · ".join(f"{k}: {v['source']} {v['latency_ms'] / 1000:.1f}с" for k, v in agents.items()))
    L.append(f"  Движок: {meta['latency_ms']['engine']} мс · всего: {meta['latency_ms']['total'] / 1000:.1f} с · "
             f"токены: {meta['usage']['total_tokens']} · агенты: {meta['agents_mode']}")
    if g["status"] == "pass":
        L.append("  Guardrails: " + c.green(f"PASS — {g['checked_numbers']} чисел в текстах агентов привязаны к ключам движка"))
    else:
        bad = [f"{u['agent']}: {u['value']}" for u in g["unverified"][:6]] + \
              [f"{u['agent']}: код {u['code']}" for u in g["unknown_codes"][:3]]
        L.append("  Guardrails: " + c.yellow("WARN — без источника в движке: " + ", ".join(bad)))
    L.append("")
    return "\n".join(L)


def render_catalog(color: bool = True) -> str:
    c = Palette(color)
    L = [c.bold("Каталог мероприятий датасета (бюджет 100 ед., ровно 5 мер, ≤2 на направление)"), ""]
    L.append(c.dim(f"  {'Код':<5}{'Мера':<42}{'Сфера':<12}{'Цена':>5}{'Лаг':>4}  {'Масштаб':<7} Эффекты"))
    for m in ds.MEASURES.values():
        eff = ", ".join(f"{k}{v:+g}" for k, v in m.effects.items())
        L.append(f"  {m.code:<5}{m.name[:41]:<42}{ds.SECTOR_NAMES[m.sector][:11]:<12}{m.cost:>5g}{m.lag:>4}  {m.scope:<7} {eff}")
    L.append("")
    L.append("  Синергии: " + "; ".join(f"{s.a}+{s.b} → {s.indicator} +{s.bonus:g} (в районе {s.anchor})" for s in ds.SYNERGIES))
    L.append("  Несовместимы: " + "; ".join(f"{a}/{b} — {r}" for a, b, r in ds.INCOMPATIBLE_GLOBAL + ds.INCOMPATIBLE_SAME_DISTRICT))
    return "\n".join(L)


def render_optimum(ranked: list[Any], n_valid: int, base: float, color: bool = True) -> str:
    c = Palette(color)
    L = [c.bold(f"Официальный оптимум: полный перебор {n_valid} валидных наборов (база {base:.3f})"), ""]
    for i, rp in enumerate(ranked, 1):
        L.append(f"  {i}. {rp.score:.3f}  ({rp.cost:g} {ds.UNIT})  " + ", ".join(ch.label() for ch in rp.choices))
    return "\n".join(L)
