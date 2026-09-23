"""CLI: python -m akim_sim [--measures ... | --alloc ... | --input file.json | --optimize | --catalog | --serve]."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from . import dataset as ds
from .api import ApiConfig, serve
from .inputs import AllocationDecision, Decision, MeasuresDecision, PlanValidationError, parse_decision
from .model import base_evaluation, parse_choice, validate_plan
from .optimizer import best_measure_plans
from .pipeline import AkimSimulator
from .report import render, render_catalog, render_optimum


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m akim_sim",
        description="Astana Akim Simulator «Аким на 5 часов» — официальная модель датасета + мультиагентный отчёт",
        epilog=("Примеры:\n"
                "  python -m akim_sim --measures M7:Нура M8:Нура M10:Нура M12 M5:Сарыарка\n"
                "  python -m akim_sim --alloc 25 15 25 15 20   (транспорт, экология, соцсфера, безопасность, сервисы)\n"
                "  python -m akim_sim --input examples/measures_example.json --json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("решение Акима")
    src.add_argument("-i", "--input", help="JSON-файл: {\"measures\": [...]} или {\"allocations\": {...}}")
    src.add_argument("--measures", nargs="+", metavar="КОД[:РАЙОН]", help="ровно 5 мер: M7:Нура … M12")
    src.add_argument("--alloc", nargs=5, type=float, metavar=("T", "E", "S", "B", "C"), help="5 сумм по сферам")
    src.add_argument("--budget", type=float, default=None, help=f"бюджет (по умолчанию {ds.BUDGET:g} {ds.UNIT})")
    src.add_argument("--statement", help="посыл Акима жителям")
    src.add_argument("--name", help="имя Акима для отчёта")
    src.add_argument("--interactive", action="store_true", help="интерактивный режим")
    act = p.add_argument_group("действия")
    act.add_argument("--validate", action="store_true", help="только проверить набор и вывести причины")
    act.add_argument("--optimize", action="store_true", help="официальный оптимум (полный перебор)")
    act.add_argument("--catalog", action="store_true", help="показать каталог мер")
    act.add_argument("--serve", action="store_true", help="запустить JSON API")
    act.add_argument("--host", default="127.0.0.1")
    act.add_argument("--port", type=int, default=8000)
    act.add_argument("--cors-origin", default=os.getenv("AKIM_CORS_ORIGIN") or None,
                     help="разрешённый Origin для CORS (по умолчанию CORS выключен)")
    act.add_argument("--api-token", default=os.getenv("AKIM_API_TOKEN") or None, help="Bearer-токен API")
    out = p.add_argument_group("вывод и агенты")
    out.add_argument("--offline", action="store_true", help="без LLM: rule-based агенты")
    out.add_argument("--model", help="модель OpenAI (перекрывает OPENAI_MODEL)")
    out.add_argument("--lang", choices=["ru", "kk", "en"], help="язык текстов LLM-агентов")
    out.add_argument("--json", action="store_true", help="печатать только JSON")
    out.add_argument("-o", "--out", help="сохранить JSON-отчёт в файл")
    out.add_argument("--no-color", action="store_true")
    return p


def _interactive() -> Decision:
    print("\n«АКИМ НА 5 ЧАСОВ» — бюджет 100 ед. Режим: 1 — 5 мер из каталога (официальный), 2 — суммы по сферам")
    mode = input("Режим [1/2]: ").strip() or "1"
    if mode == "2":
        alloc: dict[str, float] = {}
        left = ds.BUDGET
        for i, s in enumerate(ds.SECTORS, 1):
            while True:
                raw = input(f"  [{i}/5] {ds.SECTOR_NAMES[s]} — осталось {left:g}: ").strip()
                if raw == "" and i == len(ds.SECTORS):
                    val = left
                else:
                    try:
                        val = float(raw.replace(",", "."))
                    except ValueError:
                        print("    Введите число")
                        continue
                if not 0 <= val <= left + 1e-9:
                    print(f"    От 0 до {left:g}")
                    continue
                alloc[s] = val
                left -= val
                break
        return AllocationDecision(allocations=alloc)
    print(render_catalog(color=sys.stdout.isatty()))
    while True:
        items: list[str] = []
        spent = 0.0
        while len(items) < ds.N_DECISIONS:
            raw = input(f"  Решение {len(items) + 1}/5 (например M7:Нура или M12), потрачено {spent:g}: ").strip()
            choice, err = parse_choice(raw)
            if err:
                print("    " + err)
                continue
            items.append(raw)
            spent += choice.measure.cost  # type: ignore[union-attr]
        _, reasons = validate_plan(items)
        if not reasons:
            return MeasuresDecision(measures=items)
        print("Набор невалиден — Score не считается:")
        for r in reasons:
            print("  • " + r)
        print("Введите набор заново.")


def _decision(args: argparse.Namespace) -> Decision | None:
    extra: dict[str, Any] = {}
    if args.statement:
        extra["akim_statement"] = args.statement
    if args.name:
        extra["akim_name"] = args.name
    if args.budget is not None:
        extra["budget_total"] = args.budget
    if args.input:
        with open(args.input, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = {**data, **extra}
        return parse_decision(data)
    if args.measures:
        return MeasuresDecision(measures=args.measures, **extra)
    if args.alloc:
        return AllocationDecision(allocations=dict(zip(ds.SECTORS, args.alloc)), **extra)
    if args.interactive or sys.stdin.isatty():
        return _interactive()
    return None


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    color = not args.no_color and sys.stdout.isatty() and not os.getenv("NO_COLOR")

    if args.serve:
        serve(args.host, args.port, ApiConfig(offline=args.offline, model=args.model, lang=args.lang,
                                              cors_origin=args.cors_origin, api_token=args.api_token))
        return 0
    if args.catalog:
        print(render_catalog(color))
        return 0
    if args.optimize:
        ranked, n = best_measure_plans(top=5)
        print(render_optimum(ranked, n, base_evaluation().score, color))
        return 0
    try:
        decision = _decision(args)
        if decision is None:
            parser.print_help()
            return 2
        if args.validate:
            reasons = validate_plan(decision.measures, decision.budget_total)[1] \
                if isinstance(decision, MeasuresDecision) else []
            if isinstance(decision, AllocationDecision):
                ranked, _ = best_measure_plans(top=1, budget=decision.budget_total, caps=decision.allocations)
                if not ranked:
                    reasons = ["Лимиты не позволяют выбрать ровно 5 совместимых мер"]
            print(json.dumps({"valid": not reasons, "mode": decision.mode, "reasons": reasons}, ensure_ascii=False, indent=2))
            return 0 if not reasons else 2
        report = asyncio.run(AkimSimulator(offline=args.offline, model=args.model, lang=args.lang).run(decision))
    except PlanValidationError as exc:
        print("Набор невалиден — Score не считается. Причины:", file=sys.stderr)
        for r in exc.reasons:
            print(f"  • {r}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        for err in exc.errors():
            print(f"Ошибка входных данных: {err['msg'].removeprefix('Value error, ')}", file=sys.stderr)
        return 2
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Ошибка входных данных: {exc}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван.", file=sys.stderr)
        return 130
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report, color))
        if args.out:
            print(f"  JSON-отчёт сохранён: {args.out}")
    return 0
