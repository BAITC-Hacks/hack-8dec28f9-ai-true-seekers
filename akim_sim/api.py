"""JSON API на стандартной библиотеке (без веб-интерфейса).

GET  /health                         — проверка (без авторизации)
GET  /config                         — датасет, правила, каталог
GET  /optimal?mode=measures|allocation&top=5
POST /validate                       — проверка набора: {"valid": bool, "reasons": [...]}
POST /score                          — только движок (быстро, без LLM)
POST /simulate[?offline=1]           — полный прогон с агентами

Безопасность: по умолчанию слушает 127.0.0.1; CORS выключен, пока не задан --cors-origin;
при заданном токене (--api-token / AKIM_API_TOKEN) все эндпоинты, кроме /health, требуют
заголовок «Authorization: Bearer <токен>»; тело запроса ограничено 1 МБ.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from . import __version__
from . import dataset as ds
from .engine import analyze, config
from .inputs import AllocationDecision, MeasuresDecision, PlanValidationError, parse_decision
from .model import validate_plan
from .optimizer import best_measure_plans
from .pipeline import AkimSimulator

MAX_BODY = 1_000_000


@dataclass
class ApiConfig:
    offline: bool = False
    model: str | None = None
    lang: str | None = None
    cors_origin: str | None = None
    api_token: str | None = None


def _errors(exc: ValidationError) -> list[str]:
    return [err["msg"].removeprefix("Value error, ") for err in exc.errors()]


def make_handler(cfg: ApiConfig) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"AkimSimulator/{__version__}"

        def _send(self, code: int, obj: Any) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if cfg.cors_origin:
                self.send_header("Access-Control-Allow-Origin", cfg.cors_origin)
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not cfg.api_token:
                return True
            header = self.headers.get("Authorization", "")
            ok = header.startswith("Bearer ") and hmac.compare_digest(header[7:].strip(), cfg.api_token)
            if not ok:
                self._send(401, {"error": "unauthorized"})
            return ok

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > MAX_BODY:
                raise OverflowError
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(204, {})

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/health":
                self._send(200, {"status": "ok", "version": __version__})
                return
            if not self._authorized():
                return
            q = parse_qs(url.query)
            if url.path == "/config":
                self._send(200, config())
            elif url.path == "/optimal":
                mode = q.get("mode", ["measures"])[0]
                try:
                    top = max(1, min(20, int(q.get("top", ["5"])[0])))
                except ValueError:
                    self._send(400, {"error": "top должен быть целым числом"})
                    return
                if mode == "measures":
                    ranked, n = best_measure_plans(top=top)
                    self._send(200, {"mode": "measures", "valid_plans": n, "plans": [
                        {"score": round(r.score, 3), "cost": r.cost,
                         "measures": [{"code": ch.code, "district": ch.district} for ch in r.choices]} for r in ranked]})
                elif mode == "allocation":
                    ranked, _ = best_measure_plans(top=1)
                    optimal = ranked[0]
                    caps = {s: sum(ch.measure.cost for ch in optimal.choices if ch.measure.sector == s)
                            for s in ds.SECTORS}
                    self._send(200, {"mode": "allocation", "allocations": caps,
                                     "score": round(optimal.score, 3), "cost": optimal.cost,
                                     "note": "Лимиты, допускающие точный оптимум пяти мер"})
                else:
                    self._send(400, {"error": "mode: measures | allocation"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if not self._authorized():
                return
            try:
                data = self._body()
            except OverflowError:
                self.close_connection = True  # тело не читаем — закрываем соединение после ответа
                self._send(413, {"error": "тело запроса больше 1 МБ"})
                return
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"некорректный JSON: {exc}"})
                return
            if url.path == "/validate":
                try:
                    decision = parse_decision(data)
                except ValidationError as exc:
                    self._send(200, {"valid": False, "reasons": _errors(exc)})
                    return
                except ValueError as exc:
                    self._send(200, {"valid": False, "reasons": [str(exc)]})
                    return
                reasons: list[str] = []
                if isinstance(decision, MeasuresDecision):
                    _, reasons = validate_plan(decision.measures, decision.budget_total)
                if isinstance(decision, AllocationDecision):
                    ranked, _ = best_measure_plans(top=1, budget=decision.budget_total, caps=decision.allocations)
                    if not ranked:
                        reasons = ["Лимиты не позволяют выбрать ровно 5 совместимых мер"]
                self._send(200, {"valid": not reasons, "mode": decision.mode, "reasons": reasons})
                return
            if url.path not in ("/score", "/simulate"):
                self._send(404, {"error": "not found"})
                return
            try:
                decision = parse_decision(data)
                if url.path == "/score":
                    self._send(200, analyze(decision))
                else:
                    offline = cfg.offline or q_flag(url.query, "offline")
                    sim = AkimSimulator(offline=offline, model=cfg.model, lang=cfg.lang)
                    self._send(200, asyncio.run(sim.run(decision)))
            except PlanValidationError as exc:
                self._send(422, {"error": "invalid_plan", "score": None, "reasons": exc.reasons})
            except ValidationError as exc:
                self._send(422, {"error": "validation_error", "reasons": _errors(exc)})
            except ValueError as exc:
                self._send(422, {"error": "validation_error", "reasons": [str(exc)]})
            except Exception as exc:  # pragma: no cover — не отдаём трейсбек наружу
                print(f"[api] internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._send(500, {"error": "internal_error"})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[api] {self.address_string()} {fmt % args}", file=sys.stderr)

    return Handler


def q_flag(query: str, name: str) -> bool:
    return parse_qs(query).get(name, ["0"])[0].lower() in ("1", "true", "yes")


def make_server(host: str, port: int, cfg: ApiConfig) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(cfg))


def serve(host: str, port: int, cfg: ApiConfig) -> None:
    if host not in ("127.0.0.1", "localhost", "::1") and not cfg.api_token:
        print("[api] ВНИМАНИЕ: сервер слушает не только localhost и без токена. Задайте --api-token.", file=sys.stderr)
    httpd = make_server(host, port, cfg)

    def warm_up() -> None:  # оптимумы считаются один раз (~5 с) и кэшируются
        best_measure_plans(top=5)
        print("[api] кэш оптимизаторов прогрет", file=sys.stderr)

    threading.Thread(target=warm_up, daemon=True).start()
    print(f"Akim Simulator API: http://{host}:{port}  (GET /health · /config · /optimal · "
          "POST /validate · /score · /simulate)", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
