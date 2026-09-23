"""Мок OpenAI-совместимого сервера для тестов LLM-ветки (без сети и ключей).

Режимы:
  ok             — валидный ответ по схеме
  reject_schema  — 400 на response_format=json_schema (проверка перехода в JSON mode)
  bad_first      — первый ответ на каждую схему — битый JSON (проверка repair-запроса)
  fail_all       — 500 на всё (проверка rule-based fallback)
  hallucinate    — в executive_summary вставлены выдуманные числа (проверка guardrails)

Сервер также проверяет, что присланная JSON Schema удовлетворяет правилам strict-режима Structured Outputs.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def strict_violations(node: Any, path: str = "$") -> list[str]:
    errs: list[str] = []
    if isinstance(node, dict):
        for bad in ("title", "default", "$ref", "$defs", "examples"):
            if bad in node:
                errs.append(f"{path}: запрещённый ключ {bad}")
        if node.get("type") == "object":
            props = node.get("properties", {})
            if node.get("additionalProperties") is not False:
                errs.append(f"{path}: additionalProperties должен быть false")
            if sorted(node.get("required", [])) != sorted(props):
                errs.append(f"{path}: required должен перечислять все свойства")
            for k, v in props.items():
                errs += strict_violations(v, f"{path}.{k}")
        if node.get("type") == "array":
            errs += strict_violations(node.get("items", {}), f"{path}[]")
    return errs


class MockOpenAI:
    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.calls: list[dict[str, Any]] = []
        self.schema_errors: list[str] = []
        self._seen: dict[str, int] = {}
        self._lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def __enter__(self) -> "MockOpenAI":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def _gen(self, schema: dict[str, Any], path: str = "", idx: int = 0) -> Any:
        if "enum" in schema:
            return schema["enum"][idx % len(schema["enum"])]
        t = schema.get("type")
        if t == "object":
            return {k: self._gen(v, f"{path}.{k}", idx) for k, v in schema["properties"].items()}
        if t == "array":
            n = 5 if path.endswith("assessments") else 3
            return [self._gen(schema["items"], path, i) for i in range(n)]
        if t == "integer":
            return idx + 1
        if t == "number":
            return 1.0
        text = f"Мок-ответ для поля {path.lstrip('.')}"
        if self.mode == "hallucinate" and path.endswith("executive_summary"):
            text += " Score вырастет до 88.8, а пробки упадут на 42.5%."
        return text

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, code: int, obj: Any) -> None:
                body = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                rf = req.get("response_format", {})
                name = rf.get("json_schema", {}).get("name", "json_object")
                with mock._lock:
                    mock.calls.append({"type": rf.get("type"), "name": name, "model": req.get("model")})
                if mock.mode == "fail_all":
                    self._send(500, {"error": {"message": "mock outage", "type": "server_error"}})
                    return
                if rf.get("type") == "json_schema":
                    if mock.mode == "reject_schema":
                        self._send(400, {"error": {"message": "response_format json_schema is not supported",
                                                   "type": "invalid_request_error"}})
                        return
                    schema = rf["json_schema"]["schema"]
                    errs = strict_violations(schema)
                    if errs:
                        mock.schema_errors += errs
                        self._send(400, {"error": {"message": "Invalid schema: " + "; ".join(errs[:3]),
                                                   "type": "invalid_request_error"}})
                        return
                elif rf.get("type") == "json_object":
                    system = req["messages"][0]["content"]
                    if "JSON" not in system:
                        self._send(400, {"error": {"message": "messages must contain the word json"}})
                        return
                    schema = json.loads(system.split("JSON Schema:\n", 1)[1])
                    name = "json_object:" + ",".join(list(schema["properties"])[:2])
                else:
                    self._send(400, {"error": {"message": "response_format required"}})
                    return
                time.sleep(0.05)
                content = json.dumps(mock._gen(schema), ensure_ascii=False)
                with mock._lock:
                    mock._seen[name] = mock._seen.get(name, 0) + 1
                    first = mock._seen[name] == 1
                if mock.mode == "bad_first" and first:
                    content = '{"headline": "обрезанный ответ", '
                self._send(200, {
                    "id": "chatcmpl-mock", "object": "chat.completion", "created": int(time.time()),
                    "model": req.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content, "refusal": None}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                })

        return Handler
