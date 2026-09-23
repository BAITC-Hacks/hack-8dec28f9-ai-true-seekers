from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Iterator

import pytest

from akim_sim import dataset as ds
from akim_sim.api import ApiConfig, make_server

WORKED_EXAMPLE_ITEMS = [f"{c}:{d}" if d != ds.CITY else c for c, d in ds.WORKED_EXAMPLE]
OPTIMUM_ITEMS = ["M2", "M3:Нура", "M8:Нура", "M9:Нура", "M14"]


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тесты никогда не ходят в настоящий OpenAI: ключ из окружения/.env удаляется."""
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_MODEL_ORCHESTRATOR", "OPENAI_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)


class ApiClient:
    def __init__(self, base: str) -> None:
        self.base = base

    def request(self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None,
                raw: bytes | None = None) -> tuple[int, dict[str, str], Any]:
        data = raw if raw is not None else (json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
                return resp.status, dict(resp.headers), json.loads(payload) if payload else None
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            return exc.code, dict(exc.headers), json.loads(payload) if payload else None


def _start(cfg: ApiConfig) -> tuple[Any, ApiClient]:
    httpd = make_server("127.0.0.1", 0, cfg)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, ApiClient(f"http://127.0.0.1:{httpd.server_address[1]}")


@pytest.fixture
def api() -> Iterator[ApiClient]:
    httpd, client = _start(ApiConfig())
    yield client
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def api_factory() -> Iterator[Any]:
    servers = []

    def factory(**kwargs: Any) -> ApiClient:
        httpd, client = _start(ApiConfig(**kwargs))
        servers.append(httpd)
        return client

    yield factory
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()
