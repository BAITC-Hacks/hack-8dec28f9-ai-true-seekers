"""HTTP API: эндпоинты, ошибки, авторизация, CORS и fallback-режимы LLM (через мок OpenAI)."""

from __future__ import annotations

import pytest

from .conftest import OPTIMUM_ITEMS, WORKED_EXAMPLE_ITEMS
from .mock_openai import MockOpenAI

ALLOCATION = {"allocations": {"transport": 25, "green": 15, "social": 25, "safety": 15, "utilities": 20}}


def test_health_and_config(api) -> None:
    code, _, body = api.request("GET", "/health")
    assert code == 200 and body["status"] == "ok"
    code, _, cfg = api.request("GET", "/config")
    assert code == 200 and len(cfg["measures"]) == 14 and len(cfg["districts"]) == 5
    assert cfg["disclaimers"]


def test_score_measures_and_allocation(api) -> None:
    code, _, e = api.request("POST", "/score", {"measures": WORKED_EXAMPLE_ITEMS})
    assert code == 200 and e["score"] == pytest.approx(56.543, abs=0.001)
    assert e["reference"]["score"] == pytest.approx(57.237, abs=0.001)
    code, _, a = api.request("POST", "/score", ALLOCATION)
    assert code == 200 and a["mode"] == "allocation" and a["score"] == pytest.approx(56.015, abs=0.001)
    assert len(a["plan"]) == 5 and a["cost_total"] == 85


def test_invalid_plan_gets_no_score_and_all_reasons(api) -> None:
    code, _, body = api.request("POST", "/score", {"measures": ["M1:Нура", "M3:Нура", "M9:Нура", "M12"]})
    assert code == 422 and body["error"] == "invalid_plan" and body["score"] is None
    assert len(body["reasons"]) == 2  # количество + несовместимость M1/M3
    code, _, v = api.request("POST", "/validate", {"measures": ["M1:Нура", "M3:Нура", "M9:Нура", "M12"]})
    assert code == 200 and v["valid"] is False and v["reasons"] == body["reasons"]
    code, _, ok = api.request("POST", "/validate", {"measures": OPTIMUM_ITEMS})
    assert ok == {"valid": True, "mode": "measures", "reasons": []}


def test_allocation_overspend_is_422(api) -> None:
    body = {"allocations": {"transport": 60, "green": 20, "social": 20, "safety": 20, "utilities": 20}}
    code, _, resp = api.request("POST", "/score", body)
    assert code == 422 and any("Бюджет превышен" in r for r in resp["reasons"])


def test_allocation_without_five_feasible_measures_is_422(api) -> None:
    caps = {"allocations": {"transport": 100, "green": 0, "social": 0, "safety": 0, "utilities": 0}}
    code, _, resp = api.request("POST", "/score", caps)
    assert code == 422 and resp["score"] is None and resp["error"] == "invalid_plan"
    code, _, valid = api.request("POST", "/validate", caps)
    assert code == 200 and valid["valid"] is False


def test_bad_requests(api) -> None:
    assert api.request("POST", "/score", raw=b"{not json")[0] == 400
    assert api.request("GET", "/nope")[0] == 404


def test_body_size_limit(api) -> None:
    import http.client
    from urllib.parse import urlparse

    u = urlparse(api.base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    conn.putrequest("POST", "/score")
    conn.putheader("Content-Length", "2000000")  # заявлено > 1 МБ — сервер отвечает, не читая тело
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 413
    conn.close()


def test_optimal_endpoint(api) -> None:
    code, _, body = api.request("GET", "/optimal?mode=measures&top=3")
    assert code == 200 and len(body["plans"]) == 3 and body["valid_plans"] == 694_395
    code, _, suggested = api.request("GET", "/optimal?mode=allocation")
    assert code == 200 and suggested["score"] == pytest.approx(57.237, abs=0.001)
    assert sum(suggested["allocations"].values()) == 98
    assert api.request("GET", "/optimal?mode=chaos")[0] == 400


def test_token_auth(api_factory) -> None:
    client = api_factory(api_token="s3cret")
    assert client.request("GET", "/health")[0] == 200  # health без токена
    assert client.request("GET", "/config")[0] == 401
    assert client.request("POST", "/score", {"measures": OPTIMUM_ITEMS})[0] == 401
    assert client.request("GET", "/config", headers={"Authorization": "Bearer wrong"})[0] == 401
    assert client.request("GET", "/config", headers={"Authorization": "Bearer s3cret"})[0] == 200


def test_cors_is_off_by_default_and_configurable(api, api_factory) -> None:
    _, headers, _ = api.request("GET", "/health")
    assert "Access-Control-Allow-Origin" not in headers
    client = api_factory(cors_origin="http://localhost:3000")
    _, headers, _ = client.request("GET", "/health")
    assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"


def test_simulate_offline(api) -> None:
    code, _, r = api.request("POST", "/simulate?offline=1", {"measures": WORKED_EXAMPLE_ITEMS})
    assert code == 200
    assert {a["source"] for a in r["agents"].values()} == {"rules"}
    assert r["meta"]["agents_mode"] == "offline" and r["guardrails"]["status"] == "pass"


def test_simulate_without_api_key_falls_back_to_rules(api) -> None:
    code, _, r = api.request("POST", "/simulate", {"measures": WORKED_EXAMPLE_ITEMS})
    assert code == 200 and r["meta"]["agents_mode"] == "offline"


# ── fallback-режимы LLM через мок OpenAI ────────────────────────────────────


@pytest.fixture
def llm_env(monkeypatch):
    def use(mock: MockOpenAI) -> None:
        # Локальный HTTP-мок должен обходить системный SOCKS-прокси среды запуска.
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", mock.base_url)
        monkeypatch.setenv("OPENAI_MODEL", "mock-mini")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "0")
        monkeypatch.setenv("OPENAI_TIMEOUT", "10")
    return use


def _simulate(api, body=None):
    code, _, r = api.request("POST", "/simulate", body or {"measures": WORKED_EXAMPLE_ITEMS})
    assert code == 200
    return r


def test_llm_structured_outputs_ok(api, llm_env) -> None:
    with MockOpenAI("ok") as mock:
        llm_env(mock)
        r = _simulate(api)
    assert mock.schema_errors == []  # все 4 схемы прошли strict-проверку
    agents = r["agents"].values()
    assert {a["source"] for a in agents} == {"llm"}
    assert {a["response_mode"] for a in agents} == {"json_schema"}
    assert r["meta"]["usage"]["total_tokens"] == 600
    ua = r["agents"]["urban_expert"]["output"]["district_assessments"]
    assert [x["district"] for x in ua] == ["Есиль", "Алматы", "Сарыарка", "Байконур", "Нура"]


def test_llm_schema_rejected_falls_back_to_json_mode(api, llm_env) -> None:
    with MockOpenAI("reject_schema") as mock:
        llm_env(mock)
        r = _simulate(api, ALLOCATION)
    agents = r["agents"].values()
    assert {a["source"] for a in agents} == {"llm"}
    assert {a["response_mode"] for a in agents} == {"json_object"}


def test_llm_broken_json_is_repaired(api, llm_env) -> None:
    with MockOpenAI("bad_first") as mock:
        llm_env(mock)
        r = _simulate(api)
    assert {a["source"] for a in r["agents"].values()} == {"llm"}
    assert {a["attempts"] for a in r["agents"].values()} == {2}


def test_llm_outage_falls_back_to_rules(api, llm_env) -> None:
    with MockOpenAI("fail_all") as mock:
        llm_env(mock)
        r = _simulate(api)
    assert {a["source"] for a in r["agents"].values()} == {"rules_fallback"}
    assert all(a["error"] for a in r["agents"].values())
    assert r["final_report"]["recommendations"]  # отчёт полный, несмотря на сбой
    assert r["guardrails"]["status"] == "pass"


def test_llm_hallucination_is_caught_by_guardrails(api, llm_env) -> None:
    with MockOpenAI("hallucinate") as mock:
        llm_env(mock)
        r = _simulate(api)
    g = r["guardrails"]
    assert g["status"] == "warn"
    assert {u["value"] for u in g["unverified"] if u["agent"] == "orchestrator"} == {"88.8", "42.5"}
