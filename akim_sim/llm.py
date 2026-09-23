"""LLM-клиент: Structured Outputs (strict JSON Schema) → JSON mode → repair → исключение (агент уходит в fallback)."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any

from pydantic import BaseModel, ValidationError

try:  # openai нужен только в LLM-режиме
    from openai import AsyncOpenAI, BadRequestError
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]

    class BadRequestError(Exception):  # type: ignore[no-redef]
        pass

DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(RuntimeError):
    pass


def to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON Schema → strict-схема OpenAI Structured Outputs:
    инлайн $ref, additionalProperties=false, все поля required, без title/default."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            resolved = walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            if "description" in node:
                resolved["description"] = node["description"]
            return resolved
        out: dict[str, Any] = {}
        for key, val in node.items():
            if key in ("title", "default", "examples"):
                continue
            if key == "properties":
                out[key] = {pk: walk(pv) for pk, pv in val.items()}
            else:
                out[key] = walk(val)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out

    return walk(schema)


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


class LLMClient:
    def __init__(self, *, api_key: str, model: str, orchestrator_model: str | None = None,
                 base_url: str | None = None, timeout: float = 60.0, temperature: float | None = None,
                 max_retries: int = 2) -> None:
        if AsyncOpenAI is None:
            raise LLMError("Пакет openai не установлен: pip install -r requirements.txt")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.orchestrator_model = orchestrator_model or model
        self.temperature = temperature
        self.response_mode = "json_schema"  # переключится на json_object, если провайдер не поддерживает схемы

    @classmethod
    def from_env(cls, model_override: str | None = None) -> "LLMClient | None":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key or api_key.startswith("sk-your"):
            return None
        temp_raw = os.getenv("OPENAI_TEMPERATURE", "").strip()
        model = model_override or os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
        return cls(
            api_key=api_key, model=model,
            orchestrator_model=(model_override or os.getenv("OPENAI_MODEL_ORCHESTRATOR", "").strip() or model),
            base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
            timeout=float(os.getenv("OPENAI_TIMEOUT", "60") or 60),
            temperature=float(temp_raw) if temp_raw else None,
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2") or 2),
        )

    async def aclose(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass

    async def structured(self, *, system: str, payload: dict[str, Any], schema: type[BaseModel],
                         schema_name: str, model: str | None = None) -> tuple[BaseModel, dict[str, Any]]:
        model = model or self.model
        strict = to_strict_schema(schema)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Входные данные (JSON):\n" + json.dumps(payload, ensure_ascii=False)},
        ]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_error: Exception | None = None
        for attempt in range(1, 4):
            mode = self.response_mode  # фиксируем режим запроса (агенты работают параллельно)
            kwargs: dict[str, Any] = {"model": model}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if mode == "json_schema":
                kwargs["messages"] = messages
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": strict},
                }
            else:
                sys_with_schema = (system + "\n\nВерни ОДИН JSON-объект строго по этой JSON Schema:\n"
                                   + json.dumps(strict, ensure_ascii=False))
                kwargs["messages"] = [{"role": "system", "content": sys_with_schema}] + messages[1:]
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = await self.client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                last_error = exc
                if mode == "json_schema":
                    self.response_mode = "json_object"  # провайдер не принял схему — деградируем в JSON mode
                    continue
                raise LLMError(f"BadRequest: {exc}") from exc
            if getattr(resp, "usage", None) is not None:
                for k in usage:
                    usage[k] += int(getattr(resp.usage, k, 0) or 0)
            msg = resp.choices[0].message
            refusal = getattr(msg, "refusal", None)
            if refusal:
                raise LLMError(f"Модель отказалась отвечать: {refusal}")
            content = msg.content or ""
            try:
                parsed = schema.model_validate(json.loads(extract_json(content)))
                return parsed, {"model": model, "attempts": attempt, "mode": mode, "usage": usage}
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                messages = messages + [
                    {"role": "assistant", "content": content[:8000]},
                    {"role": "user", "content": f"Ответ не прошёл валидацию схемы: {str(exc)[:1200]}\n"
                                                "Верни исправленный JSON целиком, строго по схеме."},
                ]
        raise LLMError(f"Не удалось получить валидный JSON: {last_error}")
