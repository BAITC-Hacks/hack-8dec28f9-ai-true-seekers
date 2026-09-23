"""Входные данные: два режима игры.

- measures   — официальный режим датасета: ровно 5 мер из каталога M1–M14 (с районами для районных мер);
- allocation — режим исходного брифа: суммы по 5 сферам; оценивается на той же официальной шкале.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from . import dataset as ds
from .model import Choice, validate_plan


class PlanValidationError(ValueError):
    """Невалидный набор мер: Score не считается, возвращаются все причины (правило датасета)."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class AllocationDecision(BaseModel):
    mode: Literal["allocation"] = "allocation"
    budget_total: float = Field(default=ds.BUDGET, gt=0, le=10_000)
    allocations: dict[str, float]
    akim_name: Optional[str] = Field(default=None, max_length=120)
    akim_statement: Optional[str] = Field(default=None, max_length=1500)

    @model_validator(mode="before")
    @classmethod
    def _flat_format(cls, data: Any) -> Any:
        if isinstance(data, dict) and "allocations" not in data:
            data = dict(data)
            data["allocations"] = {k: data.pop(k) for k in list(data) if ds.normalize_sector(k) in ds.SECTORS}
        return data

    @field_validator("allocations", mode="before")
    @classmethod
    def _check_allocations(cls, v: Any) -> dict[str, float]:
        if not isinstance(v, dict):
            raise ValueError("allocations должен быть объектом {сфера: сумма}")
        out: dict[str, float] = {}
        for raw_key, raw_val in v.items():
            key = ds.normalize_sector(raw_key)
            if key not in ds.SECTORS:
                raise ValueError(f"Неизвестная сфера «{raw_key}». Допустимо: {', '.join(ds.SECTORS)}")
            if key in out:
                raise ValueError(f"Сфера «{key}» указана дважды")
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                raise ValueError(f"Сумма для «{raw_key}» должна быть числом, получено: {raw_val!r}") from None
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"Сумма для «{raw_key}» должна быть неотрицательным числом")
            out[key] = round(val, 4)
        missing = [k for k in ds.SECTORS if k not in out]
        if missing:
            raise ValueError("Нужно 5 решений — не хватает сфер: " + ", ".join(missing))
        return {k: out[k] for k in ds.SECTORS}

    @model_validator(mode="after")
    def _check_budget(self) -> "AllocationDecision":
        total = sum(self.allocations.values())
        if total > self.budget_total + 1e-6:
            raise ValueError(f"Бюджет превышен: распределено {total:g} из {self.budget_total:g} {ds.UNIT} "
                             f"(перерасход {total - self.budget_total:g})")
        return self


class MeasuresDecision(BaseModel):
    mode: Literal["measures"] = "measures"
    budget_total: float = Field(default=ds.BUDGET, gt=0, le=10_000)
    measures: list[Any]
    akim_name: Optional[str] = Field(default=None, max_length=120)
    akim_statement: Optional[str] = Field(default=None, max_length=1500)

    @field_validator("measures", mode="before")
    @classmethod
    def _accept_mapping(cls, v: Any) -> Any:
        if isinstance(v, dict):  # {"M7": "Нура", "M12": "город"}
            return [{"code": k, "district": d} for k, d in v.items()]
        return v

    def choices(self) -> list[Choice]:
        choices, reasons = validate_plan(self.measures, self.budget_total)
        if reasons:
            raise PlanValidationError(reasons)
        return choices


Decision = Union[AllocationDecision, MeasuresDecision]


def parse_decision(data: Any) -> Decision:
    """Определяет режим по содержимому: список мер или ключ measures → режим мер, иначе — режим сумм."""
    if isinstance(data, list):
        return MeasuresDecision(measures=data)
    if not isinstance(data, dict):
        raise ValueError("Ожидается JSON-объект с ключом measures (режим мер) или allocations (режим сумм)")
    # Принимаем прежний формат main.py из переданной версии GPT.
    if data.get("mode") == "allocations" or "allocation" in data:
        data = dict(data)
        if "allocations" not in data and "allocation" in data:
            data["allocations"] = data.pop("allocation")
        if data.get("mode") == "allocations":
            data["mode"] = "allocation"
    if "decisions" in data and data.get("mode") == "measures" and "measures" not in data:
        data = {**data, "measures": data["decisions"]}
    mode = data.get("mode")
    if mode == "measures" or (mode is None and "measures" in data):
        return MeasuresDecision.model_validate(data)
    if mode not in (None, "allocation"):
        raise ValueError(f"Неизвестный режим «{mode}». Допустимо: measures, allocation")
    return AllocationDecision.model_validate(data)
