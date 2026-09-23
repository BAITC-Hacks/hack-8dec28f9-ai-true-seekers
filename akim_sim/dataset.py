"""Официальный датасет кейса «Аким на 5 часов» (документ «Датасет районов», Astana Innovations).

Все числа этого модуля перенесены из документа организаторов без изменений. Любые другие коэффициенты
проекта (прокси рисков, сценарии чувствительности) живут в отдельных модулях и явно помечены как модельные.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

DATASET_NAME = "Датасет районов — кейс Astana Innovations «Аким на 5 часов»"

# ── Правила игры ────────────────────────────────────────────────────────────
BUDGET = 100.0  # условные единицы (в легенде игры ~ млрд ₸)
UNIT = "ед."
N_DECISIONS = 5  # ровно 5 решений
MAX_PER_DIRECTION = 2  # не более 2 мер из одного направления
HORIZON_QUARTERS = 8  # горизонт симуляции — 8 кварталов
CRIT_THRESHOLD = 40.0  # пара «район × показатель» ниже 40 — критическая
CRIT_PENALTY = 1.0  # −1 балл за каждую критическую пару
W_AVG = 0.7  # Score = 0.7·D_avg + 0.3·min_d D_d − 1.0·N_crit
W_MIN = 0.3
CITY = "город"  # «район» для мер масштаба города

# ── Направления (сферы) и показатели ────────────────────────────────────────
SECTORS: tuple[str, ...] = ("transport", "green", "social", "safety", "utilities")
SECTOR_NAMES: dict[str, str] = {
    "transport": "Транспорт",
    "green": "Экология и озеленение",
    "social": "Социальная сфера",
    "safety": "Безопасность",
    "utilities": "Городские сервисы",
}

INDICATORS: tuple[str, ...] = ("T1", "T2", "E1", "E2", "S1", "S2", "B1", "B2", "C1", "C2")
INDICATOR_SECTOR: dict[str, str] = {
    "T1": "transport", "T2": "transport", "E1": "green", "E2": "green", "S1": "social",
    "S2": "social", "B1": "safety", "B2": "safety", "C1": "utilities", "C2": "utilities",
}
INDICATOR_NAMES: dict[str, str] = {
    "T1": "Разгруженность дорог в час пик",
    "T2": "Доступность общественного транспорта",
    "E1": "Озеленение (м² зелени на жителя)",
    "E2": "Качество воздуха зимой",
    "S1": "Места в школах и детсадах",
    "S2": "Первичная медицинская помощь",
    "B1": "Безопасность улиц",
    "B2": "Безопасность дорожного движения",
    "C1": "Надёжность тепло- и водоснабжения",
    "C2": "Скорость закрытия обращений",
}
WEIGHTS: dict[str, float] = {
    "T1": 0.10, "T2": 0.10, "E1": 0.09, "E2": 0.11, "S1": 0.11,
    "S2": 0.11, "B1": 0.09, "B2": 0.09, "C1": 0.10, "C2": 0.10,
}

# ── Районы ──────────────────────────────────────────────────────────────────
DISTRICTS: tuple[str, ...] = ("Есиль", "Алматы", "Сарыарка", "Байконур", "Нура")
POPULATION: dict[str, float] = {"Есиль": 0.27, "Алматы": 0.24, "Сарыарка": 0.20, "Байконур": 0.13, "Нура": 0.16}
DISTRICT_NOTES: dict[str, str] = {
    "Есиль": "обеспеченный район; заторы на мостах, переполненные школы",
    "Алматы": "изношенные сети ЖКХ и пробки",
    "Сарыарка": "смог от частного сектора, мало зелени",
    "Байконур": "сбалансированный район без резких провалов",
    "Нура": "худшие показатели соцсферы и транспорта — главный аутсайдер",
}
_BASE_ROWS = {
    #            T1  T2  E1  E2  S1  S2  B1  B2  C1  C2
    "Есиль":    (45, 62, 68, 72, 48, 55, 78, 60, 75, 70),
    "Алматы":   (40, 75, 50, 55, 60, 65, 62, 52, 50, 60),
    "Сарыарка": (50, 70, 42, 40, 62, 68, 58, 55, 45, 55),
    "Байконур": (52, 68, 55, 50, 58, 60, 52, 58, 55, 58),
    "Нура":     (55, 40, 45, 65, 38, 35, 55, 50, 60, 50),
}
BASELINE: dict[str, dict[str, float]] = {
    d: {k: float(v) for k, v in zip(INDICATORS, row)} for d, row in _BASE_ROWS.items()
}


# ── Каталог мероприятий ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class Measure:
    code: str
    name: str
    sector: str
    cost: float
    lag: int  # кварталов до запуска
    citywide: bool  # True — эффект во всех районах; False — в одном выбранном районе
    effects: Mapping[str, float] = field(default_factory=dict)

    @property
    def realized_share(self) -> float:
        """Реализованная доля эффекта за горизонт: (8 − L) / 8."""
        return (HORIZON_QUARTERS - self.lag) / HORIZON_QUARTERS

    @property
    def scope(self) -> str:
        return "город" if self.citywide else "район"


_MEASURE_LIST = (
    Measure("M1", "Выделенные полосы для автобусов (BRT)", "transport", 18, 2, False, {"T1": 6, "T2": 9}),
    Measure("M2", "Умные светофоры", "transport", 22, 2, True, {"T1": 4, "B2": 3}),
    Measure("M3", "Линия ЛРТ / расширение", "transport", 30, 4, False, {"T1": 16, "T2": 20, "E2": 4}),
    Measure("M4", "Парк / сквер", "green", 15, 2, False, {"E1": 12, "E2": 3, "B1": 2}),
    Measure("M5", "Чистое топливо для частного сектора", "green", 25, 3, False, {"E2": 14, "C1": 4}),
    Measure("M6", "Озеленение и ветрозащитные полосы", "green", 20, 4, True, {"E1": 5, "E2": 3}),
    Measure("M7", "Школа + детский сад", "social", 24, 3, False, {"S1": 16}),
    Measure("M8", "Центр семейного здоровья", "social", 20, 3, False, {"S2": 14}),
    Measure("M9", "Дворовые спорт-хабы", "social", 10, 1, False, {"S1": 3, "S2": 3, "B1": 3}),
    Measure("M10", "Освещение и камеры Safe City", "safety", 12, 1, False, {"B1": 12, "B2": 2}),
    Measure("M11", "Безопасные переходы в школьных зонах", "safety", 10, 1, False, {"B2": 12, "T1": -2}),
    Measure("M12", "Единая цифровая платформа обращений", "utilities", 14, 1, True, {"C2": 5}),
    Measure("M13", "Модернизация тепло- и водосетей", "utilities", 28, 4, False, {"C1": 18, "E2": 2}),
    Measure("M14", "Аварийные бригады ЖКХ и оповещение", "utilities", 16, 1, True, {"C1": 5, "C2": 2}),
)
MEASURES: dict[str, Measure] = {m.code: m for m in _MEASURE_LIST}
MEASURE_CODES: tuple[str, ...] = tuple(MEASURES)


@dataclass(frozen=True)
class Synergy:
    a: str
    b: str
    anchor: str  # бонус начисляется в районе этой меры
    indicator: str
    bonus: float  # фиксированный, без масштабирования лагом


SYNERGIES: tuple[Synergy, ...] = (
    Synergy("M1", "M2", "M1", "T1", 2.0),
    Synergy("M10", "M12", "M10", "B1", 2.0),
    Synergy("M5", "M6", "M5", "E2", 2.0),
)

# Несовместимости: глобальная (в любом районе) и в пределах одного района
INCOMPATIBLE_GLOBAL: tuple[tuple[str, str, str], ...] = (("M1", "M3", "либо BRT, либо ЛРТ — в любом районе"),)
INCOMPATIBLE_SAME_DISTRICT: tuple[tuple[str, str, str], ...] = (
    ("M4", "M7", "нельзя в одном районе — конфликт за участок"),
    ("M5", "M13", "нельзя в одном районе — дублирование программы"),
)

# Эталонный пример из документа: стоимость 95, Score ≈ 56.5 (+4.0 к базе 52.56)
WORKED_EXAMPLE: tuple[tuple[str, str], ...] = (
    ("M7", "Нура"), ("M8", "Нура"), ("M10", "Нура"), ("M12", CITY), ("M5", "Сарыарка"),
)
REFERENCE_BASE_SCORE = 52.56
REFERENCE_WORKED_EXAMPLE_SCORE = 56.5

# ── Нормализация пользовательского ввода ────────────────────────────────────
SECTOR_ALIASES: dict[str, str] = {
    "transport": "transport", "транспорт": "transport", "mobility": "transport",
    "green": "green", "greening": "green", "ecology": "green", "озеленение": "green", "экология": "green",
    "social": "social", "соцсфера": "social", "социальная_сфера": "social",
    "safety": "safety", "security": "safety", "безопасность": "safety",
    "utilities": "utilities", "services": "utilities", "city_services": "utilities", "housing": "utilities",
    "городские_сервисы": "utilities", "сервисы": "utilities", "жкх": "utilities",
}
DISTRICT_ALIASES: dict[str, str] = {
    "есиль": "Есиль", "esil": "Есиль", "yesil": "Есиль", "esil'": "Есиль",
    "алматы": "Алматы", "almaty": "Алматы",
    "сарыарка": "Сарыарка", "saryarka": "Сарыарка", "sary-arka": "Сарыарка",
    "байконур": "Байконур", "baikonur": "Байконур", "baykonur": "Байконур", "baiqonyr": "Байконур",
    "нура": "Нура", "nura": "Нура",
    "город": CITY, "city": CITY, "all": CITY, "весь_город": CITY, "все": CITY, "*": CITY,
}


def normalize_sector(key: str) -> str:
    k = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return SECTOR_ALIASES.get(k, k)


def normalize_district(name: str | None) -> str | None:
    if name is None:
        return None
    k = str(name).strip().lower().replace(" ", "_")
    if not k:
        return None
    return DISTRICT_ALIASES.get(k, str(name).strip())


def normalize_code(code: str) -> str:
    c = str(code).strip().upper().replace("М", "M")  # кириллическая «М» → латинская
    return c
