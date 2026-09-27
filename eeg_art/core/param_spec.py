"""
core.param_spec
------------------
Формальное описание одного настраиваемого параметра метода — единицы,
диапазон и обоснование (откуда взяты границы), а не просто число без
контекста. Каждый метод (generator/feature extractor), который хочет
быть "научно настраиваемым" и сохраняемым в пресет, объявляет
PARAM_SCHEMA: dict[str, ParamSpec] как атрибут класса.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParamSpec:
    default: float
    min: float
    max: float
    unit: str = ""
    description: str = ""

    def validate(self, value: float, name: str = "") -> float:
        if not (self.min <= value <= self.max):
            raise ValueError(
                f"{name}={value}{self.unit} вне диапазона [{self.min}, {self.max}]{self.unit}: "
                f"{self.description}"
            )
        return value
