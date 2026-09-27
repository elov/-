"""
core.config
-------------
Пресеты: сериализация (category, method, params) в JSON и обратно.
Валидация по PARAM_SCHEMA зарегистрированного метода — пресет с
параметром вне научно обоснованного диапазона не сохранится молча.
"""

from __future__ import annotations

import json

from eeg_art.core.registry import MethodRegistry


def validate_params(category: str, name: str, params: dict) -> dict:
    schema = MethodRegistry.get_schema(category, name)
    for key, value in params.items():
        if key in schema and isinstance(value, (int, float)):
            schema[key].validate(value, name=key)
    return params


def save_preset(path: str, category: str, name: str, params: dict, meta: dict | None = None) -> None:
    validate_params(category, name, params)
    payload = {"category": category, "method": name, "params": params, "meta": meta or {}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_preset(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def instantiate_from_preset(path: str, **extra_kwargs):
    preset = load_preset(path)
    validate_params(preset["category"], preset["method"], preset["params"])
    kwargs = {**preset["params"], **extra_kwargs}
    return MethodRegistry.create(preset["category"], preset["method"], **kwargs)
