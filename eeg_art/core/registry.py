"""
core.registry
---------------
Реестр методов. Ключ — (category, name), где category — обычно
"<ветвь>.<роль>" (например "spectral_lenia.feature_extractor",
"pac_coupling.generator"), а name — конкретный метод внутри этой роли
("fft" vs "welch", и т.п.). Это и есть механизм "добавить в нужный
модуль отдельный модуль метода" — регистрация не требует трогать код
других методов той же роли.

Схема (core.param_spec.ParamSpec) берётся из аргумента register() либо
автоматически с атрибута класса PARAM_SCHEMA, если он есть.
"""

from __future__ import annotations

from typing import Callable


class MethodRegistry:
    _entries: dict[tuple[str, str], dict] = {}

    @classmethod
    def register(cls, category: str, name: str, schema: dict | None = None) -> Callable:
        def deco(target_cls):
            cls._entries[(category, name)] = {
                "cls": target_cls,
                "schema": schema if schema is not None else getattr(target_cls, "PARAM_SCHEMA", {}),
            }
            return target_cls
        return deco

    @classmethod
    def get_cls(cls, category: str, name: str):
        return cls._entries[(category, name)]["cls"]

    @classmethod
    def get_schema(cls, category: str, name: str) -> dict:
        return cls._entries[(category, name)]["schema"]

    @classmethod
    def list_categories(cls) -> list[str]:
        return sorted({c for c, _ in cls._entries})

    @classmethod
    def list_methods(cls, category: str) -> list[str]:
        return sorted(n for c, n in cls._entries if c == category)

    @classmethod
    def create(cls, category: str, name: str, **kwargs):
        target_cls = cls.get_cls(category, name)
        return target_cls(**kwargs)
