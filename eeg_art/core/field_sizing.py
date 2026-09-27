"""
core.field_sizing
---------------------
Задать размер поля клеточного автомата соразмерно пропорциям
просматриваемого/создаваемого произведения — а не всегда квадрат.

Подход: фиксируется БЮДЖЕТ ячеек (target_cells) — сколько всего клеток
может себе позволить автомат по вычислительной стоимости (это то, что
раньше было grid_size**2 при квадрате 64x64 = 4096). Форма (H, W)
подбирается так, чтобы сохранить пропорции произведения и уложиться в
этот бюджет. Это принципиально отличается от наивного "взять сторону
оригинала как одну из сторон поля": для очень вытянутой (панорамной)
картины наивный подход мог бы случайно раздуть вычислительную стоимость
в разы — бюджет ячеек этого не допускает независимо от пропорций.

Совместимо с существующими генераторами без изменения их интерфейса:
LeniaGenerator и ParticleBloomGenerator принимают необязательный
field_shape=(H, W), который, если задан, перекрывает квадратный
grid_size (полностью обратная совместимость — по умолчанию field_shape
не задан, поведение как раньше).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def aspect_ratio_from_dimensions(width: float, height: float) -> float:
    return float(width) / float(height)


def aspect_ratio_from_image(path: str) -> float:
    with Image.open(path) as im:
        w, h = im.size
    return aspect_ratio_from_dimensions(w, h)


def field_shape_for_aspect(
    aspect_ratio: float,
    target_cells: int = 64 * 64,
    min_dim: int = 16,
    max_dim: int = 512,
) -> tuple[int, int]:
    """
    aspect_ratio = width / height произведения.
    Возвращает (H, W) для поля с тем же соотношением сторон и суммарным
    числом клеток ~= target_cells (с округлением и ограничением
    min_dim..max_dim на каждую сторону).
    """
    h = np.sqrt(target_cells / aspect_ratio)
    w = h * aspect_ratio

    h = int(np.clip(round(h), min_dim, max_dim))
    w = int(np.clip(round(w), min_dim, max_dim))
    return h, w


def field_shape_for_image(
    path: str,
    target_cells: int = 64 * 64,
    min_dim: int = 16,
    max_dim: int = 512,
) -> tuple[int, int]:
    """Удобная обёртка: пропорции берутся прямо из файла произведения."""
    ratio = aspect_ratio_from_image(path)
    return field_shape_for_aspect(ratio, target_cells, min_dim, max_dim)


def field_shape_for_dimensions(
    width: float,
    height: float,
    target_cells: int = 64 * 64,
    min_dim: int = 16,
    max_dim: int = 512,
) -> tuple[int, int]:
    """То же самое, но по явно заданным пропорциям (например, физический
    размер холста в см: field_shape_for_dimensions(100, 70, ...))."""
    ratio = aspect_ratio_from_dimensions(width, height)
    return field_shape_for_aspect(ratio, target_cells, min_dim, max_dim)
