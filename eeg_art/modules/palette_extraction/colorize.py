"""
modules.palette_extraction.colorize
----------------------------------------
Реализация core/render_mapping, отложенная с этапа проектирования:
разделяет форму (состояние автомата, скаляр [0,1]) и цвет (палитра).
Строит непрерывную цветовую шкалу интерполяцией между основными
цветами палитры (уже отсортированными по кругу оттенков в extract_palette),
затем красит скалярное поле в RGB по этой шкале.
"""

from __future__ import annotations

import numpy as np

from eeg_art.modules.palette_extraction.extract import Palette


def build_colormap(palette: Palette, use_accent_as_highlight: bool = True) -> np.ndarray:
    """Возвращает LUT (256, 3) — таблицу цветов для np.interp-подобного маппинга."""
    colors = [c.rgb for c in palette.main] or [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    if len(colors) == 1:
        colors = colors * 2
    colors = np.array(colors)

    xp = np.linspace(0, 1, len(colors))
    x = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(x, xp, colors[:, ch]) for ch in range(3)], axis=-1)

    if use_accent_as_highlight and palette.accent:
        # верхний хвост шкалы (самые "активные" клетки автомата) окрашивается
        # акцентным цветом с наибольшей насыщенностью — акцент подчёркивает
        # пики активности, а не фон
        top_accent = max(palette.accent, key=lambda c: c.chroma)
        blend = np.linspace(0, 1, 40)[:, None]
        lut[-40:] = (1 - blend) * lut[-40:] + blend * np.array(top_accent.rgb)

    return lut


def colorize(state_values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = np.clip((state_values * 255).astype(int), 0, 255)
    return lut[idx]
