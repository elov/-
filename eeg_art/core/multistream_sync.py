"""
core.multistream_sync
--------------------------
Единый механизм поиска ближайшего по времени отсчёта в потоке с другой
частотой дискретизации (видео ~2-5 fps, айтрекер ~60 Гц, ЭЭГ ~250 Гц) —
то, что в реальной системе делает LSL по общим меткам времени. Здесь —
программная имитация той же роли, чтобы код был готов к замене на
настоящий LSL без изменения интерфейса вызывающих модулей.
"""

from __future__ import annotations

import numpy as np


def nearest_index(timestamps: np.ndarray, t: float) -> int:
    i = int(np.searchsorted(timestamps, t))
    return min(max(i, 0), len(timestamps) - 1)


def nearest_by_timestamp(timestamps: np.ndarray, items: list, t: float):
    """items — произвольный список (кадры, координаты и т.п.) той же длины,
    что timestamps. Возвращает (item, index, dt) — dt для контроля рассинхрона."""
    idx = nearest_index(timestamps, t)
    return items[idx], idx, abs(timestamps[idx] - t)
