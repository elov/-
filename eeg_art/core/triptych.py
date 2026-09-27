"""
core.triptych
---------------
Склеивает три панели (сырые данные / признаки-"партитура" / генерация) в
один кадр — ровно та структура зала, которую обсуждали: три
синхронизированных по таймкоду экрана.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0, 1)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return (arr * 255).astype(np.uint8)


def compose_triptych(raw_img: np.ndarray, feature_img: np.ndarray, gen_img: np.ndarray, height: int = 512) -> np.ndarray:
    panels = [Image.fromarray(_to_uint8(a)) for a in (raw_img, feature_img, gen_img)]
    resized = []
    for p in panels:
        w = int(p.width * height / p.height)
        resized.append(p.resize((w, height)))

    total_w = sum(p.width for p in resized) + 2 * 4
    sheet = Image.new("RGB", (total_w, height), color=(10, 10, 10))
    x = 0
    for p in resized:
        sheet.paste(p, (x, 0))
        x += p.width + 4
    return np.asarray(sheet).astype(float) / 255.0
