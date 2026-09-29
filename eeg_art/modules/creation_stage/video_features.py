"""
modules.creation_stage.video_features
------------------------------------------
Признаки для определения стадии создания произведения по видео камеры,
направленной на холст (см. протокол исследования: эскиз -> подмалёвок ->
работа с цветом -> детализация -> пауза/подпись).

Признаки считаются между соседним и текущим кадром, плюс кумулятивно
относительно самого первого кадра сессии (чистый холст/набросок в начале):

    motion        — |текущий кадр - предыдущий| — сколько меняется прямо сейчас
    fill_fraction — доля пикселей, заметно отличающихся от ПЕРВОГО кадра —
                    сколько холста уже тронуто с начала сессии (растёт монотонно)
    edge_density  — средняя величина пространственного градиента яркости —
                    высокая при тонких линиях (эскиз), низкая при заливках
    saturation    — средний (max-min) по каналам — низкая у подмалёвка
                    (тональная проработка), выше на этапе работы с цветом
"""

from __future__ import annotations

import numpy as np


class VideoFeatureTracker:
    def __init__(self, change_threshold: float = 0.03):
        self.first_frame: np.ndarray | None = None
        self.prev_frame: np.ndarray | None = None
        self.change_threshold = change_threshold

    def extract(self, frame: np.ndarray) -> dict:
        if self.first_frame is None:
            self.first_frame = frame.copy()
        if self.prev_frame is None:
            self.prev_frame = frame.copy()

        motion = float(np.abs(frame - self.prev_frame).mean())

        diff_from_start = np.abs(frame - self.first_frame).mean(axis=-1)
        fill_fraction = float((diff_from_start > self.change_threshold).mean())

        luminance = frame.mean(axis=-1)
        gy, gx = np.gradient(luminance)
        edge_density = float(np.sqrt(gy ** 2 + gx ** 2).mean())

        saturation = float((frame.max(axis=-1) - frame.min(axis=-1)).mean())

        self.prev_frame = frame.copy()
        return {
            "motion": motion,
            "fill_fraction": fill_fraction,
            "edge_density": edge_density,
            "saturation": saturation,
        }
