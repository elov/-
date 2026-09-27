"""
ingestion.synthetic_gaze
----------------------------
Синтетический поток координат взгляда на холсте для разработки без
реального айтрекера. Формат выхода — (timestamps, xy), где xy уже в
нормализованных координатах холста [-1, 1] x [-1, 1] (т.е. предполагается,
что core.gaze_calibration уже применена — см. модуль для реального
устройства).

Модель: несколько случайных точек фиксации, между ними — быстрые скачки
(саккады), с шумом внутри фиксации.
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_gaze(
    duration_sec: float,
    fs: float = 60.0,
    n_fixations: int = 6,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * fs)
    timestamps = np.arange(n_samples) / fs

    fixation_points = rng.uniform(-0.8, 0.8, size=(n_fixations, 2))
    fixation_durations = rng.integers(low=n_samples // (n_fixations * 2),
                                       high=n_samples // n_fixations,
                                       size=n_fixations)

    xy = np.zeros((n_samples, 2))
    idx = 0
    for i in range(n_fixations):
        point = fixation_points[i]
        dur = min(fixation_durations[i], n_samples - idx)
        if dur <= 0:
            break
        noise = rng.normal(scale=0.01, size=(dur, 2))
        xy[idx:idx + dur] = point + noise
        idx += dur
    if idx < n_samples:
        xy[idx:] = fixation_points[-1]

    return timestamps, xy


def nearest_sample(timestamps: np.ndarray, xy: np.ndarray, t: float) -> tuple[np.ndarray, bool]:
    """Возвращает (xy в момент t, is_fixation). is_fixation оценивается по
    локальной скорости (низкая скорость = фиксация)."""
    i = int(np.searchsorted(timestamps, t))
    i = min(max(i, 1), len(timestamps) - 1)
    velocity = np.linalg.norm(xy[i] - xy[i - 1])
    is_fixation = velocity < 0.02
    return xy[i], is_fixation
