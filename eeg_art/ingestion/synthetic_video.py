"""
ingestion.synthetic_video
------------------------------
Синтетические видеопотоки для разработки без реальных камер:

    video_artwork — камера на картину/холст: здесь имитируется процесс
        создания (постепенно добавляются цветовые пятна — как будто
        художник рисует), чтобы было на чём проверить dynamic_source
        палитры (палитра должна меняться по мере "рисования").
    video_space   — камера пространства/зала: чисто документационный
        поток (см. архитектурное решение — не участвует в генерации),
        здесь просто имитация сцены с "зрителем" (движущаяся точка).

Оба возвращают (timestamps, frames) с частотой кадров fps, независимой
от частоты ЭЭГ — синхронизация по общему таймкоду происходит через
core.multistream_sync.nearest_by_timestamp, как это делал бы LSL в
реальной системе.
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_video_artwork(
    duration_sec: float, fps: float = 2.0, size: int = 128, seed: int = 0
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Имитация процесса создания: фон появляется сразу, затем со временем
    добавляются пятна одно за другим (как мазки), от 0 до n_strokes."""
    rng = np.random.default_rng(seed)
    n_frames = max(2, int(duration_sec * fps))
    timestamps = np.arange(n_frames) / fps

    yy, xx = np.mgrid[0:size, 0:size] / size
    background = np.stack([
        0.15 + 0.10 * yy,
        0.18 + 0.12 * yy,
        0.30 + 0.15 * (1 - yy),
    ], axis=-1)

    strokes = []
    palette_targets = [
        [0.25, 0.45, 0.30], [0.55, 0.40, 0.20], [0.95, 0.10, 0.10],
        [0.98, 0.85, 0.05], [0.95, 0.15, 0.75],
    ]
    for i, color in enumerate(palette_targets):
        cy = rng.uniform(0.2, 0.8) * size
        cx = rng.uniform(0.2, 0.8) * size
        r = size * (0.2 if i < 2 else 0.045)
        strokes.append((cy, cx, r, color))

    frames = []
    for t_idx in range(n_frames):
        progress = t_idx / (n_frames - 1)  # 0..1 по всей сессии
        n_visible = int(progress * len(strokes)) + 1
        img = background.copy()
        for (cy, cx, r, color) in strokes[:n_visible]:
            mask = (yy * size - cy) ** 2 + (xx * size - cx) ** 2 < r ** 2
            img[mask] = color
        frames.append(np.clip(img, 0, 1))

    return timestamps, frames


def generate_synthetic_video_space(
    duration_sec: float, fps: float = 5.0, size: int = 96, seed: int = 1
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Чисто документационный поток: тёмная сцена + движущаяся точка
    ('зритель'), только для архива/валидации, не для генерации."""
    rng = np.random.default_rng(seed)
    n_frames = max(2, int(duration_sec * fps))
    timestamps = np.arange(n_frames) / fps

    yy, xx = np.mgrid[0:size, 0:size] / size
    frames = []
    for t_idx in range(n_frames):
        img = np.full((size, size, 3), 0.05)
        t = t_idx / n_frames
        cy = size * (0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
        cx = size * (0.5 + 0.3 * np.cos(2 * np.pi * 0.3 * t))
        mask = (yy * size - cy) ** 2 + (xx * size - cx) ** 2 < (size * 0.05) ** 2
        img[mask] = [0.8, 0.8, 0.8]
        frames.append(img)

    return timestamps, frames
