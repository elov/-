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


def generate_synthetic_creation_video(
    duration_sec: float, fps: float = 2.0, size: int = 128, seed: int = 0
) -> tuple:
    """
    Синтетическое видео с ЗАВЕДОМО известной последовательностью стадий —
    для валидации modules.creation_stage.stage_classifier (аналогично
    встроенной PAC-связи и известному burst-источнику в других ветках):

        0-20%   эскиз            -- тонкие линии на пустом холсте
        20-45%  подмалёвок       -- крупные малонасыщенные пятна
        45-70%  работа с цветом  -- яркие насыщенные пятна, активное движение
        70-90%  детализация      -- мелкие точечные правки
        90-100% пауза/подпись    -- без изменений

    Возвращает (timestamps, frames, ground_truth), где ground_truth --
    [(момент_начала_сек, метка_стадии), ...].
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n_frames = max(2, int(duration_sec * fps))
    timestamps = np.arange(n_frames) / fps

    canvas = np.ones((size, size, 3)) * 0.92
    yy, xx = np.mgrid[0:size, 0:size]

    bounds = [0.0, 0.20, 0.45, 0.70, 0.90, 1.0]
    ground_truth = [
        (bounds[0] * duration_sec, "эскиз"),
        (bounds[1] * duration_sec, "подмалёвок"),
        (bounds[2] * duration_sec, "работа с цветом"),
        (bounds[3] * duration_sec, "детализация"),
        (bounds[4] * duration_sec, "пауза/подпись"),
    ]

    frames = []
    for t_idx in range(n_frames):
        progress = t_idx / n_frames
        if progress < bounds[1]:
            x0, y0 = rng.uniform(0.2, 0.8, 2) * size
            ang = rng.uniform(0, 2 * np.pi)
            x1, y1 = x0 + np.cos(ang) * size * 0.3, y0 + np.sin(ang) * size * 0.3
            n_pts = 20
            for k in range(n_pts):
                px = x0 + (x1 - x0) * k / n_pts
                py = y0 + (y1 - y0) * k / n_pts
                mask = (yy - py) ** 2 + (xx - px) ** 2 < 1.2 ** 2
                canvas[mask] = [0.2, 0.2, 0.25]
        elif progress < bounds[2]:
            cy, cx = rng.uniform(0.2, 0.8, 2) * size
            r = size * 0.18
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
            tone = 0.4 + rng.uniform(-0.05, 0.05)
            canvas[mask] = [tone, tone * 0.95, tone * 0.85]
        elif progress < bounds[3]:
            cy, cx = rng.uniform(0.2, 0.8, 2) * size
            r = size * 0.12
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
            color = rng.uniform(0.2, 1.0, 3)
            color = color / (color.max() + 1e-9)
            canvas[mask] = color
        elif progress < bounds[4]:
            cy, cx = rng.uniform(0.2, 0.8, 2) * size
            r = size * 0.02
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
            canvas[mask] = rng.uniform(0.1, 0.9, 3)

        frames.append(np.clip(canvas.copy(), 0, 1))

    return timestamps, frames, ground_truth
