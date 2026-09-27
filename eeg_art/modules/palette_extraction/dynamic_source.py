"""
modules.palette_extraction.dynamic_source
----------------------------------------------
Ветвь "создание": палитра не фиксирована одним снимком (как в
static_source — наблюдение готовой картины), а пересчитывается по
ближайшему кадру камеры-на-картину для каждого окна ЭЭГ — палитра
эволюционирует синхронно с таймлайном сессии, как и обсуждалось в
архитектуре.

Пересчёт k-means на каждом окне может быть дорогим при плотной сетке
окон — recompute_every_n позволяет считать палитру раз в N окон,
переиспользуя последний результат между пересчётами (частая практика
для дорогих признаков поверх более редкого исходного потока).
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.multistream_sync import nearest_by_timestamp
from eeg_art.modules.palette_extraction.extract import extract_palette, Palette


class PaletteDynamicSource:
    def __init__(
        self,
        video_timestamps: np.ndarray,
        video_frames: list[np.ndarray],
        n_main: int = 4,
        n_accent: int = 2,
        recompute_every_n: int = 1,
    ):
        self.video_timestamps = video_timestamps
        self.video_frames = video_frames
        self.n_main = n_main
        self.n_accent = n_accent
        self.recompute_every_n = recompute_every_n

        self._call_count = 0
        self._last_palette: Palette | None = None
        self._last_frame_idx: int | None = None

    def get_palette(self, timestamp: float) -> tuple[Palette, np.ndarray, float]:
        """Возвращает (палитра, использованный кадр, рассинхрон в секундах)."""
        frame, frame_idx, dt = nearest_by_timestamp(self.video_timestamps, self.video_frames, timestamp)

        need_recompute = (
            self._last_palette is None
            or frame_idx != self._last_frame_idx
            or self._call_count % self.recompute_every_n == 0
        )
        if need_recompute:
            self._last_palette = extract_palette(frame, n_main=self.n_main, n_accent=self.n_accent)
            self._last_frame_idx = frame_idx

        self._call_count += 1
        return self._last_palette, frame, dt
