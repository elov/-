"""
core.windowing
---------------
Скользящее окно по многоканальному сигналу + единый таймкод.

В реальной системе роль единого таймкода на несколько потоков (EEG, видео,
айтрекер, аудио) должен играть LSL (Lab Streaming Layer) — стандарт
синхронизации в нейронауке. Здесь для первого среза единый таймкод —
это просто время в секундах от начала сессии; интерфейс (EEGWindow с полем
timestamp) рассчитан на то, чтобы позже подставить туда реальные LSL-метки
без изменения остального кода.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass
class EEGWindow:
    data: np.ndarray          # shape (n_channels, n_samples)
    channels: list[str]
    fs: float                 # частота дискретизации, Гц
    timestamp: float          # время центра/конца окна, сек от начала сессии


def sliding_windows(
    stream: np.ndarray,
    channels: list[str],
    fs: float,
    win_sec: float = 1.0,
    step_sec: float = 0.15,
) -> Iterator[EEGWindow]:
    """
    stream: (n_channels, n_samples) — весь сигнал (или растущий буфер live-потока).
    Возвращает EEGWindow для каждого шага скользящего окна.
    """
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    n_samples = stream.shape[1]

    start = 0
    while start + win <= n_samples:
        chunk = stream[:, start:start + win]
        timestamp = (start + win) / fs
        yield EEGWindow(data=chunk, channels=channels, fs=fs, timestamp=timestamp)
        start += step
