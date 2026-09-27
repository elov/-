"""
core.features
--------------
Общий интерфейс для всех модулей-извлекателей признаков (spectral,
connectivity, microstates, gaze, ...). Любой модуль реализует
FeatureExtractor.extract(window, spatial) -> FeatureSet; ядро и генераторы
работают с FeatureSet, не зная, каким методом он получен.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
import numpy as np

from eeg_art.core.windowing import EEGWindow
from eeg_art.core.spatial_model import SpatialModel


@dataclass
class FeatureSet:
    """
    Контейнер признаков одного окна.

    fields: произвольные скалярные/векторные метрики (например band power
            по каналам для каждого диапазона) — ключ -> {канал: значение}
            либо ключ -> число (глобальная метрика).
    timestamp: таймкод окна, для синхронизации с остальными потоками.
    """
    timestamp: float
    fields: dict = field(default_factory=dict)


class FeatureExtractor(Protocol):
    def extract(self, window: EEGWindow, spatial: SpatialModel) -> FeatureSet:
        ...
