"""
modules.source_volumetric.features_source
----------------------------------------------
FeatureExtractor ветви "локализация источников -> 3D-автомат".

В отличие от spectral (мощность по диапазонам) здесь нужен сам вектор
потенциалов на скальпе — вход обратной задачи. Берётся RMS-амплитуда
каждого канала за окно (устойчивее одиночного отсчёта к шуму), в
широкой полосе — источник должен объяснять весь наблюдаемый сигнал,
а не один ритм; при желании можно передать конкретный band для
ветви "локализация активности именно этого ритма".
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.windowing import EEGWindow
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.features import FeatureSet


class ScalpActivityFeatureExtractor:
    def extract(self, window: EEGWindow, spatial: SpatialModel) -> FeatureSet:
        rms = np.sqrt(np.mean(window.data ** 2, axis=1))
        fields = {"scalp_activity": {ch: float(v) for ch, v in zip(window.channels, rms)}}
        return FeatureSet(timestamp=window.timestamp, fields=fields)
