"""
modules.gaze_features.features_gaze
----------------------------------------
FeatureExtractor ветви "айтрекер -> пространственное условие генерации".

Не читает ЭЭГ-окно напрямую (у него другая частота дискретизации, ~60 Гц
против ~250 Гц у ЭЭГ) — вместо этого использует общий таймкод (window.timestamp)
для поиска ближайшего отсчёта взгляда. Так реализуется мультистрим-синхронизация
без общего ingestion-класса — она обсуждалась как роль LSL в реальной системе.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.windowing import EEGWindow
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.features import FeatureSet
from eeg_art.ingestion.synthetic_gaze import nearest_sample
from eeg_art.core.registry import MethodRegistry


@MethodRegistry.register("gaze_features.feature_extractor", "nearest_sample")
class GazeFeatureExtractor:
    def __init__(self, gaze_timestamps: np.ndarray, gaze_xy: np.ndarray):
        self.gaze_timestamps = gaze_timestamps
        self.gaze_xy = gaze_xy

    def extract(self, window: EEGWindow, spatial: SpatialModel) -> FeatureSet:
        xy, is_fixation = nearest_sample(self.gaze_timestamps, self.gaze_xy, window.timestamp)
        return FeatureSet(
            timestamp=window.timestamp,
            fields={"gaze_xy": xy, "is_fixation": bool(is_fixation)},
        )
