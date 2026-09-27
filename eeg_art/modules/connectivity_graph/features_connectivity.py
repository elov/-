"""
modules.connectivity_graph.features_connectivity
----------------------------------------------------
Feature-модуль ветви "связность -> граф-автомат".

Метод: Phase Locking Value (PLV) между каждой парой каналов — устоявшаяся,
задокументированная в литературе метрика функциональной связности:

    1. Гильберт-преобразование сигнала каждого канала -> мгновенная фаза.
    2. PLV(i, j) = | mean_t( exp( i * (phase_i(t) - phase_j(t)) ) ) |,
       значение в [0, 1]: 1 — фазы идеально синхронны, 0 — независимы.

По умолчанию считается на широкополосном сигнале окна; для более строгого
анализа стоит сначала полосовым фильтром выделить конкретный ритм (напр.
alpha) — параметр band оставлен для этого.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, butter, filtfilt

from eeg_art.core.windowing import EEGWindow
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.features import FeatureSet
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry

PARAM_SCHEMA = {
    "band_low": ParamSpec(8.0, 0.5, 100.0, "Гц", "Нижняя граница полосы перед Гильберт-преобразованием (0 -> широкополосный PLV не поддерживается схемой, см. band_low=band_high=0 как особый случай в коде)."),
    "band_high": ParamSpec(13.0, 0.5, 100.0, "Гц", "Верхняя граница полосы; по умолчанию alpha (8-13 Гц) — классический диапазон для PLV в литературе."),
}


@MethodRegistry.register("connectivity_graph.feature_extractor", "plv", schema=PARAM_SCHEMA)
class ConnectivityFeatureExtractor:
    def __init__(self, band_low: float | None = 8.0, band_high: float | None = 13.0):
        """band_low=band_high=None -> широкополосный PLV; иначе полосовой
        фильтр перед Гильбертом."""
        self.band_low = band_low
        self.band_high = band_high

    def _bandpass(self, data: np.ndarray, fs: float) -> np.ndarray:
        nyq = fs / 2
        b, a = butter(4, [self.band_low / nyq, self.band_high / nyq], btype="band")
        return filtfilt(b, a, data, axis=1)

    def extract(self, window: EEGWindow, spatial: SpatialModel) -> FeatureSet:
        data = window.data
        if self.band_low is not None:
            data = self._bandpass(data, window.fs)

        analytic = hilbert(data, axis=1)
        phase = np.angle(analytic)  # (n_channels, n_samples)

        n_ch = phase.shape[0]
        plv = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                diff = phase[i] - phase[j]
                val = np.abs(np.mean(np.exp(1j * diff)))
                plv[i, j] = plv[j, i] = val
        np.fill_diagonal(plv, 1.0)

        return FeatureSet(
            timestamp=window.timestamp,
            fields={"plv_matrix": plv, "channels": window.channels},
        )
