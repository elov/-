"""
modules.spectral_lenia.features_spectral_welch
---------------------------------------------------
Второй метод в роли "spectral_lenia.feature_extractor" — Метод Уэлча
(scipy.signal.welch): сигнал окна делится на перекрывающиеся под-окна,
спектр усредняется — ниже дисперсия оценки мощности по сравнению с
одним "голым" FFT всего окна (features_spectral.py, метод "fft"),
ценой немного худшего частотного разрешения. Стандартная замена,
упомянутая как рекомендация ещё в первой версии features_spectral.py.

Пример того, как добавляется альтернативный метод в существующую роль:
тот же интерфейс FeatureExtractor, та же регистрационная категория,
другое имя ("welch" вместо "fft") — выбирается на месте вызова.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from eeg_art.core.windowing import EEGWindow
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.features import FeatureSet
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry
from eeg_art.modules.spectral_lenia.features_spectral import BANDS

PARAM_SCHEMA = {
    "n_segments": ParamSpec(4, 1, 16, "",
        "На сколько перекрывающихся под-окон делится окно для усреднения "
        "спектра (метод Уэлча). Больше -> ниже дисперсия оценки, но грубее "
        "частотное разрешение (типичный компромисс для коротких окон ЭЭГ)."),
}


@MethodRegistry.register("spectral_lenia.feature_extractor", "welch", schema=PARAM_SCHEMA)
class WelchFeatureExtractor:
    def __init__(self, bands: dict[str, tuple[float, float]] | None = None, n_segments: int = 4):
        self.bands = bands or BANDS
        self.n_segments = n_segments

    def extract(self, window: EEGWindow, spatial: SpatialModel) -> FeatureSet:
        n_samp = window.data.shape[1]
        nperseg = max(16, n_samp // self.n_segments)

        freqs, psd = welch(window.data, fs=window.fs, nperseg=nperseg, axis=1)

        band_power: dict[str, dict[str, float]] = {}
        for band_name, (lo, hi) in self.bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            band_vals = psd[:, mask].sum(axis=1)
            band_power[band_name] = {
                ch: float(v) for ch, v in zip(window.channels, band_vals)
            }

        return FeatureSet(timestamp=window.timestamp, fields={"band_power": band_power})
