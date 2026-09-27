"""
modules.spectral_lenia.features_spectral
------------------------------------------
Первый feature-модуль ветви "спектральный -> Lenia": band power по
5 классическим диапазонам ЭЭГ, для каждого канала отдельно.

Метод: FFT окна с окном Ханна, суммирование мощности спектра в границах
диапазона. Это стандартный, задокументированный метод (эквивалент первого
шага Уэлча без усреднения по под-окнам) — сознательно простой для первого
среза; при необходимости заменить на scipy.signal.welch, не меняя интерфейс.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.windowing import EEGWindow
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.features import FeatureSet
from eeg_art.core.registry import MethodRegistry

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@MethodRegistry.register("spectral_lenia.feature_extractor", "fft")
class SpectralFeatureExtractor:
    def __init__(self, bands: dict[str, tuple[float, float]] | None = None):
        self.bands = bands or BANDS

    def extract(self, window: EEGWindow, spatial: SpatialModel) -> FeatureSet:
        data = window.data  # (n_channels, n_samples)
        n_ch, n_samp = data.shape
        fs = window.fs

        hann = np.hanning(n_samp)
        windowed = data * hann[None, :]
        spectrum = np.fft.rfft(windowed, axis=1)
        power = (np.abs(spectrum) ** 2) / n_samp
        freqs = np.fft.rfftfreq(n_samp, d=1.0 / fs)

        band_power: dict[str, dict[str, float]] = {}
        for band_name, (lo, hi) in self.bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            band_vals = power[:, mask].sum(axis=1)  # (n_channels,)
            band_power[band_name] = {
                ch: float(v) for ch, v in zip(window.channels, band_vals)
            }

        return FeatureSet(timestamp=window.timestamp, fields={"band_power": band_power})
