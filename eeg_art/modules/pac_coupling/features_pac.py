"""
modules.pac_coupling.features_pac
--------------------------------------
FeatureExtractor ветви "кросс-частотное взаимодействие -> связь слоёв".

По умолчанию theta (фаза) -> gamma (амплитуда) — классическая пара в
литературе по PAC. Возвращает по каждому каналу: modulation index (сила
связи), фазу medленного ритма и нормированную амплитуду быстрого — это и
есть исходные данные для генератора, который свяжет ими "медленный" и
"быстрый" слои поля.
"""

from __future__ import annotations

from eeg_art.core.windowing import EEGWindow
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.features import FeatureSet
from eeg_art.core.pac import channel_pac


class PACFeatureExtractor:
    def __init__(self, phase_band=(4.0, 8.0), amp_band=(30.0, 45.0), n_bins: int = 18):
        self.phase_band = phase_band
        self.amp_band = amp_band
        self.n_bins = n_bins

    def extract(self, window: EEGWindow, spatial: SpatialModel) -> FeatureSet:
        mi, phase, amp = channel_pac(
            window.data, window.fs, self.phase_band, self.amp_band, self.n_bins
        )
        fields = {
            "pac_mi": {ch: float(v) for ch, v in zip(window.channels, mi)},
            "slow_phase": {ch: float(v) for ch, v in zip(window.channels, phase)},
            "fast_amp": {ch: float(v) for ch, v in zip(window.channels, amp)},
        }
        return FeatureSet(timestamp=window.timestamp, fields=fields)
