"""
core.dsp
----------
Общие DSP-утилиты. Полосовая фильтрация уже дублировалась в
connectivity_graph — вынесена сюда для pac_coupling и последующих
модулей.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(data: np.ndarray, fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    """data: (..., n_samples), фильтрация по последней оси."""
    lo, hi = band
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)
