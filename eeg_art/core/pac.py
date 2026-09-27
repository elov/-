"""
core.pac
----------
Phase-Amplitude Coupling: Modulation Index по методу Tort et al. (2010) —
устоявшаяся, широко цитируемая метрика PAC, не изобретённая с нуля.

Идея: если амплитуда быстрого ритма (например gamma) систематически
зависит от фазы медленного ритма (например theta), распределение
средней амплитуды по бинам фазы будет неравномерным. MI — нормированное
отклонение этого распределения от равномерного (через энтропию
Кульбака-Лейблера), в диапазоне [0, 1]: 0 — связи нет, больше — сильнее
связь. Именно поэтому PAC, в отличие от band power, не требует
baseline-калибровки — он уже нормирован по построению.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from eeg_art.core.dsp import bandpass_filter


def modulation_index(phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18) -> float:
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(phase, bin_edges) - 1, 0, n_bins - 1)

    mean_amp = np.zeros(n_bins)
    for b in range(n_bins):
        sel = amplitude[bin_idx == b]
        mean_amp[b] = sel.mean() if len(sel) > 0 else 0.0

    total = mean_amp.sum()
    if total <= 0:
        return 0.0
    p = mean_amp / total
    p_safe = np.clip(p, 1e-12, None)
    H = -np.sum(p_safe * np.log(p_safe))
    H_max = np.log(n_bins)
    return float((H_max - H) / H_max)


def channel_pac(
    data: np.ndarray,
    fs: float,
    phase_band: tuple[float, float] = (4.0, 8.0),
    amp_band: tuple[float, float] = (30.0, 45.0),
    n_bins: int = 18,
    ref_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data: (n_channels, n_samples).
    Возвращает (mi, ref_phase, ref_amp_norm) — по одному значению на канал:
        mi            — Modulation Index за всё окно
        ref_phase     — фаза медленного ритма в опорной точке окна (для
                         последующей пространственной интерполяции)
        ref_amp_norm  — амплитуда быстрого ритма в той же точке, нормированная
                         на максимум внутри окна (0..1)
    """
    phase_signal = bandpass_filter(data, fs, phase_band)
    amp_signal = bandpass_filter(data, fs, amp_band)

    phase = np.angle(hilbert(phase_signal, axis=-1))
    amp_env = np.abs(hilbert(amp_signal, axis=-1))

    n_ch = data.shape[0]
    mi = np.array([modulation_index(phase[i], amp_env[i], n_bins) for i in range(n_ch)])

    ref_index = ref_index if ref_index is not None else data.shape[1] // 2
    ref_phase = phase[:, ref_index]
    ref_amp = amp_env[:, ref_index]
    ref_amp_norm = ref_amp / (amp_env.max(axis=-1) + 1e-12)

    return mi, ref_phase, ref_amp_norm
