"""
ingestion.synthetic_eeg
--------------------------
Синтетический генератор многоканального ЭЭГ-подобного сигнала для
разработки/демонстрации без реального устройства. Не претендует на
физиологическую точность — только на правдоподобную спектральную
структуру (доминирующие ритмы, которые "плывут" во времени), чтобы можно
было проверить весь пайплайн end-to-end.

Как только появится реальное устройство — этот модуль заменяется на
ingestion/lsl_eeg.py с тем же выходным форматом (n_channels, n_samples),
без изменения остального кода.
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_eeg(
    channels: list[str],
    fs: float = 250.0,
    duration_sec: float = 20.0,
    seed: int | None = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs

    band_freqs = {"delta": 2.0, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 35.0}

    data = np.zeros((len(channels), n_samples))
    for i, ch in enumerate(channels):
        signal = np.zeros(n_samples)
        for band, f0 in band_freqs.items():
            # амплитуда каждого ритма медленно "плывёт" во времени (разная для каждого канала)
            drift = 0.5 + 0.5 * np.sin(2 * np.pi * (0.02 + 0.01 * rng.random()) * t + rng.random() * 2 * np.pi)
            amp = drift * (1.0 + 0.3 * rng.standard_normal())
            phase = rng.random() * 2 * np.pi
            signal += amp * np.sin(2 * np.pi * f0 * t + phase)
        signal += 0.3 * rng.standard_normal(n_samples)  # широкополосный шум
        data[i] = signal

    return data


def generate_synthetic_eeg_with_pac(
    channels: list[str],
    fs: float = 250.0,
    duration_sec: float = 20.0,
    coupled_channels: list[str] | None = None,
    theta_freq: float = 6.0,
    gamma_freq: float = 35.0,
    coupling_depth: float = 0.85,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Как generate_synthetic_eeg, но для channels из coupled_channels
    амплитуда gamma-компоненты промодулирована фазой theta-компоненты —
    нужно, чтобы был сигнал с ЗАВЕДОМО присутствующей PAC-связью, на
    котором можно проверить, что core.pac.channel_pac действительно её
    обнаруживает (а не просто выдаёт правдоподобные с виду числа).
    """
    rng = np.random.default_rng(seed)
    coupled_channels = set(coupled_channels or [])
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs

    band_freqs = {"delta": 2.0, "theta": theta_freq, "alpha": 10.0, "beta": 20.0, "gamma": gamma_freq}

    data = np.zeros((len(channels), n_samples))
    for i, ch in enumerate(channels):
        signal = np.zeros(n_samples)
        theta_phase0 = rng.random() * 2 * np.pi
        theta_component = np.sin(2 * np.pi * theta_freq * t + theta_phase0)

        for band, f0 in band_freqs.items():
            if band == "gamma" and ch in coupled_channels:
                envelope = 1.0 + coupling_depth * np.cos(2 * np.pi * theta_freq * t + theta_phase0)
                envelope = np.clip(envelope, 0.0, None)
                phase_g = rng.random() * 2 * np.pi
                signal += 0.8 * envelope * np.sin(2 * np.pi * gamma_freq * t + phase_g)
                continue
            drift = 0.5 + 0.5 * np.sin(2 * np.pi * (0.02 + 0.01 * rng.random()) * t + rng.random() * 2 * np.pi)
            amp = drift * (1.0 + 0.3 * rng.standard_normal())
            phase = rng.random() * 2 * np.pi
            signal += amp * np.sin(2 * np.pi * f0 * t + phase)

        signal += 0.3 * rng.standard_normal(n_samples)
        data[i] = signal

    return data
