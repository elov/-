"""
modules.sonification.parameter_mapping
--------------------------------------------
Слой 2: художественная сонификация, симметричная spectral_lenia — те же
z-оценки band power (core.calibration.Baseline), что управляют Lenia,
здесь управляют амплитудами пяти осцилляторов (по одному на диапазон,
частоты разнесены на октаву — музыкально осмысленный, а не случайный
выбор). Панорама (лево/право) — по балансу лобных/затылочных каналов,
прямое использование реальной топологии, а не декоративный эффект.

Синтез идёт блоками, синхронно со sliding_windows того же шага, что и
визуальный ряд (step_sec) — один блок звука на один кадр генерации,
общая временная ось абсолютна (не по-блочный сброс фазы), поэтому
осцилляторы не щёлкают на границах блоков.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.windowing import sliding_windows
from eeg_art.core.calibration import Baseline
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry
from eeg_art.modules.spectral_lenia.features_spectral import SpectralFeatureExtractor

BAND_TONES = {  # частоты осцилляторов — по октаве друг от друга
    "delta": 110.0, "theta": 220.0, "alpha": 440.0, "beta": 880.0, "gamma": 1760.0,
}

FRONTAL = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"]
POSTERIOR = ["P7", "P3", "Pz", "P4", "P8", "O1", "O2"]

PARAM_SCHEMA = {
    "win_sec": ParamSpec(1.0, 0.2, 5.0, "с", "Длина окна анализа band power."),
    "step_sec": ParamSpec(0.15, 0.02, 2.0, "с", "Длина одного звукового блока / шаг между окнами."),
    "sample_rate": ParamSpec(44100, 8000, 96000, "Гц", "Частота дискретизации выходного аудио."),
}


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


@MethodRegistry.register("sonification.parameter_mapping", "band_oscillators", schema=PARAM_SCHEMA)
def render_parameter_mapping(
    stream: np.ndarray,
    channels: list[str],
    fs: float,
    baseline: Baseline,
    win_sec: float = 1.0,
    step_sec: float = 0.15,
    sample_rate: int = 44100,
) -> tuple[np.ndarray, int]:
    """Возвращает (stereo_audio (n_samples, 2) float32 в [-1,1], sample_rate)."""
    extractor = SpectralFeatureExtractor()
    spatial = SpatialModel()
    blocks = []
    global_sample = 0

    for window in sliding_windows(stream, channels, fs, win_sec=win_sec, step_sec=step_sec):
        raw = extractor.extract(window, spatial)
        z = baseline.zscore(raw.fields)["band_power"]

        block_len = int(round(step_sec * sample_rate))
        t = (global_sample + np.arange(block_len)) / sample_rate
        global_sample += block_len

        mono = np.zeros(block_len)
        for band, freq in BAND_TONES.items():
            z_mean = float(np.mean(list(z[band].values())))
            amp = _logistic(z_mean)
            mono += amp * np.sin(2 * np.pi * freq * t)
        mono /= len(BAND_TONES)

        # баланс лобные/затылочные — усреднённый по диапазонам z для каждой группы каналов
        frontal_vals = [z[b][ch] for b in z for ch in FRONTAL if ch in z[b]]
        posterior_vals = [z[b][ch] for b in z for ch in POSTERIOR if ch in z[b]]
        pan = np.tanh((np.mean(frontal_vals) - np.mean(posterior_vals)) / 2.0)

        left_gain = np.sqrt(max(0.0, (1 - pan) / 2))
        right_gain = np.sqrt(max(0.0, (1 + pan) / 2))
        blocks.append(np.stack([mono * left_gain, mono * right_gain], axis=1))

    audio = np.concatenate(blocks, axis=0)
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
    return audio.astype(np.float32), sample_rate
