"""
modules.sonification.audification
--------------------------------------
Слой 1 (см. проектную дискуссию): аудификация, не сонификация — прямой
перенос сигнала в слышимый диапазон, БЕЗ интерпретации. ЭЭГ (0.5-45 Гц)
математически ускоряется в speed_factor раз, что транспонирует весь
спектр в слышимую область (0.5-45 Гц * 30 = 15-1350 Гц). Реализация:
сами отсчёты не меняются — меняется только заявленная частота
дискретизации при записи файла (буквальное ускорение воспроизведения).
Это звуковой аналог экрана "сырые каналы" — максимально прозрачный,
без художественной интерпретации.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry

PARAM_SCHEMA = {
    "speed_factor": ParamSpec(30.0, 1.0, 200.0, "x",
        "Во сколько раз ускоряется воспроизведение. ЭЭГ 0.5-45 Гц * 30 = "
        "15-1350 Гц (слышимый диапазон). Меньше 10x может остаться неслышимым внизу."),
}


@MethodRegistry.register("sonification.audification", "speed_shift", schema=PARAM_SCHEMA)
def audify(
    stream: np.ndarray,
    fs: float,
    speed_factor: float = 30.0,
    channels_to_mix: list[int] | None = None,
) -> tuple[np.ndarray, int]:
    """
    stream: (n_channels, n_samples).
    channels_to_mix: индексы каналов для усреднения в моно-дорожку;
        по умолчанию — все каналы (глобальная аудификация скальпа).
    Возвращает (audio_mono float32 в [-1,1], output_sample_rate).
    """
    if channels_to_mix is None:
        mono = stream.mean(axis=0)
    else:
        mono = stream[channels_to_mix].mean(axis=0)

    mono = mono / (np.max(np.abs(mono)) + 1e-9)
    output_sample_rate = int(round(fs * speed_factor))
    return mono.astype(np.float32), output_sample_rate
