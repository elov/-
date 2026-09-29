"""
modules.creation_stage.generator_dual_axis
------------------------------------------------
Двухосевой генератор ветви "создание": видео-стадия (медленная, дискретная
ось — какой сейчас этап работы над произведением) + band power ЭЭГ
(быстрая, непрерывная ось — состояние художника прямо сейчас).

Механизм:
    1. Стадия (см. stage_classifier.py) выбирает БАЗОВЫЙ пресет параметров
       Lenia — свой "визуальный жанр" на каждый этап (эскиз выглядит иначе,
       чем работа с цветом).
    2. band power (z-оценка, как в spectral_lenia) добавляет НЕБОЛЬШОЕ
       отклонение вокруг базового значения каждого параметра, ограниченное
       заданной долей — состояние художника окрашивает этап, но не
       подменяет его.

Это прямая реализация протокола исследования: "стойко воспроизводимые
состояния" (стадии) как основа, поверх которой видна более быстрая,
непрерывная физиологическая динамика.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.features import FeatureSet
from eeg_art.core.field_abstraction import Grid
from eeg_art.core.lenia_ops import ring_kernel, growth_bell, fft_convolve_same
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry
from eeg_art.modules.creation_stage.stage_classifier import STAGES

# Базовые пресеты — по одному "визуальному жанру" на стадию (см. словарь
# протокола). Подобраны для наглядно разного поведения; при желании
# каждый уточняется по данным конкретного художника.
STAGE_PRESETS = {
    "эскиз":            dict(kernel_radius=5,  growth_center=0.13, growth_width=0.025, dt=0.14, noise_amp=0.012),
    "подмалёвок":       dict(kernel_radius=12, growth_center=0.16, growth_width=0.07,  dt=0.07, noise_amp=0.003),
    "работа с цветом":  dict(kernel_radius=8,  growth_center=0.20, growth_width=0.05,  dt=0.16, noise_amp=0.015),
    "детализация":      dict(kernel_radius=4,  growth_center=0.15, growth_width=0.02,  dt=0.10, noise_amp=0.006),
    "пауза/подпись":    dict(kernel_radius=9,  growth_center=0.12, growth_width=0.06,  dt=0.04, noise_amp=0.001),
}

PARAM_SCHEMA = {
    "eeg_modulation_strength": ParamSpec(0.35, 0.0, 1.0, "",
        "Доля, на которую z-оценка band power может отклонить growth_center "
        "от базового значения стадии (0 = стадия жёстко фиксирует параметры, "
        "1 = ЭЭГ может полностью перекрыть базовый пресет)."),
}


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


@MethodRegistry.register("creation_stage.generator", "dual_axis", schema=PARAM_SCHEMA)
class DualAxisGenerator:
    def __init__(self, grid_size: int = 64, eeg_modulation_strength: float = 0.35, rng=None):
        self.grid_size = grid_size
        self.eeg_modulation_strength = eeg_modulation_strength
        self.rng = rng or np.random.default_rng()
        self.last_stage: str | None = None

    def init_state(self) -> Grid:
        return Grid.random_seed(self.grid_size, radius_frac=0.15, rng=self.rng)

    def step(self, stage: str, band_power_z: dict, state: Grid) -> Grid:
        preset = STAGE_PRESETS.get(stage, STAGE_PRESETS["детализация"])
        self.last_stage = stage

        # непрерывная модуляция: среднее по всем каналам каждого диапазона,
        # тот же принцип, что и в spectral_lenia
        band_scalar = {b: _logistic(float(np.mean(list(vals.values())))) - 0.5 for b, vals in band_power_z.items()}

        m = self.eeg_modulation_strength
        growth_center = preset["growth_center"] + m * 0.15 * band_scalar.get("beta", 0.0)
        growth_width = max(0.005, preset["growth_width"] + m * 0.03 * band_scalar.get("theta", 0.0))
        dt = max(0.01, preset["dt"] + m * 0.10 * band_scalar.get("delta", 0.0))
        noise_amp = max(0.0, preset["noise_amp"] + m * 0.01 * band_scalar.get("gamma", 0.0))
        kernel_radius = max(2, int(preset["kernel_radius"] + m * 4 * band_scalar.get("alpha", 0.0)))

        kernel = ring_kernel(kernel_radius)
        convolved = fft_convolve_same(state.values, kernel)
        growth = growth_bell(convolved, growth_center, growth_width)
        noise = noise_amp * self.rng.standard_normal(state.values.shape)

        new_values = np.clip(state.values + dt * growth + noise, 0.0, 1.0)
        return Grid(values=new_values)
