"""
modules.microstates_regime.generator_regime
-----------------------------------------------
Generator ветви "микросостояния -> дискретная смена режима".

В отличие от spectral_lenia (непрерывный мэппинг признак->параметр),
здесь параметры Lenia — это фиксированный, заранее заданный пресет на
каждый класс микросостояния ("партитура" в виде таблицы режимов, а не
формулы). Смена микросостояния между окнами -> резкая, видимая смена
поведения автомата, а не плавный дрейф.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.features import FeatureSet
from eeg_art.core.field_abstraction import Grid
from eeg_art.core.lenia_ops import ring_kernel, growth_bell, fft_convolve_same
from eeg_art.core.registry import MethodRegistry

# Таблица режимов — по одному пресету на класс микросостояния (A/B/C/D по
# умолчанию). Значения выбраны для наглядно различного поведения; в
# реальной системе стоит откалибровать по значимости каждого класса.
DEFAULT_PRESETS = {
    "A": dict(kernel_radius=6,  growth_center=0.12, growth_width=0.03, dt=0.10, noise_amp=0.005),
    "B": dict(kernel_radius=10, growth_center=0.20, growth_width=0.05, dt=0.15, noise_amp=0.010),
    "C": dict(kernel_radius=14, growth_center=0.15, growth_width=0.08, dt=0.08, noise_amp=0.002),
    "D": dict(kernel_radius=8,  growth_center=0.25, growth_width=0.02, dt=0.20, noise_amp=0.020),
}


@MethodRegistry.register("microstates_regime.generator", "default")
class MicrostateRegimeGenerator:
    def __init__(self, grid_size: int = 64, presets: dict | None = None, rng=None):
        self.grid_size = grid_size
        self.presets = presets or DEFAULT_PRESETS
        self.rng = rng or np.random.default_rng()

    def init_state(self) -> Grid:
        return Grid.random_seed(self.grid_size, radius_frac=0.15, rng=self.rng)

    def step(self, features: FeatureSet, spatial, state: Grid) -> Grid:
        label = features.fields["microstate_label"]
        preset = self.presets.get(label, next(iter(self.presets.values())))

        kernel = ring_kernel(preset["kernel_radius"])
        convolved = fft_convolve_same(state.values, kernel)
        growth = growth_bell(convolved, preset["growth_center"], preset["growth_width"])
        noise = preset["noise_amp"] * self.rng.standard_normal(state.values.shape)

        new_values = np.clip(state.values + preset["dt"] * growth + noise, 0.0, 1.0)
        return Grid(values=new_values)
