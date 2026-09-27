"""
modules.pac_coupling.generator_pac
----------------------------------------
Generator ветви PAC.

Отличие от spectral_lenia принципиально: там каждый диапазон управлял
своим отдельным параметром (радиус, ширина, dt...) независимо. Здесь PAC
по определению — не самостоятельная величина, а мера того, НАСКОЛЬКО
фаза одного ритма определяет амплитуду другого. Поэтому и в генераторе
это должна быть не отдельная ручка, а модуляция одного параметра другим,
взвешенная по MI.

Механизм: growth_center поля в каждой точке колеблется вместе с фазой
медленного ритма (theta) в этой точке скальпа, с амплитудой колебания,
пропорциональной MI(x) * amp_fast(x):

    growth_center(x) = base_center + coupling_strength * MI(x) * amp_fast(x) * cos(phase_slow(x))

При MI=0 (связи нет) growth_center(x) = base_center везде — автомат
ведёт себя как обычный однородный Lenia. При высоком MI в какой-то
области скальпа локально возникает колебательная "накачка" роста —
видимая, пространственно локализованная сигнатура связи.

phase_slow интерполируется через cos/sin компоненты (см. spatial_model) —
это корректный способ интерполировать циклическую величину.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.features import FeatureSet
from eeg_art.core.field_abstraction import Grid
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.lenia_ops import ring_kernel, growth_bell, fft_convolve_same
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry

PARAM_SCHEMA = {
    "base_growth_center": ParamSpec(0.15, 0.0, 1.0, "",
        "Центр bell-функции роста при MI=0 (связи нет) — однородный Lenia."),
    "growth_width": ParamSpec(0.045, 0.001, 0.5, ""),
    "kernel_radius": ParamSpec(8, 1, 32, "px"),
    "dt": ParamSpec(0.12, 0.001, 1.0, ""),
    "noise_amp": ParamSpec(0.01, 0.0, 0.2, ""),
    "coupling_strength": ParamSpec(0.15, 0.0, 2.0, "",
        "Множитель MI(x)*amp_fast(x)*cos(phase(x)) в growth_center(x) — "
        "сила, с которой PAC-связь локально отклоняет рост от базового значения."),
}


@MethodRegistry.register("pac_coupling.generator", "default", schema=PARAM_SCHEMA)
class PACGenerator:
    def __init__(
        self,
        grid_size: int = 64,
        base_growth_center: float = 0.15,
        growth_width: float = 0.045,
        kernel_radius: int = 8,
        dt: float = 0.12,
        noise_amp: float = 0.01,
        coupling_strength: float = 0.15,
        rng=None,
    ):
        self.grid_size = grid_size
        self.base_growth_center = base_growth_center
        self.growth_width = growth_width
        self.kernel_radius = kernel_radius
        self.dt = dt
        self.noise_amp = noise_amp
        self.coupling_strength = coupling_strength
        self.rng = rng or np.random.default_rng()
        self._kernel = ring_kernel(kernel_radius)

        # последнее вычисленное поле MI — для панели признаков в демо
        # (не часть основного интерфейса Generator, вспомогательный атрибут)
        self.last_mi_field: np.ndarray | None = None
        self.last_growth_center_field: np.ndarray | None = None

    def init_state(self) -> Grid:
        return Grid.random_seed(self.grid_size, radius_frac=0.15, rng=self.rng)

    def step(self, features: FeatureSet, spatial: SpatialModel, state: Grid) -> Grid:
        mi = features.fields["pac_mi"]
        phase = features.fields["slow_phase"]
        amp = features.fields["fast_amp"]

        mi_field = spatial.interpolate_to_grid(mi, grid_size=self.grid_size)
        amp_field = spatial.interpolate_to_grid(amp, grid_size=self.grid_size)
        cos_field = spatial.interpolate_to_grid(
            {ch: float(np.cos(v)) for ch, v in phase.items()}, grid_size=self.grid_size
        )
        sin_field = spatial.interpolate_to_grid(
            {ch: float(np.sin(v)) for ch, v in phase.items()}, grid_size=self.grid_size
        )
        phase_field = np.arctan2(sin_field, cos_field)

        growth_center_field = (
            self.base_growth_center
            + self.coupling_strength * mi_field * amp_field * np.cos(phase_field)
        )
        self.last_mi_field = mi_field
        self.last_growth_center_field = growth_center_field

        convolved = fft_convolve_same(state.values, self._kernel)
        growth = growth_bell(convolved, growth_center_field, self.growth_width)
        noise = self.noise_amp * self.rng.standard_normal(state.values.shape)

        new_values = np.clip(state.values + self.dt * growth + noise, 0.0, 1.0)
        return Grid(values=new_values)
