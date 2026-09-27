"""
modules.source_volumetric.generator_volume
------------------------------------------------
Generator ветви "локализация источников -> 3D-автомат".

Пайплайн: scalp_activity (вектор по каналам) -> sLORETA -> мощность по
источникам на коре -> IDW-интерполяция в воксельный куб -> нормировка
-> growth_center_field (по аналогии со spectral_lenia, только в 3D:
карта источников напрямую сообщает, в каких зонах объёма усиливается
рост автомата) -> 3D-свёртка со сферическим ядром -> growth_bell -> шаг.

Lead field и позиции источников строятся один раз в __init__ (дорогая
часть, менять от кадра к кадру незачем — сама физическая модель головы
не меняется, меняются только потенциалы на скальпе).
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.features import FeatureSet
from eeg_art.core.field_abstraction import Volume
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.lenia_ops import growth_bell
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry
from eeg_art.core.source_model import (
    fibonacci_hemisphere, sensor_positions_3d, build_lead_field,
    sloreta_source_power, interpolate_to_volume,
)
from eeg_art.modules.source_volumetric.lenia_3d import shell_kernel_3d, fft_convolve_same_3d

PARAM_SCHEMA = {
    "grid_size": ParamSpec(20, 8, 40, "voxels", "Размер стороны воксельного куба; растёт как n^3 по стоимости."),
    "n_sources": ParamSpec(300, 50, 1000, "", "Число точек-источников на полусфере коры."),
    "base_growth_center": ParamSpec(0.15, 0.0, 1.0, ""),
    "growth_width": ParamSpec(0.05, 0.001, 0.5, ""),
    "kernel_radius": ParamSpec(3, 1, 10, "voxels"),
    "dt": ParamSpec(0.07, 0.001, 1.0, ""),
    "noise_amp": ParamSpec(0.008, 0.0, 0.2, ""),
    "coupling_strength": ParamSpec(0.30, 0.0, 2.0, "", "Насколько сильно карта мощности источников сдвигает growth_center."),
}


@MethodRegistry.register("source_volumetric.generator", "sloreta", schema=PARAM_SCHEMA)
class SourceVolumeGenerator:
    def __init__(
        self,
        spatial: SpatialModel,
        grid_size: int = 20,
        n_sources: int = 300,
        base_growth_center: float = 0.15,
        growth_width: float = 0.05,
        kernel_radius: int = 3,
        dt: float = 0.07,
        noise_amp: float = 0.008,
        coupling_strength: float = 0.30,
        rng=None,
    ):
        self.grid_size = grid_size
        self.base_growth_center = base_growth_center
        self.growth_width = growth_width
        self.dt = dt
        self.noise_amp = noise_amp
        self.coupling_strength = coupling_strength
        self.rng = rng or np.random.default_rng()

        # --- физическая модель (строится один раз) ---
        self.channels = spatial.channels
        self.source_xyz = fibonacci_hemisphere(n_sources, radius=0.8)
        self.sensor_xyz = sensor_positions_3d(spatial.channel_positions, radius=1.0)
        self.L = build_lead_field(self.sensor_xyz, self.source_xyz)
        self._kernel = shell_kernel_3d(kernel_radius)

        self.last_source_power: np.ndarray | None = None  # для панели/отладки

    def init_state(self) -> Volume:
        return Volume.random_seed(self.grid_size, radius_frac=0.15, rng=self.rng)

    def step(self, features: FeatureSet, spatial: SpatialModel, state: Volume) -> Volume:
        activity = features.fields["scalp_activity"]
        v = np.array([activity[ch] for ch in self.channels])

        source_power = sloreta_source_power(self.L, v)
        self.last_source_power = source_power

        # нормировка кадра (min-max по текущему распределению мощности
        # источников) -- sLORETA уже частично стандартизирует, доп.
        # baseline здесь не строго обязателен, в отличие от band power
        p = source_power - source_power.min()
        p = p / (p.max() + 1e-9)

        vol_field = interpolate_to_volume(self.source_xyz, p, grid_size=self.grid_size)
        growth_center_field = self.base_growth_center + self.coupling_strength * vol_field

        convolved = fft_convolve_same_3d(state.values, self._kernel)
        growth = growth_bell(convolved, growth_center_field, self.growth_width)
        noise = self.noise_amp * self.rng.standard_normal(state.values.shape)

        new_values = np.clip(state.values + self.dt * growth + noise, 0.0, 1.0)
        return Volume(values=new_values)
