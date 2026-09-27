"""
modules.spectral_lenia.generator_lenia
-----------------------------------------
Генератор ветви "спектральный -> Lenia".

Мэппинг признак -> параметр (это и есть "партитура" метода, должна
публиковаться рядом с работой на выставке):

    alpha (среднее по полю)  -> радиус ядра свёртки
    beta  (среднее по полю)  -> центр функции роста (growth center)
    theta (среднее по полю)  -> ширина функции роста (growth width)
    gamma (среднее по полю)  -> амплитуда шума на шаге обновления
    delta (среднее по полю)  -> dt (скорость эволюции автомата)

Границы каждого диапазона маппинга — настраиваемые параметры (см.
PARAM_SCHEMA), а не зашитые числа: их можно менять по науке (например,
по данным о типичном диапазоне мощности альфа-ритма у конкретного
человека) и сохранять как именованный пресет через core.config.

Реализация — стандартная схема Lenia: state (H,W) в [0,1], свёртка с
кольцевым ядром, колоколообразная growth-функция, обновление
state += dt * growth(convolved) + noise, клип в [0,1].
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.features import FeatureSet
from eeg_art.core.field_abstraction import Grid
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.lenia_ops import ring_kernel, growth_bell, fft_convolve_same
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry


def _map_range(x: float, lo: float, hi: float, x_lo: float = 0.0, x_hi: float = 1.0) -> float:
    x = max(x_lo, min(x_hi, x))
    t = (x - x_lo) / (x_hi - x_lo + 1e-9)
    return lo + t * (hi - lo)


def _normalize(field: np.ndarray) -> float:
    """Скаляр из поля: среднее по полю, пропущенное через логистическую
    функцию. Ожидается, что field уже в z-оценках относительно baseline
    (core.calibration.Baseline.zscore) — тогда 0 соответствует норме
    человека, а не произвольной константе."""
    m = float(field.mean())
    return 1.0 / (1.0 + np.exp(-m))


PARAM_SCHEMA = {
    "kernel_radius_min": ParamSpec(4, 1, 32, "px",
        "Мин. радиус кольцевого ядра свёртки при alpha (z-оценка) -> -inf. "
        "Меньше — мельче структуры паттерна."),
    "kernel_radius_max": ParamSpec(16, 1, 32, "px",
        "Макс. радиус ядра при alpha -> +inf. Должен быть заметно меньше grid_size/2."),
    "growth_center_min": ParamSpec(0.10, 0.0, 1.0, "",
        "Мин. центр bell-функции роста при beta -> -inf. Низкие значения -> реже устойчивые формы."),
    "growth_center_max": ParamSpec(0.25, 0.0, 1.0, "",
        "Макс. центр функции роста при beta -> +inf."),
    "growth_width_min": ParamSpec(0.02, 0.001, 0.5, "",
        "Мин. ширина функции роста при theta -> -inf. Уже -> резче переходы между состояниями клетки."),
    "growth_width_max": ParamSpec(0.06, 0.001, 0.5, "",
        "Макс. ширина функции роста при theta -> +inf."),
    "noise_amp_min": ParamSpec(0.0, 0.0, 0.2, "",
        "Мин. амплитуда шума обновления при gamma -> -inf."),
    "noise_amp_max": ParamSpec(0.02, 0.0, 0.2, "",
        "Макс. амплитуда шума при gamma -> +inf. Выше — быстрее теряется устойчивая структура."),
    "dt_min": ParamSpec(0.05, 0.001, 1.0, "",
        "Мин. шаг эволюции автомата за одно окно при delta -> -inf."),
    "dt_max": ParamSpec(0.25, 0.001, 1.0, "",
        "Макс. шаг эволюции при delta -> +inf. Выше — автомат быстрее уходит в насыщение."),
}


@MethodRegistry.register("spectral_lenia.generator", "default", schema=PARAM_SCHEMA)
class LeniaGenerator:
    def __init__(
        self,
        grid_size: int = 64,
        field_shape: tuple[int, int] | None = None,
        kernel_radius_min: float = 4, kernel_radius_max: float = 16,
        growth_center_min: float = 0.10, growth_center_max: float = 0.25,
        growth_width_min: float = 0.02, growth_width_max: float = 0.06,
        noise_amp_min: float = 0.0, noise_amp_max: float = 0.02,
        dt_min: float = 0.05, dt_max: float = 0.25,
        rng=None,
    ):
        # field_shape=(H,W), если задан, перекрывает квадратный grid_size —
        # это и есть точка входа для core.field_sizing (поле соразмерно
        # пропорциям конкретного произведения). По умолчанию не задан ->
        # поведение точно как раньше (квадрат grid_size x grid_size).
        self.shape: tuple[int, int] = field_shape if field_shape is not None else (grid_size, grid_size)
        self.grid_size = grid_size  # сохраняем для обратной совместимости/схемы пресетов
        self.kernel_radius_min, self.kernel_radius_max = kernel_radius_min, kernel_radius_max
        self.growth_center_min, self.growth_center_max = growth_center_min, growth_center_max
        self.growth_width_min, self.growth_width_max = growth_width_min, growth_width_max
        self.noise_amp_min, self.noise_amp_max = noise_amp_min, noise_amp_max
        self.dt_min, self.dt_max = dt_min, dt_max
        self.rng = rng or np.random.default_rng()

    def init_state(self) -> Grid:
        return Grid.random_seed(self.shape, radius_frac=0.15, rng=self.rng)

    def step(self, features: FeatureSet, spatial: SpatialModel, state: Grid) -> Grid:
        band_power = features.fields["band_power"]  # {band: {channel: value}}

        band_scalar = {}
        for band, per_channel in band_power.items():
            field = spatial.interpolate_to_grid(per_channel, grid_size=self.shape)
            band_scalar[band] = _normalize(field)

        kernel_radius = int(_map_range(band_scalar.get("alpha", 0.5), self.kernel_radius_min, self.kernel_radius_max))
        growth_center = _map_range(band_scalar.get("beta", 0.5), self.growth_center_min, self.growth_center_max)
        growth_width = _map_range(band_scalar.get("theta", 0.5), self.growth_width_min, self.growth_width_max)
        noise_amp = _map_range(band_scalar.get("gamma", 0.5), self.noise_amp_min, self.noise_amp_max)
        dt = _map_range(band_scalar.get("delta", 0.5), self.dt_min, self.dt_max)

        kernel = ring_kernel(max(2, kernel_radius))
        convolved = fft_convolve_same(state.values, kernel)
        growth = growth_bell(convolved, growth_center, growth_width)
        noise = noise_amp * self.rng.standard_normal(state.values.shape)

        new_values = np.clip(state.values + dt * growth + noise, 0.0, 1.0)
        return Grid(values=new_values)
