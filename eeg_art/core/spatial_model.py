"""
core.spatial_model
-------------------
Координаты электродов системы 10/20 (упрощённая азимутальная проекция на
плоскость, аналогичная той, что используется в топокартах ЭЭГ) и
интерполяция значений из электродов в регулярное поле (Grid).

ВАЖНО: координаты ниже — упрощённая, но топологически корректная схема
(круг, Fpz сверху, Oz снизу, T7/T8 по бокам, Cz в центре). Для клинической
точности стоит заменить на реальный монтаж (например, standard_1020 из
MNE-Python), не меняя остальной код: он работает с любым словарём
{канал: (x, y)}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# Нормализованные координаты в единичном круге, y>0 — лобная область.
CHANNEL_POSITIONS_1020 = {
    "Fp1": (-0.31, 0.95), "Fp2": (0.31, 0.95),
    "F7":  (-0.81, 0.58), "F3":  (-0.55, 0.55), "Fz": (0.00, 0.50),
    "F4":  (0.55, 0.55),  "F8":  (0.81, 0.58),
    "T7":  (-1.00, 0.00), "C3":  (-0.55, 0.00), "Cz": (0.00, 0.00),
    "C4":  (0.55, 0.00),  "T8":  (1.00, 0.00),
    "P7":  (-0.81, -0.58), "P3": (-0.55, -0.55), "Pz": (0.00, -0.50),
    "P4":  (0.55, -0.55),  "P8": (0.81, -0.58),
    "O1":  (-0.31, -0.95), "O2": (0.31, -0.95),
}


@dataclass
class SpatialModel:
    """Хранит позиции каналов и умеет интерполировать значения в Grid."""

    channel_positions: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(CHANNEL_POSITIONS_1020)
    )

    @property
    def channels(self) -> list[str]:
        return list(self.channel_positions.keys())

    def positions_array(self) -> np.ndarray:
        """(n_channels, 2) массив координат в порядке self.channels."""
        return np.array([self.channel_positions[ch] for ch in self.channels])

    def interpolate_to_grid(
        self,
        values: dict[str, float],
        grid_size: int | tuple[int, int] = 64,
        power: float = 2.0,
        smoothing: float = 1e-6,
    ) -> np.ndarray:
        """
        Interpolate per-channel scalar values (например, мощность одного
        частотного диапазона) в регулярное поле, методом inverse distance
        weighting (IDW). grid_size — int (квадрат, как раньше) или (H, W)
        для прямоугольного поля (см. core.field_sizing) — интерполяция не
        предполагает квадратности, только совпадение системы координат
        электродов [-1,1]x[-1,1] с более длинной стороной поля.

        IDW выбран как простой, зависимостей не требующий и достаточно
        устойчивый метод для первого среза; при необходимости научной
        строгости стоит заменить на сферический сплайн (как в EEGLAB/MNE
        topomap) — интерфейс метода при этом не меняется.
        """
        chans = self.channels
        pts = self.positions_array()  # (n_channels, 2)
        vals = np.array([values[ch] for ch in chans], dtype=float)

        h, w = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        lin_x = np.linspace(-1.05, 1.05, w)
        lin_y = np.linspace(-1.05, 1.05, h)
        gx, gy = np.meshgrid(lin_x, lin_y)
        grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (N, 2)

        # (N, n_channels) расстояния от каждой точки сетки до каждого электрода
        dists = np.linalg.norm(
            grid_pts[:, None, :] - pts[None, :, :], axis=2
        )
        weights = 1.0 / (dists ** power + smoothing)
        weights /= weights.sum(axis=1, keepdims=True)
        field_flat = weights @ vals  # (N,)

        field = field_flat.reshape(h, w)

        # маска: за пределами круга головы поле обнуляем (косметика для рендера)
        mask = (gx ** 2 + gy ** 2) <= 1.05 ** 2
        field = np.where(mask, field, 0.0)
        return field
