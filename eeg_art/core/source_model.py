"""
core.source_model
--------------------
Упрощённая локализация источников для ветви "3D объём".

ВАЖНО (как и spatial_model.py — тот же принцип честного упрощения):
здесь используется однородная безграничная среда (потенциал точечного
диполя V = 1/(4*pi*sigma*r)) вместо настоящей BEM/FEM-модели головы
(реальные слои: мозг/ликвор/череп/кожа с разной проводимостью, реальная
геометрия по MRI). Это даёт качественно верное поведение (чувствительность
падает с глубиной и расстоянием, поверхностные источники доминируют над
глубокими) ценой количественной точности. Для клинического использования
интерфейс (build_lead_field -> L, sloreta_source_power(L, v) -> power)
рассчитан на замену на mne.make_forward_solution + mne.minimum_norm без
изменения остального пайплайна.

Обратное решение — sLORETA (standardized low resolution electromagnetic
tomography, Pascual-Marqui 2002): взвешенная минимально-норменная оценка,
стандартизированная по диагонали матрицы разрешения — устоявшийся,
широко используемый метод, а не изобретённый с нуля.
"""

from __future__ import annotations

import numpy as np


def fibonacci_hemisphere(n_sources: int, radius: float = 0.8, z_min: float = 0.05, seed: int = 0) -> np.ndarray:
    """N точек, примерно равномерно покрывающих верхнюю полусферу (кора
    под шапочкой электродов, а не весь объём головы) на фиксированной
    глубине radius. Возвращает (n_sources, 3)."""
    n_full = n_sources * 3  # с запасом, т.к. половину (z<0) отбросим
    i = np.arange(n_full)
    phi = np.arccos(1 - 2 * (i + 0.5) / n_full)          # полярный угол, равномерно по площади
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * i

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    mask = z >= z_min
    pts = np.stack([x[mask], y[mask], z[mask]], axis=1)[:n_sources]
    return pts * radius


def sensor_positions_3d(channel_positions: dict[str, tuple[float, float]], radius: float = 1.0) -> np.ndarray:
    """2D-позиции 10/20 (см. spatial_model) -> 3D, проекцией на верхнюю
    полусферу той же радиальной модели, что и источники."""
    pts = []
    for x, y in channel_positions.values():
        r2 = x ** 2 + y ** 2
        z = np.sqrt(max(0.0, 1.0 - r2))
        pts.append((x, y, z))
    return np.array(pts) * radius


def build_lead_field(sensor_xyz: np.ndarray, source_xyz: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """L: (n_channels, n_sources). Простая модель потенциала точечного
    источника в однородной среде: V = 1 / (4*pi*sigma*r)."""
    diff = sensor_xyz[:, None, :] - source_xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=2) + 1e-3
    return 1.0 / (4 * np.pi * sigma * dist)


def sloreta_source_power(L: np.ndarray, v: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """
    L: (n_channels, n_sources) lead field.
    v: (n_channels,) потенциалы на скальпе в этот момент/окно.
    lam: коэффициент регуляризации Тихонова относительно масштаба L@L.T
         (типичная практика — доля от среднего собственного значения).
    Возвращает (n_sources,) — неотрицательную стандартизированную мощность
    источника (sLORETA statistic).
    """
    n_ch = L.shape[0]
    LLt = L @ L.T
    reg = lam * np.trace(LLt) / n_ch
    W = L.T @ np.linalg.inv(LLt + reg * np.eye(n_ch))   # (n_sources, n_channels) — MNE-оператор

    s_hat = W @ v                                        # (n_sources,) минимально-норменная оценка
    R = W @ L                                             # матрица разрешения (n_sources, n_sources)
    r_diag = np.clip(np.diag(R), 1e-9, None)

    return (s_hat ** 2) / r_diag                          # sLORETA: стандартизация по диагонали R


def interpolate_to_volume(
    source_xyz: np.ndarray,
    values: np.ndarray,
    grid_size: int = 20,
    extent: float = 1.05,
    power: float = 2.0,
) -> np.ndarray:
    """Аналог spatial_model.interpolate_to_grid, но в 3D: IDW из значений
    на источниках в регулярный воксельный куб grid_size^3."""
    lin = np.linspace(-extent, extent, grid_size)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (N, 3)

    dists = np.linalg.norm(grid_pts[:, None, :] - source_xyz[None, :, :], axis=2)
    weights = 1.0 / (dists ** power + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    vol_flat = weights @ values

    volume = vol_flat.reshape(grid_size, grid_size, grid_size)
    mask = (gx ** 2 + gy ** 2 + gz ** 2) <= extent ** 2
    return np.where(mask, volume, 0.0)
