"""
core.field_abstraction
------------------------
Единая абстракция представления пространственного состояния для
генераторов. Сейчас реализован только Grid (2D-сетка) — под
spectral_lenia и connectivity-подобные методы позже можно добавить Graph
и Volume, не меняя интерфейс Generator (core/generator.py).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Grid:
    values: np.ndarray  # (H, W), обычно в диапазоне [0, 1]

    @classmethod
    def zeros(cls, shape: int | tuple[int, int]) -> "Grid":
        h, w = (shape, shape) if isinstance(shape, int) else shape
        return cls(values=np.zeros((h, w), dtype=float))

    @classmethod
    def random_seed(cls, shape: int | tuple[int, int], radius_frac: float = 0.2, rng=None) -> "Grid":
        """Небольшой случайный 'зародыш' в центре — Lenia часто гаснет на
        пустом поле, стартовый паттерн нужен почти всегда. shape — int
        (квадрат, как раньше) или (H, W) для прямоугольного поля."""
        rng = rng or np.random.default_rng()
        h, w = (shape, shape) if isinstance(shape, int) else shape
        g = np.zeros((h, w))
        rh, rw = max(1, int(h * radius_frac)), max(1, int(w * radius_frac))
        cy, cx = h // 2, w // 2
        g[cy - rh:cy + rh, cx - rw:cx + rw] = rng.random((2 * rh, 2 * rw))
        return cls(values=g)


@dataclass
class Graph:
    """
    Вторая реализация той же абстракции поля: состояние живёт не на
    регулярной сетке, а прямо на узлах — электродах системы 10/20.
    Используется connectivity-модулями, где связи между узлами (не их
    координаты) определяют динамику.
    """
    node_names: list[str]
    node_values: np.ndarray       # (n_nodes,), обычно в [0, 1]
    edge_weights: np.ndarray      # (n_nodes, n_nodes), симметричная матрица связности

    @classmethod
    def random_seed(cls, node_names: list[str], edge_weights: np.ndarray, rng=None) -> "Graph":
        rng = rng or np.random.default_rng()
        n = len(node_names)
        return cls(
            node_names=node_names,
            node_values=rng.random(n) * 0.3,
            edge_weights=edge_weights,
        )

    def laplacian(self) -> np.ndarray:
        """Графовый лапласиан L = D - W — стандартный оператор диффузии по графу."""
        deg = np.diag(self.edge_weights.sum(axis=1))
        return deg - self.edge_weights


@dataclass
class Volume:
    """
    Третья реализация абстракции поля: состояние живёт в 3D-воксельной
    сетке (grid_size^3) — используется source_volumetric, где значения
    получены обратным решением (sLORETA) в объёме коры, а не на
    плоскости скальпа.
    """
    values: np.ndarray  # (D, H, W), обычно в [0, 1]

    @classmethod
    def zeros(cls, size: int) -> "Volume":
        return cls(values=np.zeros((size, size, size), dtype=float))

    @classmethod
    def random_seed(cls, size: int, radius_frac: float = 0.2, rng=None) -> "Volume":
        rng = rng or np.random.default_rng()
        v = np.zeros((size, size, size))
        r = max(1, int(size * radius_frac))
        c = size // 2
        v[c - r:c + r, c - r:c + r, c - r:c + r] = rng.random((2 * r, 2 * r, 2 * r))
        return cls(values=v)
