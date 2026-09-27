"""
modules.palette_extraction.color_space
------------------------------------------
sRGB <-> CIE Lab (D65), векторизовано на numpy. Lab выбран для
кластеризации, а не RGB, т.к. евклидово расстояние в Lab приблизительно
соответствует воспринимаемой разнице цветов — важно, чтобы кластеры
действительно соответствовали "визуально разным" цветам, а не просто
разным числам в RGB.
"""

from __future__ import annotations

import numpy as np

_D65 = (0.95047, 1.0, 1.08883)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)


def rgb_to_xyz(rgb01: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(rgb01)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return lin @ M.T


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz / np.array(_D65)
    delta = 6 / 29

    def f(t):
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    return xyz_to_lab(rgb_to_xyz(rgb01))


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    delta = 6 / 29

    def finv(t):
        return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * np.array(_D65)
    return xyz


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    M_inv = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ])
    lin = xyz @ M_inv.T
    return linear_to_srgb(lin)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(xyz_to_rgb(lab_to_xyz(lab)), 0, 1)
