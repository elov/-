"""
modules.source_volumetric.lenia_3d
---------------------------------------
3D-аналоги core.lenia_ops (кольцевое ядро -> сферическая оболочка, FFT-
свёртка в 3D). growth_bell из core.lenia_ops переиспользуется как есть —
он поэлементный и не зависит от размерности.
"""

from __future__ import annotations

import numpy as np


def shell_kernel_3d(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    zz, yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2) / radius
    k = np.exp(-((r - 0.5) ** 2) / (2 * 0.15 ** 2))
    k[r > 1] = 0.0
    s = k.sum()
    return k / s if s > 0 else k


def fft_convolve_same_3d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    s1 = np.array(a.shape)
    s2 = np.array(k.shape)
    size = s1 + s2 - 1
    fsize = [int(2 ** np.ceil(np.log2(s))) for s in size]

    fa = np.fft.rfftn(a, fsize)
    fk = np.fft.rfftn(k, fsize)
    res = np.fft.irfftn(fa * fk, fsize)

    start = (s2 - 1) // 2
    end = start + s1
    return res[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
