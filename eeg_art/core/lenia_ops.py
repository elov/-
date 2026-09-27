"""
core.lenia_ops
-----------------
Базовые операции Lenia (кольцевое ядро, bell growth, FFT-свёртка),
вынесенные из modules.spectral_lenia, чтобы modules.microstates_regime
(и любой будущий модуль) мог использовать тот же механизм эволюции поля
с другим источником параметров, не дублируя код.
"""

from __future__ import annotations

import numpy as np


def ring_kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    r = np.sqrt(xx ** 2 + yy ** 2) / radius
    k = np.exp(-((r - 0.5) ** 2) / (2 * 0.15 ** 2))
    k[r > 1] = 0.0
    s = k.sum()
    return k / s if s > 0 else k


def growth_bell(u: np.ndarray, center: float, width: float) -> np.ndarray:
    return 2 * np.exp(-((u - center) ** 2) / (2 * width ** 2)) - 1


def fft_convolve_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    s1 = np.array(a.shape)
    s2 = np.array(k.shape)
    size = s1 + s2 - 1
    fsize = [int(2 ** np.ceil(np.log2(s))) for s in size]

    fa = np.fft.rfft2(a, fsize)
    fk = np.fft.rfft2(k, fsize)
    res = np.fft.irfft2(fa * fk, fsize)

    start = (s2 - 1) // 2
    end = start + s1
    return res[start[0]:end[0], start[1]:end[1]]
