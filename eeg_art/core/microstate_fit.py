"""
core.microstate_fit
----------------------
Кластеризация топографий ЭЭГ в канонические микросостояния — метод
Pascual-Marqui et al. (модифицированный k-means, инвариантный к полярности
топографии, т.к. для микросостояний знак топографии не имеет значения).

Шаги:
    1. Global Field Power (GFP) в каждый момент времени = std по каналам.
    2. Топографии в локальных максимумах GFP — самые устойчивые, наименее
       зашумлённые моменты (стандартная практика в литературе).
    3. Polarity-invariant k-means по этим топографиям -> k шаблонов.
"""

from __future__ import annotations

import numpy as np


def global_field_power(data: np.ndarray) -> np.ndarray:
    """data: (n_channels, n_samples) -> (n_samples,)"""
    return data.std(axis=0)


def gfp_peak_topographies(data: np.ndarray) -> np.ndarray:
    """Возвращает (n_peaks, n_channels) — топографии в локальных максимумах GFP."""
    gfp = global_field_power(data)
    peaks = []
    for t in range(1, len(gfp) - 1):
        if gfp[t] > gfp[t - 1] and gfp[t] > gfp[t + 1]:
            peaks.append(t)
    if not peaks:
        peaks = [int(np.argmax(gfp))]
    topo = data[:, peaks].T  # (n_peaks, n_channels)
    # усреднённая ссылка + нормировка на единичную норму (стандарт для микросостояний)
    topo = topo - topo.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(topo, axis=1, keepdims=True) + 1e-12
    return topo / norms


def fit_microstates(topographies: np.ndarray, k: int = 4, n_iter: int = 50, seed: int = 0) -> np.ndarray:
    """topographies: (n_peaks, n_channels), уже нормированные (см. gfp_peak_topographies).
    Возвращает (k, n_channels) — шаблоны микросостояний."""
    rng = np.random.default_rng(seed)
    n_peaks = topographies.shape[0]
    init_idx = rng.choice(n_peaks, size=k, replace=False)
    centroids = topographies[init_idx].copy()

    for _ in range(n_iter):
        # корреляция (топографии уже центрированы/нормированы -> dot product = cosine)
        corr = topographies @ centroids.T  # (n_peaks, k)
        labels = np.argmax(np.abs(corr), axis=1)

        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            members = topographies[labels == c]
            if len(members) == 0:
                new_centroids[c] = centroids[c]
                continue
            signs = np.sign(corr[labels == c, c])
            signs[signs == 0] = 1.0
            aligned = members * signs[:, None]
            mean_topo = aligned.mean(axis=0)
            norm = np.linalg.norm(mean_topo) + 1e-12
            new_centroids[c] = mean_topo / norm
        centroids = new_centroids

    return centroids


def label_topography(topo: np.ndarray, templates: np.ndarray) -> tuple[int, np.ndarray]:
    """topo: (n_channels,) сырая топография (например, среднее окно) ->
    (label, corr_per_template)."""
    t = topo - topo.mean()
    t = t / (np.linalg.norm(t) + 1e-12)
    corr = templates @ t  # (k,)
    label = int(np.argmax(np.abs(corr)))
    return label, corr
