"""
modules.palette_extraction.extract
--------------------------------------
Извлечение палитры из изображения (наблюдаемой картины или холста
художника), с раздельным управляемым числом:
    - основные цвета (main)   — по доле площади изображения
    - акцентные цвета (accent) — по насыщенности (chroma в Lab) среди
      цветов, не попавших в основные, при этом занимающих заметную,
      не шумовую площадь

Метод:
    1. k-means в CIE Lab (k заметно больше n_main+n_accent, чтобы не
       усреднять визуально разные пятна в один кластер раньше времени).
    2. Кластеры сортируются по доле площади -> первые n_main это основные.
    3. Из оставшихся кластеров с площадью выше min_area_frac (защита от
       шумовых единичных пикселей) выбираются n_accent с наибольшим
       chroma = sqrt(a^2+b^2) в Lab — это и есть "акцент": заметный по
       насыщенности, но не преобладающий по площади цвет.
    4. Оба списка сортируются по кругу оттенков (hue в Lab: atan2(b,a))
       для консистентного порядка при последующем маппинге в рендер.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from eeg_art.modules.palette_extraction.color_space import rgb_to_lab, lab_to_rgb
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry


@dataclass
class PaletteColor:
    rgb: tuple[float, float, float]
    lab: tuple[float, float, float]
    area_frac: float
    chroma: float


@dataclass
class Palette:
    main: list[PaletteColor]
    accent: list[PaletteColor]


def _kmeans_pp_init(data: np.ndarray, k: int, rng) -> np.ndarray:
    """k-means++ инициализация: каждый следующий центр выбирается с
    вероятностью, пропорциональной квадрату расстояния до ближайшего уже
    выбранного центра. Критично для этой задачи: маленькие, но заметно
    отличающиеся по цвету пятна (кандидаты в акцент) при случайной
    инициализации почти никогда не получают свой кластер и тонут в фоне."""
    n = data.shape[0]
    centers = [data[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            np.linalg.norm(data[:, None, :] - np.array(centers)[None, :, :], axis=2) ** 2,
            axis=1,
        )
        probs = d2 / (d2.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centers.append(data[idx])
    return np.array(centers)


def _kmeans_lab(pixels_lab: np.ndarray, k: int, n_iter: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """k-means (k-means++ init) в Lab. Возвращает (labels, centers)."""
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(pixels_lab, k, rng)

    for _ in range(n_iter):
        dists = np.linalg.norm(pixels_lab[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for c in range(k):
            members = pixels_lab[labels == c]
            if len(members) > 0:
                new_centers[c] = members.mean(axis=0)
        if np.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers

    return labels, centers


def _hue(lab: np.ndarray) -> float:
    return float(np.arctan2(lab[2], lab[1]))


def _merge_similar_clusters(
    centers: np.ndarray, counts: np.ndarray, threshold: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """Объединяет кластеры, чей центр в Lab различается меньше threshold
    (примерно 'едва различимая разница цвета' по ощущению глаза, delta-E).
    Без этого k-means иногда дробит одно визуально однородное пятно на
    два близких кластера, и оба попадают в 'main' как будто это разные цвета."""
    centers = centers.copy()
    counts = counts.copy()
    while len(centers) > 1:
        dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        if dists[i, j] >= threshold:
            break
        total = counts[i] + counts[j]
        centers[i] = (centers[i] * counts[i] + centers[j] * counts[j]) / total
        counts[i] = total
        centers = np.delete(centers, j, axis=0)
        counts = np.delete(counts, j, axis=0)
    return centers, counts


PALETTE_PARAM_SCHEMA = {
    "n_main": ParamSpec(4, 1, 12, "", "Число основных цветов (по площади)."),
    "n_accent": ParamSpec(2, 0, 8, "", "Число акцентных цветов (по насыщенности среди некрупных пятен)."),
    "min_area_frac": ParamSpec(0.005, 0.0001, 0.2, "", "Мин. доля площади кластера, чтобы не считаться шумом."),
}


@MethodRegistry.register("palette_extraction.extract", "kmeans_lab", schema=PALETTE_PARAM_SCHEMA)
def extract_palette(
    image_rgb01: np.ndarray,
    n_main: int = 4,
    n_accent: int = 2,
    k_clusters: int | None = None,
    min_area_frac: float = 0.005,
    max_pixels_for_fit: int = 20000,
    seed: int = 0,
) -> Palette:
    """
    image_rgb01: (H, W, 3), значения в [0, 1].
    n_main, n_accent: сколько цветов каждого типа вернуть (управляется вызывающим кодом).
    k_clusters: число кластеров k-means; по умолчанию max(n_main+n_accent+4, 10) —
        с запасом, чтобы акцентные пятна не усреднялись с фоном раньше классификации.
    """
    if k_clusters is None:
        k_clusters = max(n_main + n_accent + 4, 10)

    h, w, _ = image_rgb01.shape
    flat_rgb = image_rgb01.reshape(-1, 3)
    total_pixels = flat_rgb.shape[0]

    rng = np.random.default_rng(seed)
    if total_pixels > max_pixels_for_fit:
        idx = rng.choice(total_pixels, size=max_pixels_for_fit, replace=False)
        fit_rgb = flat_rgb[idx]
    else:
        fit_rgb = flat_rgb

    fit_lab = rgb_to_lab(fit_rgb)
    labels, centers_lab = _kmeans_lab(fit_lab, k_clusters, seed=seed)
    counts = np.bincount(labels, minlength=k_clusters)

    centers_lab, counts = _merge_similar_clusters(centers_lab, counts, threshold=8.0)
    area_frac = counts / counts.sum()

    order_by_area = np.argsort(-area_frac)
    main_idx = list(order_by_area[:n_main])

    remaining = [i for i in order_by_area if i not in main_idx and area_frac[i] >= min_area_frac]
    chroma = np.sqrt(centers_lab[:, 1] ** 2 + centers_lab[:, 2] ** 2)
    remaining_sorted_by_chroma = sorted(remaining, key=lambda i: -chroma[i])
    accent_idx = remaining_sorted_by_chroma[:n_accent]

    def build(idx_list: list[int]) -> list[PaletteColor]:
        colors = []
        for i in idx_list:
            lab = centers_lab[i]
            rgb = lab_to_rgb(lab)
            colors.append(PaletteColor(
                rgb=tuple(float(v) for v in rgb),
                lab=tuple(float(v) for v in lab),
                area_frac=float(area_frac[i]),
                chroma=float(chroma[i]),
            ))
        colors.sort(key=lambda c: _hue(np.array(c.lab)))
        return colors

    return Palette(main=build(main_idx), accent=build(accent_idx))
