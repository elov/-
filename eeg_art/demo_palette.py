"""
demo_palette.py
-------------------
Извлечение палитры с выбором количества основных и акцентных цветов.
Число цветов каждого типа задаётся отдельно — как и просили: n_main и
n_accent независимы.

По умолчанию использует синтетическое тестовое "полотно" (нет реальной
загруженной картины). Чтобы применить к реальному изображению — раскомментировать
load_real_image(path) и передать свой путь.

Запуск:
    python3 demo_palette.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from eeg_art.modules.palette_extraction.extract import extract_palette
from eeg_art.modules.palette_extraction.render_palette import render_palette


def make_synthetic_artwork(size: int = 256, seed: int = 0) -> np.ndarray:
    """Синтетическое 'полотно': спокойный градиентный фон (кандидаты в main)
    + несколько крупных приглушённых пятен (main) + мелкие яркие акценты
    (кандидаты в accent — маленькая площадь, высокая насыщенность)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size

    img = np.zeros((size, size, 3))
    img[..., 0] = 0.15 + 0.10 * yy          # тёмно-синий фон
    img[..., 1] = 0.18 + 0.12 * yy
    img[..., 2] = 0.30 + 0.15 * (1 - yy)

    # крупное приглушённое зелёное пятно (main-кандидат)
    cy, cx, r = size * 0.6, size * 0.4, size * 0.25
    mask = (yy * size - cy) ** 2 + (xx * size - cx) ** 2 < r ** 2
    img[mask] = [0.25, 0.45, 0.30]

    # крупное приглушённое охра-пятно (main-кандидат)
    cy, cx, r = size * 0.3, size * 0.65, size * 0.2
    mask = (yy * size - cy) ** 2 + (xx * size - cx) ** 2 < r ** 2
    img[mask] = [0.55, 0.40, 0.20]

    # несколько мелких ярких акцентов
    for color, (cy_f, cx_f) in zip(
        [[0.95, 0.10, 0.10], [0.98, 0.85, 0.05], [0.95, 0.15, 0.75]],
        [(0.2, 0.2), (0.75, 0.7), (0.5, 0.85)],
    ):
        cy, cx, r = size * cy_f, size * cx_f, size * 0.045
        mask = (yy * size - cy) ** 2 + (xx * size - cx) ** 2 < r ** 2
        img[mask] = color

    img += rng.normal(scale=0.01, size=img.shape)
    return np.clip(img, 0, 1)


def load_real_image(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im).astype(float) / 255.0


def main():
    n_main = 4      # <-- сколько основных цветов
    n_accent = 3    # <-- сколько акцентных цветов

    print("Синтетическое тестовое полотно (нет загруженного изображения) ...")
    image = make_synthetic_artwork()
    Image.fromarray((image * 255).astype(np.uint8)).save("/home/claude/synthetic_artwork.png")

    palette = extract_palette(image, n_main=n_main, n_accent=n_accent, k_clusters=16, min_area_frac=0.001)

    print(f"\nОсновные цвета ({len(palette.main)}):")
    for c in palette.main:
        print(f"  RGB={tuple(round(v,3) for v in c.rgb)}  площадь={c.area_frac*100:.1f}%  chroma={c.chroma:.1f}")

    print(f"\nАкцентные цвета ({len(palette.accent)}):")
    for c in palette.accent:
        print(f"  RGB={tuple(round(v,3) for v in c.rgb)}  площадь={c.area_frac*100:.1f}%  chroma={c.chroma:.1f}")

    swatch = render_palette(palette)
    Image.fromarray((swatch * 255).astype(np.uint8)).save("/home/claude/palette_swatch.png")
    print("\nСохранено: synthetic_artwork.png, palette_swatch.png")


if __name__ == "__main__":
    main()
