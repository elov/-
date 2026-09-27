"""
modules.palette_extraction.render_palette
----------------------------------------------
Визуализация результата extract_palette: полосы основных цветов сверху,
акцентных снизу, с подписанными долями площади — для проверки/выставочной
подписи ("партитура" палитры).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_art.modules.palette_extraction.extract import Palette


def render_palette(palette: Palette, size_px: tuple[int, int] = (300, 512)) -> np.ndarray:
    w_px, h_px = size_px
    fig, axes = plt.subplots(2, 1, figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")

    for ax, colors, title in zip(axes, [palette.main, palette.accent], ["основные (main)", "акцентные (accent)"]):
        ax.set_facecolor("black")
        n = max(len(colors), 1)
        for i, c in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c.rgb))
            ax.text(i + 0.5, -0.15, f"{c.area_frac*100:.1f}%", color="white",
                    ha="center", fontsize=8)
        ax.set_xlim(0, n)
        ax.set_ylim(-0.3, 1)
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=10, loc="left")

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf
