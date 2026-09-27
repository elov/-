"""
modules.spectral_lenia.render_panel
--------------------------------------
Средняя панель триптиха для ветви spectral_lenia: bar chart band power
(в z-оценках после калибровки) по 5 диапазонам, усреднённых по каналам.
Это и есть видимая "партитура" — числа, которые непосредственно управляют
параметрами Lenia на этом кадре.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_band_power_panel(band_power_z: dict, size_px: tuple[int, int] = (300, 512)) -> np.ndarray:
    w_px, h_px = size_px
    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    bands = list(band_power_z.keys())
    values = [np.mean(list(band_power_z[b].values())) for b in bands]

    colors = ["#4444ff", "#44ffff", "#44ff44", "#ffff44", "#ff4444"]
    ax.barh(bands, values, color=colors[: len(bands)])
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_xlim(-3, 3)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_title("band power (z)", color="white", fontsize=8)
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf
