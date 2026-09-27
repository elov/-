"""
modules.connectivity_graph.render_panel
------------------------------------------
Средняя панель триптиха для ветви connectivity_graph: тепловая карта
PLV-матрицы (каналы x каналы) — числа, которые напрямую становятся
весами рёбер графа-автомата.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_plv_panel(plv_matrix: np.ndarray, channels: list[str], size_px: tuple[int, int] = (340, 512)) -> np.ndarray:
    w_px, h_px = size_px
    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")

    im = ax.imshow(plv_matrix, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(channels)))
    ax.set_yticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=90, fontsize=5, color="white")
    ax.set_yticklabels(channels, fontsize=5, color="white")
    ax.set_title("PLV (alpha)", color="white", fontsize=8)
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf
