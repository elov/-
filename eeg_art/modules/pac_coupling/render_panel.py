"""
modules.pac_coupling.render_panel
--------------------------------------
Средняя панель триптиха: modulation index по каждому каналу — числа,
которые напрямую входят в growth_center_field генератора.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_pac_panel(mi_dict: dict, size_px: tuple[int, int] = (300, 512)) -> np.ndarray:
    w_px, h_px = size_px
    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    channels = list(mi_dict.keys())
    values = [mi_dict[ch] for ch in channels]
    order = np.argsort(values)
    channels = [channels[i] for i in order]
    values = [values[i] for i in order]

    colors = plt.cm.plasma(np.array(values) / (max(values) + 1e-9))
    ax.barh(channels, values, color=colors)
    ax.set_xlim(0, max(0.3, max(values) * 1.1))
    ax.tick_params(colors="white", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_title("PAC: theta->gamma MI", color="white", fontsize=8)
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf
