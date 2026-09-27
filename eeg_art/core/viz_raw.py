"""
core.viz_raw
--------------
Рендер сырого многоканального сигнала окна — левая панель триптиха
(«сырые данные»). Не интерпретирует сигнал, просто рисует трассы.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_art.core.windowing import EEGWindow


def render_raw_traces(window: EEGWindow, size_px: tuple[int, int] = (340, 512)) -> np.ndarray:
    w_px, h_px = size_px
    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    n_ch = window.data.shape[0]
    offset = 4.0
    for i, ch in enumerate(window.channels):
        trace = window.data[i]
        trace = trace / (np.abs(trace).max() + 1e-9)  # нормировка на канал для читаемости
        ax.plot(trace + (n_ch - i) * offset, color="lime", linewidth=0.6)

    ax.set_yticks([(n_ch - i) * offset for i in range(n_ch)])
    ax.set_yticklabels(window.channels, color="white", fontsize=6)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"raw  t={window.timestamp:0.2f}s", color="white", fontsize=8)
    fig.tight_layout(pad=0.3)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf
