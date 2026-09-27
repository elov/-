"""
modules.connectivity_graph.render
-------------------------------------
Graph -> numpy image array (H, W, 3), чтобы использовать тот же
core.session_log.SessionLog, что и для Grid (spectral_lenia). Это и есть
практическая проверка расширяемости: core ничего не знает про то, как
устроен рендер конкретного модуля, лишь бы на выходе был массив.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_art.core.field_abstraction import Graph
from eeg_art.core.spatial_model import SpatialModel


def render_graph(graph: Graph, spatial: SpatialModel, size_px: int = 512, edge_threshold: float = 0.5) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(size_px / 100, size_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    pos = {ch: spatial.channel_positions[ch] for ch in graph.node_names}
    n = len(graph.node_names)

    for i in range(n):
        for j in range(i + 1, n):
            w = graph.edge_weights[i, j]
            if w < edge_threshold:
                continue
            xi, yi = pos[graph.node_names[i]]
            xj, yj = pos[graph.node_names[j]]
            ax.plot([xi, xj], [yi, yj], color="cyan", alpha=float(w), linewidth=1.2)

    xs = [pos[ch][0] for ch in graph.node_names]
    ys = [pos[ch][1] for ch in graph.node_names]
    sizes = 200 + 800 * graph.node_values
    ax.scatter(xs, ys, s=sizes, c=graph.node_values, cmap="inferno", vmin=0, vmax=1, edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf
