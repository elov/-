"""
modules.source_volumetric.render_volume
----------------------------------------------
Рендер 3D-состояния для триптиха. Полный объём не показать на плоском
экране, поэтому — три ортогональные проекции максимальной интенсивности
(MIP, как в медицинской визуализации): вид сверху/спереди/сбоку.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_mip(volume_values: np.ndarray, size_px: tuple[int, int] = (600, 220)) -> np.ndarray:
    w_px, h_px = size_px
    fig, axes = plt.subplots(1, 3, figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")

    views = {
        "top (xy)": volume_values.max(axis=0),
        "front (xz)": volume_values.max(axis=1),
        "side (yz)": volume_values.max(axis=2),
    }
    for ax, (title, proj) in zip(axes, views.items()):
        ax.set_facecolor("black")
        ax.imshow(proj, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title, color="white", fontsize=8)
        ax.axis("off")

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf


def render_source_power_panel(source_xyz: np.ndarray, source_power: np.ndarray, size_px: tuple[int, int] = (300, 512)) -> np.ndarray:
    w_px, h_px = size_px
    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    p = source_power - source_power.min()
    p = p / (p.max() + 1e-9)
    order = np.argsort(p)  # яркие поверх тусклых

    sc = ax.scatter(
        source_xyz[order, 0], source_xyz[order, 1],
        c=p[order], cmap="inferno", s=25, vmin=0, vmax=1,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("source power\n(top-down)", color="white", fontsize=8)
    fig.tight_layout(pad=0.3)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf
