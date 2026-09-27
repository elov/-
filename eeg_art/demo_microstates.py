"""
demo_microstates.py
-----------------------
Ветвь microstates_regime: baseline -> фитинг k=4 шаблонов микросостояний ->
основная сессия классифицируется по ним -> дискретная смена режима Lenia.

Запуск:
    python3 demo_microstates.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.viz_raw import render_raw_traces
from eeg_art.core.triptych import compose_triptych
from eeg_art.core.microstate_fit import gfp_peak_topographies, fit_microstates
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg
from eeg_art.modules.microstates_regime.features_microstates import MicrostateFeatureExtractor
from eeg_art.modules.microstates_regime.generator_regime import MicrostateRegimeGenerator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_microstate_panel(label: str, corr: np.ndarray, labels: list[str], size_px=(260, 512)) -> np.ndarray:
    w_px, h_px = size_px
    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    colors = ["white" if l != label else "orange" for l in labels]
    ax.bar(labels, np.abs(corr), color=colors)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_title(f"microstate: {label}", color="white", fontsize=9)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0

    print("Запись baseline (20 сек) и фитинг шаблонов микросостояний (k=4) ...")
    baseline_stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=20.0, seed=1)
    topo = gfp_peak_topographies(baseline_stream)
    templates = fit_microstates(topo, k=4, seed=0)
    print(f"Шаблоны получены из {topo.shape[0]} пиков GFP.")

    extractor = MicrostateFeatureExtractor(templates)
    generator = MicrostateRegimeGenerator(grid_size=64, rng=np.random.default_rng(3))
    state = generator.init_state()

    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=15.0, seed=42)
    log = SessionLog(output_dir="/home/claude/eeg_art_output_microstates")

    n_frames = 0
    label_sequence = []
    for window in sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=0.15):
        features = extractor.extract(window, spatial)
        state = generator.step(features, spatial, state)
        label_sequence.append(features.fields["microstate_label"])

        raw_img = render_raw_traces(window)
        feature_img = render_microstate_panel(
            features.fields["microstate_label"], features.fields["microstate_corr"], extractor.labels
        )
        gen_img = np.stack([state.values] * 3, axis=-1)
        frame = compose_triptych(raw_img, feature_img, gen_img)

        log.append_frame(frame, timestamp=window.timestamp)
        log.append_features(window.timestamp, {"microstate_label": features.fields["microstate_label"]})
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров. Последовательность микросостояний: {''.join(label_sequence)}")


if __name__ == "__main__":
    main()
