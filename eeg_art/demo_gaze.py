"""
demo_gaze.py
---------------
Ветвь gaze_features поверх spectral_lenia: обычный спектральный Lenia-шаг
+ локальное возмущение в точке фиксации взгляда, амплитуда которого
зависит от z-оценки gamma (arousal-подобная метрика) в этот момент.

Запуск:
    python3 demo_gaze.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.calibration import Baseline
from eeg_art.core.viz_raw import render_raw_traces
from eeg_art.core.triptych import compose_triptych
from eeg_art.core.features import FeatureSet
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg
from eeg_art.ingestion.synthetic_gaze import generate_synthetic_gaze
from eeg_art.modules.spectral_lenia.features_spectral import SpectralFeatureExtractor
from eeg_art.modules.spectral_lenia.generator_lenia import LeniaGenerator
from eeg_art.modules.gaze_features.features_gaze import GazeFeatureExtractor
from eeg_art.modules.gaze_features.generator_gaze_condition import GazeCondition


def render_gaze_panel(gaze_xy, is_fixation, size_px=(260, 512)):
    w_px, h_px = size_px
    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    color = "orange" if is_fixation else "gray"
    ax.scatter([gaze_xy[0]], [gaze_xy[1]], s=300, c=color)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(f"gaze  fixation={is_fixation}", color="white", fontsize=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0
    grid_size = 64
    extractor = SpectralFeatureExtractor()

    print("Baseline (10 сек) ...")
    baseline_stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=10.0, seed=1)
    baseline_features = [
        extractor.extract(w, spatial)
        for w in sliding_windows(baseline_stream, channels, fs, win_sec=1.0, step_sec=0.25)
    ]
    baseline = Baseline.fit(baseline_features)

    print("Синтетический взгляд (айтрекер) ...")
    gaze_t, gaze_xy = generate_synthetic_gaze(duration_sec=15.0, fs=60.0, n_fixations=5, seed=7)
    gaze_extractor = GazeFeatureExtractor(gaze_t, gaze_xy)
    gaze_condition = GazeCondition(grid_size=grid_size, bump_radius=6, base_amplitude=0.6)

    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=15.0, seed=42)
    generator = LeniaGenerator(grid_size=grid_size, rng=np.random.default_rng(5))
    state = generator.init_state()
    log = SessionLog(output_dir="/home/claude/eeg_art_output_gaze")

    n_frames = 0
    for window in sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=0.15):
        raw_features = extractor.extract(window, spatial)
        z_fields = baseline.zscore(raw_features.fields)
        z_features = FeatureSet(timestamp=window.timestamp, fields=z_fields)
        state = generator.step(z_features, spatial, state)

        gaze_features = gaze_extractor.extract(window, spatial)
        gamma_z = float(np.mean(list(z_fields["band_power"]["gamma"].values())))
        state = gaze_condition.apply(
            state, gaze_features.fields["gaze_xy"], gaze_features.fields["is_fixation"],
            eeg_weight=1.0 / (1.0 + np.exp(-gamma_z)),
        )

        raw_img = render_raw_traces(window)
        feature_img = render_gaze_panel(gaze_features.fields["gaze_xy"], gaze_features.fields["is_fixation"])
        gen_img = np.stack([state.values] * 3, axis=-1)
        frame = compose_triptych(raw_img, feature_img, gen_img)

        log.append_frame(frame, timestamp=window.timestamp)
        log.append_features(window.timestamp, {
            "is_fixation": gaze_features.fields["is_fixation"],
            "gamma_z": gamma_z,
        })
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров сохранено в {log.output_dir}/frames/")


if __name__ == "__main__":
    main()
