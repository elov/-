"""
demo_creation_stage.py
---------------------------
Ветвь creation_stage: видео холста -> признаки -> стадия (эскиз/подмалёвок/
работа с цветом/детализация/пауза) -> двухосевой генератор (стадия +
band power ЭЭГ).

Запуск:
    python3 demo_creation_stage.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.calibration import Baseline
from eeg_art.core.viz_raw import render_raw_traces
from eeg_art.core.triptych import compose_triptych
from eeg_art.core.multistream_sync import nearest_by_timestamp
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg
from eeg_art.ingestion.synthetic_video import generate_synthetic_creation_video
from eeg_art.modules.spectral_lenia.features_spectral import SpectralFeatureExtractor
from eeg_art.modules.creation_stage.video_features import VideoFeatureTracker
from eeg_art.modules.creation_stage.stage_classifier import RuleBasedStageClassifier, STAGES
from eeg_art.modules.creation_stage.generator_dual_axis import DualAxisGenerator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_stage_panel(stage, features, size_px=(300, 512)):
    w_px, h_px = size_px
    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.text(0.05, 0.92, "СТАДИЯ:", color="gray", fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.83, stage, color="orange", fontsize=13, weight="bold", transform=ax.transAxes)
    labels = ["motion", "fill_fraction", "edge_density", "saturation"]
    for i, k in enumerate(labels):
        y = 0.65 - i * 0.12
        ax.text(0.05, y, f"{k}", color="white", fontsize=9, transform=ax.transAxes)
        ax.text(0.95, y, f"{features[k]:.3f}", color="cyan", fontsize=9, ha="right", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float) / 255.0
    plt.close(fig)
    return buf


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0
    extractor = SpectralFeatureExtractor()

    print("Baseline (10 сек) ...")
    baseline_stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=10.0, seed=1)
    baseline = Baseline.fit([
        extractor.extract(w, spatial)
        for w in sliding_windows(baseline_stream, channels, fs, win_sec=1.0, step_sec=0.25)
    ])

    print("Синтетическое видео холста (заведомо известная последовательность стадий, 30 сек) ...")
    art_t, art_frames, ground_truth = generate_synthetic_creation_video(30.0, fps=2.0, size=128, seed=5)
    print("Ground truth:")
    for t, label in ground_truth:
        print(f"  t={t:5.1f}с -> {label}")
    tracker = VideoFeatureTracker()
    classifier = RuleBasedStageClassifier()

    print("Основная сессия ЭЭГ (30 сек) ...")
    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=30.0, seed=42)
    generator = DualAxisGenerator(grid_size=64, eeg_modulation_strength=0.35, rng=np.random.default_rng(11))
    state = generator.init_state()
    log = SessionLog(output_dir="/home/claude/eeg_art_output_creation_stage")

    n_frames = 0
    stage_sequence = []
    last_frame_idx = None
    current_stage, current_vfeat = STAGES[0], {"motion": 0.0, "fill_fraction": 0.0, "edge_density": 0.0, "saturation": 0.0}
    for window in sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=0.15):
        raw = extractor.extract(window, spatial)
        z = baseline.zscore(raw.fields)

        frame, frame_idx, _ = nearest_by_timestamp(art_t, art_frames, window.timestamp)
        if frame_idx != last_frame_idx:
            current_vfeat = tracker.extract(frame)
            current_stage = classifier.classify(current_vfeat)
            last_frame_idx = frame_idx
        stage_sequence.append(STAGES.index(current_stage))

        state = generator.step(current_stage, z["band_power"], state)

        raw_img = render_raw_traces(window)
        stage_img = render_stage_panel(current_stage, current_vfeat)
        gen_img = np.stack([state.values] * 3, axis=-1)
        frame_out = compose_triptych(raw_img, stage_img, gen_img)

        log.append_frame(frame_out, timestamp=window.timestamp)
        log.append_features(window.timestamp, {"stage": current_stage, **current_vfeat})
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров в {log.output_dir}/frames/")

    print("\nОбнаруженная классификатором последовательность стадий (сравнить с ground truth выше):")
    prev = None
    for i, idx in enumerate(stage_sequence):
        s = STAGES[idx]
        if s != prev:
            t = i * 0.15
            print(f"  t={t:5.1f}с -> {s}")
            prev = s


if __name__ == "__main__":
    main()
