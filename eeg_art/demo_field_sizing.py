"""
demo_field_sizing.py
------------------------
Поле КА, соразмерное пропорциям конкретного произведения (core.field_sizing),
на примере particle_bloom: панорамный холст 100x50 см (2:1) вместо
дефолтного квадрата.

Запуск:
    python3 demo_field_sizing.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.calibration import Baseline
from eeg_art.core.features import FeatureSet
from eeg_art.core.field_sizing import field_shape_for_dimensions
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg
from eeg_art.modules.spectral_lenia.features_spectral import SpectralFeatureExtractor
from eeg_art.modules.particle_bloom.generator_particles import ParticleBloomGenerator


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0
    win_sec, step_sec = 1.0, 0.15
    extractor = SpectralFeatureExtractor()

    # холст 100x50 см -> пропорции 2:1, бюджет ~220*220 ячеек, как в прошлом прогоне
    field_shape = field_shape_for_dimensions(100, 50, target_cells=220 * 220)
    print(f"Холст 100x50 см -> поле {field_shape} (H, W), бюджет клеток ~{220*220}")

    print("Baseline (10 сек) ...")
    baseline_stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=10.0, seed=1)
    baseline = Baseline.fit([
        extractor.extract(w, spatial)
        for w in sliding_windows(baseline_stream, channels, fs, win_sec=1.0, step_sec=0.25)
    ])

    print("Основная сессия (30 сек) ...")
    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=30.0, seed=42)

    generator = ParticleBloomGenerator(
        spatial, field_shape=field_shape,
        decay=0.985, channel_spawn_rate=70, background_spawn_rate=250,
        rng=np.random.default_rng(7),
    )
    canvas = generator.init_state()
    print(f"Реальная форма холста генератора: {canvas.shape}")
    log = SessionLog(output_dir="/home/claude/eeg_art_output_panorama")

    n_frames = 0
    for window in sliding_windows(stream, channels, fs, win_sec=win_sec, step_sec=step_sec):
        raw = extractor.extract(window, spatial)
        z = baseline.zscore(raw.fields)
        canvas = generator.step(FeatureSet(timestamp=window.timestamp, fields=z), spatial, canvas)

        log.append_frame(canvas, timestamp=window.timestamp)
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров, форма кадра: {canvas.shape}")


if __name__ == "__main__":
    main()
