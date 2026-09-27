"""
demo_source.py
------------------
Ветвь source_volumetric.

Шаг 1 — валидация forward/inverse модели: искусственно усиливаем
потенциал на затылочных каналах (O1, O2) и проверяем, что sLORETA
находит максимум мощности у источников, ближайших именно к этим
сенсорам (иначе вся модель не заслуживает доверия).

Шаг 2 — прогон генератора: 3D-автомат, чья карта роста определяется
восстановленной мощностью источников в объёме коры.

Запуск:
    python3 demo_source.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.viz_raw import render_raw_traces
from eeg_art.core.triptych import compose_triptych
from eeg_art.core.source_model import (
    fibonacci_hemisphere, sensor_positions_3d, build_lead_field, sloreta_source_power,
)
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg
from eeg_art.modules.source_volumetric.features_source import ScalpActivityFeatureExtractor
from eeg_art.modules.source_volumetric.generator_volume import SourceVolumeGenerator
from eeg_art.modules.source_volumetric.render_volume import render_mip, render_source_power_panel


def validate_forward_inverse(spatial):
    print("Валидация: искусственно усиливаем O1/O2, проверяем, что источник локализуется рядом с ними.")
    source_xyz = fibonacci_hemisphere(300, radius=0.8)
    sensor_xyz = sensor_positions_3d(spatial.channel_positions, radius=1.0)
    L = build_lead_field(sensor_xyz, source_xyz)

    v = np.ones(len(spatial.channels)) * 0.1
    idx_o1 = spatial.channels.index("O1")
    idx_o2 = spatial.channels.index("O2")
    v[idx_o1] *= 6
    v[idx_o2] *= 6

    power = sloreta_source_power(L, v)
    best_source = source_xyz[np.argmax(power)]

    o1_pos = sensor_xyz[idx_o1]
    fp1_pos = sensor_xyz[spatial.channels.index("Fp1")]
    dist_to_o1 = np.linalg.norm(best_source - o1_pos)
    dist_to_fp1 = np.linalg.norm(best_source - fp1_pos)

    print(f"  Источник с максимальной мощностью: {best_source.round(2)}")
    print(f"  Расстояние до O1: {dist_to_o1:.2f}, до Fp1 (лобный, контроль): {dist_to_fp1:.2f}")
    print(f"  {'ОК: ближе к O1' if dist_to_o1 < dist_to_fp1 else 'ПРОБЛЕМА: не локализовалось у O1!'}\n")


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0

    validate_forward_inverse(spatial)

    print("Основная сессия (12 сек, grid 20^3 — воксельная свёртка дороже 2D) ...")
    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=12.0, seed=42)

    extractor = ScalpActivityFeatureExtractor()
    generator = SourceVolumeGenerator(spatial, grid_size=20, n_sources=300, rng=np.random.default_rng(9))
    state = generator.init_state()
    log = SessionLog(output_dir="/home/claude/eeg_art_output_source")

    n_frames = 0
    for window in sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=0.2):
        features = extractor.extract(window, spatial)
        state = generator.step(features, spatial, state)

        raw_img = render_raw_traces(window)
        feature_img = render_source_power_panel(generator.source_xyz, generator.last_source_power)
        gen_img = render_mip(state.values)
        frame = compose_triptych(raw_img, feature_img, gen_img)

        log.append_frame(frame, timestamp=window.timestamp)
        log.append_features(window.timestamp, {"max_source_power": float(generator.last_source_power.max())})
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров сохранено в {log.output_dir}/frames/")


if __name__ == "__main__":
    main()
