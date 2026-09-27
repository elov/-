"""
demo_multicam.py
--------------------
Мультикамера: ЭЭГ (spectral_lenia) + камера-на-картину (dynamic palette,
палитра эволюционирует по мере "рисования") + камера пространства
(чисто документационный архивный поток, не участвует в генерации).

Всё синхронизировано по общему таймкоду (core.multistream_sync) —
программная имитация роли LSL.

Запуск:
    python3 demo_multicam.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.calibration import Baseline
from eeg_art.core.viz_raw import render_raw_traces
from eeg_art.core.triptych import compose_triptych
from eeg_art.core.features import FeatureSet
from eeg_art.core.multistream_sync import nearest_by_timestamp
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg
from eeg_art.ingestion.synthetic_video import generate_synthetic_video_artwork, generate_synthetic_video_space
from eeg_art.modules.spectral_lenia.features_spectral import SpectralFeatureExtractor
from eeg_art.modules.spectral_lenia.generator_lenia import LeniaGenerator
from eeg_art.modules.palette_extraction.dynamic_source import PaletteDynamicSource
from eeg_art.modules.palette_extraction.colorize import build_colormap, colorize
from eeg_art.modules.palette_extraction.render_palette import render_palette


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0
    duration = 15.0
    extractor = SpectralFeatureExtractor()

    print("Baseline (10 сек) ...")
    baseline_stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=10.0, seed=1)
    baseline_features = [
        extractor.extract(w, spatial)
        for w in sliding_windows(baseline_stream, channels, fs, win_sec=1.0, step_sec=0.25)
    ]
    baseline = Baseline.fit(baseline_features)

    print("Синтетические видеопотоки (камера на картину + камера пространства) ...")
    art_t, art_frames = generate_synthetic_video_artwork(duration, fps=2.0, size=128, seed=3)
    space_t, space_frames = generate_synthetic_video_space(duration, fps=5.0, size=96, seed=4)
    palette_source = PaletteDynamicSource(art_t, art_frames, n_main=4, n_accent=2, recompute_every_n=1)

    print("Основная сессия ЭЭГ (15 сек) ...")
    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=duration, seed=42)
    generator = LeniaGenerator(grid_size=64, rng=np.random.default_rng(1))
    state = generator.init_state()
    log = SessionLog(output_dir="/home/claude/eeg_art_output_multicam")

    n_frames = 0
    for window in sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=0.15):
        raw_features = extractor.extract(window, spatial)
        z_fields = baseline.zscore(raw_features.fields)
        z_features = FeatureSet(timestamp=window.timestamp, fields=z_fields)
        state = generator.step(z_features, spatial, state)

        palette, art_frame_used, art_dt = palette_source.get_palette(window.timestamp)
        lut = build_colormap(palette)
        colored = colorize(state.values, lut)

        space_frame, space_idx, space_dt = nearest_by_timestamp(space_t, space_frames, window.timestamp)

        raw_img = render_raw_traces(window)
        palette_img = render_palette(palette)
        frame = compose_triptych(raw_img, palette_img, colored)

        log.append_frame(frame, timestamp=window.timestamp)
        log.append_stream("artwork_camera", art_frame_used, window.timestamp, n_frames)
        log.append_stream("room_camera", space_frame, window.timestamp, n_frames)
        log.append_features(window.timestamp, {
            "artwork_camera_dt": art_dt, "room_camera_dt": space_dt,
            "n_main": len(palette.main), "n_accent": len(palette.accent),
        })
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров-триптихов + архив 2 вспомогательных потоков в {log.output_dir}")


if __name__ == "__main__":
    main()
