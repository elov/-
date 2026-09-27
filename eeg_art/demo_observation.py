"""
demo_observation.py
----------------------
Ветвь spectral_lenia с baseline-калибровкой и триптих-рендером
(сырые каналы / band power-партитура / Lenia).

Запуск:
    python3 demo_observation.py
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
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg
from eeg_art.modules.spectral_lenia.features_spectral import SpectralFeatureExtractor
from eeg_art.modules.spectral_lenia.generator_lenia import LeniaGenerator
from eeg_art.modules.spectral_lenia.render_panel import render_band_power_panel


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0
    extractor = SpectralFeatureExtractor()

    print("Запись baseline (10 сек) ...")
    baseline_stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=10.0, seed=1)
    baseline_features = [
        extractor.extract(w, spatial)
        for w in sliding_windows(baseline_stream, channels, fs, win_sec=1.0, step_sec=0.25)
    ]
    baseline = Baseline.fit(baseline_features)
    print(f"Baseline посчитан по {len(baseline_features)} окнам.")

    print("Генерация основной сессии (15 сек) ...")
    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=15.0, seed=42)

    generator = LeniaGenerator(grid_size=64, rng=np.random.default_rng(1))
    state = generator.init_state()
    log = SessionLog(output_dir="/home/claude/eeg_art_output")

    n_frames = 0
    for window in sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=0.15):
        raw_features = extractor.extract(window, spatial)
        z_fields = baseline.zscore(raw_features.fields)
        z_features = FeatureSet(timestamp=window.timestamp, fields=z_fields)

        state = generator.step(z_features, spatial, state)

        raw_img = render_raw_traces(window)
        feature_img = render_band_power_panel(z_fields["band_power"])
        gen_img = np.stack([state.values] * 3, axis=-1)
        frame = compose_triptych(raw_img, feature_img, gen_img)

        log.append_frame(frame, timestamp=window.timestamp)
        log.append_features(window.timestamp, z_fields)
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров-триптихов сохранено в {log.output_dir}/frames/")


if __name__ == "__main__":
    main()
