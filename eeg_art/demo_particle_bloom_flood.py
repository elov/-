"""
demo_particle_bloom_flood.py
---------------------------------
Тот же particle_bloom, но пресет "flood" — выше спавн, медленнее
затухание, холст быстрее насыщается до плотной заливки (ближе к
референсному ролику).

Запуск:
    python3 demo_particle_bloom_flood.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.calibration import Baseline
from eeg_art.core.features import FeatureSet
from eeg_art.core.config import save_preset, instantiate_from_preset
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg
from eeg_art.modules.spectral_lenia.features_spectral import SpectralFeatureExtractor
import eeg_art.modules.particle_bloom.generator_particles  # noqa: F401 регистрирует метод


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0
    win_sec, step_sec = 1.0, 0.15
    extractor = SpectralFeatureExtractor()

    preset_path = "/home/claude/preset_particle_flood.json"
    save_preset(
        preset_path, "particle_bloom.generator", "flow_attractor",
        dict(
            grid_size=220,
            decay=0.985,                # медленнее затухание (было 0.965)
            channel_spawn_rate=70,      # вдвое выше (было 35)
            background_spawn_rate=250,   # выше (было 120)
            particle_speed=2.5,
            lifespan_channel=25,
            lifespan_background=10,
            channel_alpha=0.35,
            background_alpha=0.10,
        ),
        meta={"name": "flood_v1", "note": "ближе к референсному ролику — плотная заливка к концу сессии"},
    )
    print(f"Пресет сохранён: {preset_path}")

    print("Baseline (10 сек) ...")
    baseline_stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=10.0, seed=1)
    baseline = Baseline.fit([
        extractor.extract(w, spatial)
        for w in sliding_windows(baseline_stream, channels, fs, win_sec=1.0, step_sec=0.25)
    ])

    print("Основная сессия (30 сек) ...")
    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=30.0, seed=42)

    generator = instantiate_from_preset(preset_path, spatial=spatial, rng=np.random.default_rng(7))
    canvas = generator.init_state()
    log = SessionLog(output_dir="/home/claude/eeg_art_output_particles_flood")

    n_frames = 0
    for window in sliding_windows(stream, channels, fs, win_sec=win_sec, step_sec=step_sec):
        raw = extractor.extract(window, spatial)
        z = baseline.zscore(raw.fields)
        canvas = generator.step(FeatureSet(timestamp=window.timestamp, fields=z), spatial, canvas)

        log.append_frame(canvas, timestamp=window.timestamp)
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров, живых частиц: {len(generator.pos)}")


if __name__ == "__main__":
    main()
