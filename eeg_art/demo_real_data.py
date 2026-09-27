"""
demo_real_data.py
---------------------
Первый прогон на РЕАЛЬНЫХ (не синтетических) данных пользователя:
PrimerPosledovatelnostEEG_8-13Гц_4-7Гц.txt, fs=256 Гц, 19 каналов.

Предварительная проверка спектра (см. чат) показала: сигнал узкополосный,
~7.5-14.5 Гц, а НЕ широкополосный и НЕ в заявленных 4-7 отдельно —
значит delta/theta/beta/gamma-экстракторы будут честно мерить шумовой
пол, а не реальный ритм. Ветвь spectral_lenia (5 диапазонов) поэтому
здесь не лучший выбор — используем particle_bloom, но с равным успехом
подошла бы любая ветвь; репрезентативность результата в первую очередь
зависит от входных данных, а не от генератора.

Baseline берётся из первых 20 сек ЭТОГО ЖЕ файла (в реальном протоколе
музея baseline должен быть отдельной записью — здесь это вынужденное
упрощение, т.к. другого файла нет).

Запуск:
    python3 demo_real_data.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.calibration import Baseline
from eeg_art.core.features import FeatureSet
from eeg_art.ingestion.file_eeg import load_ascii_eeg
from eeg_art.modules.spectral_lenia.features_spectral import SpectralFeatureExtractor
from eeg_art.modules.particle_bloom.generator_particles import ParticleBloomGenerator

PATH = "/mnt/user-data/uploads/PrimerPosledovatelnostEEG_8-13Гц_4-7Гц.txt"


def main():
    spatial = SpatialModel()
    fs = 256.0
    extractor = SpectralFeatureExtractor()

    print("Загрузка реального файла ...")
    data, channels = load_ascii_eeg(PATH)  # допущение: столбцы в порядке SpatialModel().channels
    duration = data.shape[1] / fs
    print(f"  {data.shape[0]} каналов x {data.shape[1]} отсчётов, {duration:.1f} сек при fs={fs}")

    baseline_end = int(20 * fs)
    baseline_stream = data[:, :baseline_end]
    main_stream = data[:, baseline_end:baseline_end + int(30 * fs)]

    print("Baseline (первые 20 сек файла) ...")
    baseline = Baseline.fit([
        extractor.extract(w, spatial)
        for w in sliding_windows(baseline_stream, channels, fs, win_sec=1.0, step_sec=0.25)
    ])

    print("Основная сессия (следующие 30 сек файла) ...")
    generator = ParticleBloomGenerator(
        spatial, grid_size=220, decay=0.975, channel_spawn_rate=50, background_spawn_rate=180,
        rng=np.random.default_rng(3),
    )
    canvas = generator.init_state()
    log = SessionLog(output_dir="/home/claude/eeg_art_output_realdata")

    n_frames = 0
    band_means = {b: [] for b in ["delta", "theta", "alpha", "beta", "gamma"]}
    for window in sliding_windows(main_stream, channels, fs, win_sec=1.0, step_sec=0.15):
        raw = extractor.extract(window, spatial)
        z = baseline.zscore(raw.fields)
        for b in band_means:
            band_means[b].append(np.mean(list(z["band_power"][b].values())))
        canvas = generator.step(FeatureSet(timestamp=window.timestamp, fields=z["band_power"] and z), spatial, canvas)

        log.append_frame(canvas, timestamp=window.timestamp)
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров в {log.output_dir}/frames/")
    print("\nСредняя |z| по диапазонам за сессию (ожидаемо: alpha сильно выделяется, остальные ~шум):")
    for b, vals in band_means.items():
        print(f"  {b:6s}: mean(z)={np.mean(vals):+.2f}  std(z)={np.std(vals):.2f}")


if __name__ == "__main__":
    main()
