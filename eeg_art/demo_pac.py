"""
demo_pac.py
--------------
Ветвь pac_coupling.

Шаг 1 — валидация метода: считаем MI на сигнале с заведомо встроенной
theta->gamma связью для части каналов и без неё для остальных, проверяем,
что channel_pac действительно различает их (иначе дальше нет смысла).

Шаг 2 — прогон генератора: PACGenerator, где связь фазы и амплитуды
управляет одним параметром (growth_center), а не является отдельной
независимой ручкой (см. docstring generator_pac.py).

Запуск:
    python3 demo_pac.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.viz_raw import render_raw_traces
from eeg_art.core.triptych import compose_triptych
from eeg_art.core.pac import channel_pac
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg_with_pac
from eeg_art.modules.pac_coupling.features_pac import PACFeatureExtractor
from eeg_art.modules.pac_coupling.generator_pac import PACGenerator
from eeg_art.modules.pac_coupling.render_panel import render_pac_panel


def validate_method(channels, fs):
    coupled = ["O1", "O2", "P3", "P4"]
    print(f"Валидация: встроенная PAC-связь в каналах {coupled}, остальные — без связи.")
    data = generate_synthetic_eeg_with_pac(channels, fs=fs, duration_sec=10.0, coupled_channels=coupled, seed=11)
    mi, _, _ = channel_pac(data, fs)
    for ch, v in sorted(zip(channels, mi), key=lambda x: -x[1]):
        marker = " <-- встроена связь" if ch in coupled else ""
        print(f"  {ch:4s}  MI={v:.3f}{marker}")

    mean_coupled = np.mean([v for ch, v in zip(channels, mi) if ch in coupled])
    mean_rest = np.mean([v for ch, v in zip(channels, mi) if ch not in coupled])
    print(f"Среднее MI: связанные={mean_coupled:.3f}, остальные={mean_rest:.3f}  "
          f"({'метод РАЗЛИЧАЕТ' if mean_coupled > mean_rest * 1.5 else 'слабое различие!'})\n")
    return coupled


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0

    coupled = validate_method(channels, fs)

    print("Основная сессия (15 сек) с той же встроенной связью ...")
    stream = generate_synthetic_eeg_with_pac(channels, fs=fs, duration_sec=15.0, coupled_channels=coupled, seed=42)

    extractor = PACFeatureExtractor(phase_band=(4.0, 8.0), amp_band=(30.0, 45.0))
    generator = PACGenerator(grid_size=64, rng=np.random.default_rng(6))
    state = generator.init_state()
    log = SessionLog(output_dir="/home/claude/eeg_art_output_pac")

    n_frames = 0
    for window in sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=0.15):
        features = extractor.extract(window, spatial)
        state = generator.step(features, spatial, state)

        raw_img = render_raw_traces(window)
        feature_img = render_pac_panel(features.fields["pac_mi"])
        gen_img = np.stack([state.values] * 3, axis=-1)
        frame = compose_triptych(raw_img, feature_img, gen_img)

        log.append_frame(frame, timestamp=window.timestamp)
        log.append_features(window.timestamp, {"pac_mi": features.fields["pac_mi"]})
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров-триптихов сохранено в {log.output_dir}/frames/")


if __name__ == "__main__":
    main()
