"""
demo_connectivity.py
------------------------
Ветвь connectivity_graph с триптих-рендером (сырые каналы / PLV-партитура /
граф-автомат). PLV уже нормирован в [0,1] по построению, доп. калибровка
baseline здесь не требуется (в отличие от band power).

Запуск:
    python3 demo_connectivity.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.session_log import SessionLog
from eeg_art.core.viz_raw import render_raw_traces
from eeg_art.core.triptych import compose_triptych
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg
from eeg_art.modules.connectivity_graph.features_connectivity import ConnectivityFeatureExtractor
from eeg_art.modules.connectivity_graph.generator_graph import GraphGenerator
from eeg_art.modules.connectivity_graph.render import render_graph
from eeg_art.modules.connectivity_graph.render_panel import render_plv_panel


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0

    print(f"Генерация синтетического ЭЭГ: {len(channels)} каналов, fs={fs} Гц ...")
    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=15.0, seed=42)

    extractor = ConnectivityFeatureExtractor(band_low=8.0, band_high=13.0)
    generator = GraphGenerator(spatial, rng=np.random.default_rng(2))
    state = generator.init_state()

    log = SessionLog(output_dir="/home/claude/eeg_art_output_graph")

    n_frames = 0
    for window in sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=0.15):
        features = extractor.extract(window, spatial)
        state = generator.step(features, spatial, state)

        raw_img = render_raw_traces(window)
        feature_img = render_plv_panel(features.fields["plv_matrix"], channels)
        gen_img = render_graph(state, spatial)
        frame = compose_triptych(raw_img, feature_img, gen_img)

        log.append_frame(frame, timestamp=window.timestamp)
        log.append_features(window.timestamp, {"plv_mean": float(features.fields["plv_matrix"].mean())})
        n_frames += 1

    log.flush()
    print(f"Готово: {n_frames} кадров-триптихов сохранено в {log.output_dir}/frames/")


if __name__ == "__main__":
    main()
