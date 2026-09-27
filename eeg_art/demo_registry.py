"""
demo_registry.py
--------------------
Проверка реестра методов и пресетов:
    1. список доступных методов по ролям
    2. альтернативный метод внутри той же роли (fft vs welch) — оба дают
       осмысленно похожий (не идентичный) результат
    3. пресет с настроенными параметрами -> сохранение -> загрузка ->
       воспроизводимость (одинаковый seed должен дать БИТ-В-БИТ тот же
       результат — иначе пресетам нельзя доверять)

Запуск:
    python3 demo_registry.py
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.registry import MethodRegistry
from eeg_art.core.config import save_preset, load_preset, instantiate_from_preset
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.windowing import sliding_windows
from eeg_art.core.calibration import Baseline
from eeg_art.core.features import FeatureSet
from eeg_art.ingestion.synthetic_eeg import generate_synthetic_eeg

import eeg_art.modules.spectral_lenia.features_spectral  # noqa: F401 регистрирует "fft"
import eeg_art.modules.spectral_lenia.features_spectral_welch  # noqa: F401 регистрирует "welch"
import eeg_art.modules.spectral_lenia.generator_lenia as lenia_mod
import eeg_art.modules.pac_coupling.generator_pac  # noqa: F401 регистрирует pac_coupling.generator
import eeg_art.modules.connectivity_graph.features_connectivity  # noqa: F401
import eeg_art.modules.connectivity_graph.generator_graph  # noqa: F401
import eeg_art.modules.microstates_regime.features_microstates  # noqa: F401
import eeg_art.modules.microstates_regime.generator_regime  # noqa: F401
import eeg_art.modules.gaze_features.features_gaze  # noqa: F401
import eeg_art.modules.gaze_features.generator_gaze_condition  # noqa: F401
import eeg_art.modules.palette_extraction.extract  # noqa: F401
import eeg_art.modules.source_volumetric.generator_volume  # noqa: F401
import eeg_art.modules.sonification.audification  # noqa: F401
import eeg_art.modules.sonification.parameter_mapping  # noqa: F401


def show_registry():
    print("Категории и методы в реестре:")
    for category in MethodRegistry.list_categories():
        methods = MethodRegistry.list_methods(category)
        print(f"  {category}: {methods}")
    print()


def compare_fft_vs_welch(spatial, channels, fs):
    print("Сравнение методов feature_extractor: fft vs welch (та же роль, разные реализации)")
    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=2.0, seed=5)
    window = next(iter(sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=1.0)))

    fft_extractor = MethodRegistry.create("spectral_lenia.feature_extractor", "fft")
    welch_extractor = MethodRegistry.create("spectral_lenia.feature_extractor", "welch", n_segments=4)

    fft_result = fft_extractor.extract(window, spatial)
    welch_result = welch_extractor.extract(window, spatial)

    for band in ["alpha", "gamma"]:
        fft_val = np.mean(list(fft_result.fields["band_power"][band].values()))
        welch_val = np.mean(list(welch_result.fields["band_power"][band].values()))
        print(f"  {band}: fft={fft_val:.2f}  welch={welch_val:.2f}")
    print()


def preset_roundtrip(spatial, channels, fs):
    print("Пресет: настроенный LeniaGenerator -> сохранение -> загрузка -> воспроизводимость")

    custom_params = dict(
        grid_size=48,
        kernel_radius_min=6, kernel_radius_max=10,   # уже, чем дефолт — компактные структуры
        growth_center_min=0.12, growth_center_max=0.18,
        growth_width_min=0.03, growth_width_max=0.05,
        noise_amp_min=0.0, noise_amp_max=0.01,        # тише дефолта — меньше шансов на насыщение
        dt_min=0.05, dt_max=0.12,
    )
    preset_path = "/home/claude/preset_calm_gallery.json"
    save_preset(
        preset_path, "spectral_lenia.generator", "default", custom_params,
        meta={"name": "calm_gallery_v1", "note": "тише и компактнее дефолта, для тихого зала"},
    )
    print(f"  Сохранён пресет: {preset_path}")
    print(f"  Содержимое: {load_preset(preset_path)['meta']}")

    extractor = MethodRegistry.create("spectral_lenia.feature_extractor", "fft")
    stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=3.0, seed=42)
    baseline_stream = generate_synthetic_eeg(channels, fs=fs, duration_sec=5.0, seed=1)
    baseline = Baseline.fit([
        extractor.extract(w, spatial)
        for w in sliding_windows(baseline_stream, channels, fs, win_sec=1.0, step_sec=0.5)
    ])

    def run_from_preset(seed: int) -> np.ndarray:
        gen = instantiate_from_preset(preset_path, rng=np.random.default_rng(seed))
        state = gen.init_state()
        for window in sliding_windows(stream, channels, fs, win_sec=1.0, step_sec=0.3):
            raw = extractor.extract(window, spatial)
            z = baseline.zscore(raw.fields)
            state = gen.step(FeatureSet(timestamp=window.timestamp, fields=z), spatial, state)
        return state.values

    result_a = run_from_preset(seed=100)
    result_b = run_from_preset(seed=100)
    identical = np.array_equal(result_a, result_b)
    print(f"  Два независимых запуска из одного пресета с одним seed идентичны: {identical}")

    # и проверка, что параметры пресета реально применились (а не тихо проигнорированы)
    gen_from_preset = instantiate_from_preset(preset_path, rng=np.random.default_rng(0))
    print(f"  grid_size из пресета применился: {gen_from_preset.grid_size == 48}")
    print(f"  dt_max из пресета применился: {gen_from_preset.dt_max == 0.12}")


def validation_rejects_bad_value():
    print("\nВалидация схемы: попытка сохранить параметр вне научно обоснованного диапазона")
    try:
        save_preset(
            "/home/claude/preset_bad.json", "spectral_lenia.generator", "default",
            {"dt_max": 5.0},  # вне диапазона [0.001, 1.0] по PARAM_SCHEMA
        )
        print("  ОШИБКА: должно было отклонить, но не отклонило!")
    except ValueError as e:
        print(f"  Отклонено корректно: {e}")


def main():
    spatial = SpatialModel()
    channels = spatial.channels
    fs = 250.0

    show_registry()
    compare_fft_vs_welch(spatial, channels, fs)
    preset_roundtrip(spatial, channels, fs)
    validation_rejects_bad_value()


if __name__ == "__main__":
    main()
