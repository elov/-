"""
ingestion.file_eeg
---------------------
Загрузка реальных (не синтетических) записей ЭЭГ из простого ASCII-формата:
пробел-разделённые столбцы, каждая строка — один момент времени, каждый
столбец — канал. Без заголовка — соответствие столбцов конкретным
электродам 10/20 НЕ поставляется файлом и должно быть либо известно
заранее (протокол записи), либо явно указано вызывающим кодом; по
умолчанию предполагается порядок core.spatial_model.SpatialModel().channels
(19 столбцов — ровно под эту систему), но это ДОПУЩЕНИЕ, а не факт,
установленный из данных, и должно быть прямо помечено как таковое при
использовании.
"""

from __future__ import annotations

import numpy as np


def load_ascii_eeg(path: str, channels: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Возвращает (data, channels): data shape (n_channels, n_samples) — тот
    же формат, что и ingestion.synthetic_eeg.generate_synthetic_eeg, так
    что дальше по пайплайну (core.windowing.sliding_windows и т.д.)
    ничего не меняется независимо от источника данных.
    """
    with open(path, encoding="utf-8") as f:
        rows = [line.split() for line in f if line.strip()]
    data = np.array(rows, dtype=float)  # (n_samples, n_columns)

    n_columns = data.shape[1]
    if channels is None:
        from eeg_art.core.spatial_model import SpatialModel
        default_channels = SpatialModel().channels
        if n_columns != len(default_channels):
            raise ValueError(
                f"В файле {n_columns} столбцов, а не {len(default_channels)} — "
                f"укажите channels явно, соответствие столбцов электродам неизвестно."
            )
        channels = default_channels

    return data.T, channels
