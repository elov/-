"""
core.calibration
-------------------
Baseline-калибровка: без неё пороги/диапазоны в генераторах (growth_center,
kernel_radius и т.п.) произвольны и не воспроизводимы между людьми/сеансами.

Идея: перед основной сессией пишется короткая baseline-запись (глаза
закрыты/открыты, ~1-2 мин). На ней считается среднее и std каждого
"листового" признака (band_power по каждому каналу и диапазону, plv по
каждой паре и т.д. — работает с любой вложенностью словарей). Основная
сессия нормируется этим baseline в z-оценки, дальше маппинг признак->
параметр применяется к z-оценке, а не к сырому числу.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from eeg_art.core.features import FeatureSet


def _flatten(d, prefix=()):
    """dict (произвольно вложенный) -> {(k1,k2,...): scalar}"""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, prefix + (k,)))
    elif isinstance(d, (int, float, np.floating, np.integer)):
        out[prefix] = float(d)
    # массивы (напр. plv_matrix) и списки (channels) baseline не нормирует —
    # калибруются только скалярные листья
    return out


@dataclass
class Baseline:
    means: dict
    stds: dict

    @classmethod
    def fit(cls, feature_sets: list[FeatureSet]) -> "Baseline":
        acc: dict[tuple, list[float]] = {}
        for fs in feature_sets:
            flat = _flatten(fs.fields)
            for key, val in flat.items():
                acc.setdefault(key, []).append(val)

        means = {k: float(np.mean(v)) for k, v in acc.items()}
        stds = {k: float(np.std(v)) + 1e-9 for k, v in acc.items()}
        return cls(means=means, stds=stds)

    def zscore(self, fields: dict) -> dict:
        """Возвращает копию словаря той же формы, но со скалярными листьями,
        заменёнными на z-оценку относительно baseline (неизвестные ключи —
        не встречавшиеся при fit — возвращаются как есть, без нормировки)."""
        def _walk(d, prefix):
            if isinstance(d, dict):
                return {k: _walk(v, prefix + (k,)) for k, v in d.items()}
            if isinstance(d, (int, float, np.floating, np.integer)):
                if prefix in self.means:
                    return (float(d) - self.means[prefix]) / self.stds[prefix]
                return float(d)
            return d
        return _walk(fields, ())
