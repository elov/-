"""
modules.gaze_features.generator_gaze_condition
----------------------------------------------------
"Отдельное условие генерации" по взгляду — обёртка (decorator) над любым
Grid-генератором: после обычного шага, если в этот момент зафиксирована
точка фиксации взгляда, в соответствующей точке сетки добавляется
локальное возмущение поля.

Это прямой аналог fixation-related potentials: ЭЭГ-реакция (через
eeg_weight — например, z-оценку gamma в этот момент) привязывается не
только ко времени, но и к месту на картине, куда смотрел человек.
Мэппинг ("партитура"): амплитуда и радиус возмущения не зависят от
силы фиксации — только амплитуда масштабируется eeg_weight, радиус
фиксирован (упрощение первой версии).
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.field_abstraction import Grid
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry

PARAM_SCHEMA = {
    "bump_radius": ParamSpec(5, 1, 32, "px", "Радиус локального возмущения в точке фиксации взгляда."),
    "base_amplitude": ParamSpec(0.5, 0.0, 2.0, "", "Амплитуда возмущения при eeg_weight=1.0."),
}


@MethodRegistry.register("gaze_features.condition", "local_bump", schema=PARAM_SCHEMA)
class GazeCondition:
    def __init__(self, grid_size: int, bump_radius: int = 5, base_amplitude: float = 0.5):
        self.grid_size = grid_size
        self.bump_radius = bump_radius
        self.base_amplitude = base_amplitude

        r = bump_radius
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        self._bump_template = np.exp(-(xx ** 2 + yy ** 2) / (2 * (r / 2) ** 2))

    def _xy_to_index(self, xy: np.ndarray) -> tuple[int, int]:
        # xy в [-1, 1] -> индекс сетки [0, grid_size)
        x, y = xy
        col = int((x + 1) / 2 * (self.grid_size - 1))
        row = int((1 - (y + 1) / 2) * (self.grid_size - 1))  # y инвертирован: верх холста = верх изображения
        return row, col

    def apply(self, grid: Grid, gaze_xy: np.ndarray, is_fixation: bool, eeg_weight: float = 1.0) -> Grid:
        if not is_fixation:
            return grid

        row, col = self._xy_to_index(gaze_xy)
        r = self.bump_radius
        values = grid.values.copy()

        r0, r1 = max(0, row - r), min(self.grid_size, row + r + 1)
        c0, c1 = max(0, col - r), min(self.grid_size, col + r + 1)
        tr0, tc0 = r0 - (row - r), c0 - (col - r)
        tr1, tc1 = tr0 + (r1 - r0), tc0 + (c1 - c0)

        amplitude = self.base_amplitude * max(0.0, eeg_weight)
        values[r0:r1, c0:c1] = np.clip(
            values[r0:r1, c0:c1] + amplitude * self._bump_template[tr0:tr1, tc0:tc1], 0.0, 1.0
        )
        return Grid(values=values)
