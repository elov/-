"""
modules.microstates_regime.features_microstates
----------------------------------------------------
FeatureExtractor ветви "микросостояния -> дискретная смена режима".

Шаблоны микросостояний фитятся один раз на baseline-записи (см.
core.microstate_fit.fit_microstates) и передаются сюда готовыми. Для
каждого окна берётся топография в момент пика GFP внутри окна и
классифицируется по ближайшему (по модулю корреляции) шаблону.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.windowing import EEGWindow
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.features import FeatureSet
from eeg_art.core.microstate_fit import global_field_power, label_topography
from eeg_art.core.registry import MethodRegistry


@MethodRegistry.register("microstates_regime.feature_extractor", "gfp_peak")
class MicrostateFeatureExtractor:
    def __init__(self, templates: np.ndarray, labels: list[str] | None = None):
        self.templates = templates
        self.labels = labels or [chr(ord("A") + i) for i in range(templates.shape[0])]

    def extract(self, window: EEGWindow, spatial: SpatialModel) -> FeatureSet:
        gfp = global_field_power(window.data)
        peak_t = int(np.argmax(gfp))
        topo = window.data[:, peak_t]

        label_idx, corr = label_topography(topo, self.templates)
        return FeatureSet(
            timestamp=window.timestamp,
            fields={
                "microstate_label": self.labels[label_idx],
                "microstate_index": label_idx,
                "microstate_corr": corr,
                "gfp_peak": float(gfp[peak_t]),
            },
        )
