"""
modules.connectivity_graph.generator_graph
----------------------------------------------
Генератор ветви "связность -> граф-автомат".

Мэппинг признак -> динамика ("партитура" метода):

    edge_weights = PLV-матрица окна напрямую (без искусственных порогов) —
        связь между двумя электродами в автомате в точности равна
        измеренной фазовой синхронности между ними в этом окне.

Уравнение обновления — реакционно-диффузионное на графе (аналог
Lenia, но на реальной топологии связности, а не на регулярной сетке):

    state += dt * ( -L @ state            # диффузия по графовому лапласиану
                     + growth(state)        # реакция (та же bell-функция, что в Lenia)
                     + noise )

где L — лапласиан графа (core.field_abstraction.Graph.laplacian()).
growth_center/width, dt и noise_amp — настраиваемые параметры (см.
PARAM_SCHEMA), раньше были зашитыми константами модуля.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.features import FeatureSet
from eeg_art.core.field_abstraction import Graph
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry

PARAM_SCHEMA = {
    "growth_center": ParamSpec(0.15, 0.0, 1.0, "", "Центр bell-функции реакции на графе."),
    "growth_width": ParamSpec(0.05, 0.001, 0.5, ""),
    "dt": ParamSpec(0.12, 0.001, 1.0, "", "Общий шаг для диффузии и реакции за одно окно."),
    "noise_amp": ParamSpec(0.01, 0.0, 0.2, ""),
}


def _growth(u: np.ndarray, center: float, width: float) -> np.ndarray:
    return 2 * np.exp(-((u - center) ** 2) / (2 * width ** 2)) - 1


@MethodRegistry.register("connectivity_graph.generator", "default", schema=PARAM_SCHEMA)
class GraphGenerator:
    def __init__(
        self,
        spatial: SpatialModel,
        growth_center: float = 0.15,
        growth_width: float = 0.05,
        dt: float = 0.12,
        noise_amp: float = 0.01,
        rng=None,
    ):
        self.spatial = spatial
        self.growth_center = growth_center
        self.growth_width = growth_width
        self.dt = dt
        self.noise_amp = noise_amp
        self.rng = rng or np.random.default_rng()

    def init_state(self) -> Graph:
        n = len(self.spatial.channels)
        identity_weights = np.eye(n)  # до первого окна связности ещё нет
        return Graph.random_seed(self.spatial.channels, identity_weights, rng=self.rng)

    def step(self, features: FeatureSet, spatial: SpatialModel, state: Graph) -> Graph:
        plv = features.fields["plv_matrix"]
        channels = features.fields["channels"]
        assert channels == state.node_names, "порядок каналов должен совпадать"

        new_graph = Graph(
            node_names=state.node_names,
            node_values=state.node_values.copy(),
            edge_weights=plv,
        )
        L = new_graph.laplacian()

        diffusion = -L @ new_graph.node_values
        growth = _growth(new_graph.node_values, self.growth_center, self.growth_width)
        noise = self.noise_amp * self.rng.standard_normal(new_graph.node_values.shape)

        new_graph.node_values = np.clip(
            new_graph.node_values + self.dt * diffusion + self.dt * growth + noise, 0.0, 1.0
        )
        return new_graph
