"""
modules.particle_bloom.generator_particles
------------------------------------------------
Ветвь "particle_bloom": визуальный стиль, вдохновлённый роликом
генеративщика (рой мелких частиц + растущие кляксы-аттракторы,
сливающиеся в плотную живописную текстуру с накоплением следа), но с
полностью документированным маппингом на band power — тем же, что
использует spectral_lenia (переиспользует SpectralFeatureExtractor и
core.calibration.Baseline, отдельный feature-модуль не нужен).

МАППИНГ ("партитура" метода):

    - У каждого электрода (10/20) — своя точка на холсте. Каждое окно из
      неё рождаются новые частицы с вероятностью, растущей с z-оценкой
      средней (по 5 диапазонам) мощности этого канала. Так электроды с
      устойчиво высокой активностью со временем "разрастаются" в
      кляксу — прямая визуализация НАКОПЛЕННОЙ, а не мгновенной
      активности области.
    - Цвет частицы = доминирующий частотный диапазон в этом канале в
      этот момент (argmax z по delta/theta/alpha/beta/gamma), палитра
      цветов совпадает с баром band power в spectral_lenia (та же
      цветовая легенда во всём проекте: синий-голубой-зелёный-жёлтый-
      красный = delta-theta-alpha-beta-gamma).
    - Направление сноса частиц (flow field) = градиент интерполированного
      поля alpha (частицы стекают в сторону роста alpha — визуальная
      метафора "внимание утекает к спокойствию"); скорость сноса
      масштабируется полем gamma (больше высокочастотной активности —
      быстрее и хаотичнее движение).
    - Фоновая "пыль" (мелкие несвязанные штрихи, как в референсе) —
      частицы, рождающиеся по всему холсту случайно, с интенсивностью
      спавна = общая широкополосная z-активность (среднее по всем
      каналам и диапазонам); тусклые, короткоживущие.
    - Холст — накопительный буфер с затуханием (decay < 1 за кадр):
      область без новых частиц медленно гаснет, область с устойчивым
      спавном насыщается и "цветёт" — то самое превращение разреженных
      точек в плотную заливку к концу сессии, как в референсном ролике.
"""

from __future__ import annotations

import numpy as np

from eeg_art.core.features import FeatureSet
from eeg_art.core.spatial_model import SpatialModel
from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry

BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_COLORS = np.array([
    [0.27, 0.27, 1.00],   # delta — синий
    [0.27, 1.00, 1.00],   # theta — голубой
    [0.27, 1.00, 0.27],   # alpha — зелёный
    [1.00, 1.00, 0.27],   # beta  — жёлтый
    [1.00, 0.27, 0.27],   # gamma — красный
])

PARAM_SCHEMA = {
    "grid_size": ParamSpec(220, 64, 512, "px", "Сторона квадратного холста."),
    "decay": ParamSpec(0.965, 0.80, 0.999, "", "Коэффициент затухания холста за кадр — ниже -> быстрее гаснет след."),
    "channel_spawn_rate": ParamSpec(35, 1, 200, "частиц/окно", "Макс. число новых частиц с канала при z->+inf."),
    "background_spawn_rate": ParamSpec(120, 0, 1000, "частиц/окно", "Макс. число фоновых частиц при общей активности->+inf."),
    "particle_speed": ParamSpec(2.5, 0.0, 20.0, "px/окно", "Базовая скорость сноса частиц по полю alpha-градиента."),
    "lifespan_channel": ParamSpec(25, 1, 200, "окон", "Время жизни частицы, рождённой на канале."),
    "lifespan_background": ParamSpec(10, 1, 200, "окон", "Время жизни фоновой частицы (короче — она тусклее и мельче)."),
    "channel_alpha": ParamSpec(0.35, 0.01, 1.0, "", "Непрозрачность отпечатка частицы канала за кадр."),
    "background_alpha": ParamSpec(0.10, 0.01, 1.0, "", "Непрозрачность отпечатка фоновой частицы."),
}


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


@MethodRegistry.register("particle_bloom.generator", "flow_attractor", schema=PARAM_SCHEMA)
class ParticleBloomGenerator:
    def __init__(
        self,
        spatial: SpatialModel,
        grid_size: int = 220,
        field_shape: tuple[int, int] | None = None,
        decay: float = 0.965,
        channel_spawn_rate: float = 35,
        background_spawn_rate: float = 120,
        particle_speed: float = 2.5,
        lifespan_channel: int = 25,
        lifespan_background: int = 10,
        channel_alpha: float = 0.35,
        background_alpha: float = 0.10,
        rng=None,
    ):
        self.spatial = spatial
        # field_shape=(H,W) перекрывает квадратный grid_size — см.
        # core.field_sizing для вычисления формы по пропорциям произведения.
        self.shape: tuple[int, int] = field_shape if field_shape is not None else (grid_size, grid_size)
        self.grid_size = grid_size
        self.decay = decay
        self.channel_spawn_rate = channel_spawn_rate
        self.background_spawn_rate = background_spawn_rate
        self.particle_speed = particle_speed
        self.lifespan_channel = lifespan_channel
        self.lifespan_background = lifespan_background
        self.channel_alpha = channel_alpha
        self.background_alpha = background_alpha
        self.rng = rng or np.random.default_rng()

        h, w = self.shape
        # пиксельные координаты электродов на холсте (из [-1,1] в [0,W)/[0,H))
        self.channel_px = {}
        for ch, (x, y) in spatial.channel_positions.items():
            px = (x + 1) / 2 * (w - 1)
            py = (1 - (y + 1) / 2) * (h - 1)
            self.channel_px[ch] = (px, py)

        # частицы храним как обычные numpy-массивы (не Grid/Graph/Volume —
        # система частиц принципиально другая парадигма состояния)
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.color = np.zeros((0, 3))
        self.life = np.zeros((0,))
        self.alpha_val = np.zeros((0,))

    def init_state(self) -> np.ndarray:
        h, w = self.shape
        return np.zeros((h, w, 3))

    def _spawn(self, positions, color, life, alpha_val):
        n = len(positions)
        if n == 0:
            return
        self.pos = np.vstack([self.pos, positions])
        self.vel = np.vstack([self.vel, np.zeros((n, 2))])
        self.color = np.vstack([self.color, color])
        self.life = np.concatenate([self.life, np.full(n, life)])
        self.alpha_val = np.concatenate([self.alpha_val, np.full(n, alpha_val)])

    def step(self, features: FeatureSet, spatial: SpatialModel, canvas: np.ndarray) -> np.ndarray:
        z = features.fields["band_power"]  # {band: {channel: z}}
        h, w = self.shape

        # --- спавн: частицы каналов ---
        all_z_vals = []
        for ch in spatial.channels:
            band_z = np.array([z[b][ch] for b in BAND_ORDER])
            all_z_vals.extend(band_z.tolist())
            z_total = float(band_z.mean())
            dominant = int(np.argmax(band_z))

            n_new = int(self.rng.poisson(self.channel_spawn_rate * _logistic(z_total) * 0.3))
            if n_new > 0:
                px, py = self.channel_px[ch]
                jitter = self.rng.normal(scale=2.5, size=(n_new, 2))
                positions = np.array([px, py]) + jitter
                colors = np.tile(BAND_COLORS[dominant], (n_new, 1))
                self._spawn(positions, colors, self.lifespan_channel, self.channel_alpha)

        # --- спавн: фоновая пыль ---
        global_z = float(np.mean(all_z_vals))
        n_bg = int(self.rng.poisson(self.background_spawn_rate * _logistic(global_z) * 0.3))
        if n_bg > 0:
            positions = np.stack([
                self.rng.uniform(0, w, size=n_bg),
                self.rng.uniform(0, h, size=n_bg),
            ], axis=1)
            dim = 0.5 + 0.3 * self.rng.random((n_bg, 3))
            self._spawn(positions, dim, self.lifespan_background, self.background_alpha)

        # --- flow field: градиент alpha-поля, масштаб скорости — gamma-поле ---
        if len(self.pos) > 0:
            alpha_field = spatial.interpolate_to_grid(z["alpha"], grid_size=self.shape)
            gamma_field = spatial.interpolate_to_grid(z["gamma"], grid_size=self.shape)
            gy, gx = np.gradient(alpha_field)
            speed_field = self.particle_speed * (0.3 + _logistic(gamma_field))

            ix = np.clip(self.pos[:, 0].astype(int), 0, w - 1)
            iy = np.clip(self.pos[:, 1].astype(int), 0, h - 1)
            flow = np.stack([gx[iy, ix], gy[iy, ix]], axis=1)
            flow_norm = flow / (np.linalg.norm(flow, axis=1, keepdims=True) + 1e-6)

            self.vel = 0.7 * self.vel + 0.3 * flow_norm * speed_field[iy, ix, None]
            self.vel += self.rng.normal(scale=0.3, size=self.vel.shape)
            self.pos += self.vel
            self.life -= 1

            alive = (
                (self.life > 0)
                & (self.pos[:, 0] >= 0) & (self.pos[:, 0] < w)
                & (self.pos[:, 1] >= 0) & (self.pos[:, 1] < h)
            )
            self.pos, self.vel = self.pos[alive], self.vel[alive]
            self.color, self.life, self.alpha_val = self.color[alive], self.life[alive], self.alpha_val[alive]

        # --- растеризация ---
        splat = np.zeros_like(canvas)
        if len(self.pos) > 0:
            ix = np.clip(self.pos[:, 0].astype(int), 0, w - 1)
            iy = np.clip(self.pos[:, 1].astype(int), 0, h - 1)
            np.add.at(splat, (iy, ix), self.color * self.alpha_val[:, None])

        new_canvas = np.clip(canvas * self.decay + splat, 0.0, 1.0)
        return new_canvas
