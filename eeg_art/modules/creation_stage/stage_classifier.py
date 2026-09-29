"""
modules.creation_stage.stage_classifier
--------------------------------------------
Классификация стадии создания произведения по признакам видео холста
(video_features.py). Правило-based — каждый порог задокументирован и
настраиваем (PARAM_SCHEMA), решение воспроизводимо и объяснимо.

Пороги откалиброваны эмпирически на синтетике с известной последовательностью
стадий (ingestion.synthetic_video.generate_synthetic_creation_video) — см.
demo_creation_stage.py, сравнение с ground truth.

Важное ограничение, вскрывшееся при калибровке: edge_density, усреднённая
по всему кадру, слабо разделяет стадии — тонкий штрих на большом холсте
даёт слишком разбавленный сигнал. Полезнее оказались fill_fraction
(кумулятивная, монотонно растёт и выходит на плато после подмалёвка) и
дуэт motion+saturation для стадий после плато. edge_density оставлена в
video_features.py для отображения и как кандидат на будущую доработку
(считать не по всему кадру, а по локально изменившейся области).
"""

from __future__ import annotations

from eeg_art.core.param_spec import ParamSpec
from eeg_art.core.registry import MethodRegistry

STAGES = ["эскиз", "подмалёвок", "работа с цветом", "детализация", "пауза/подпись"]

PARAM_SCHEMA = {
    "low_fill_threshold": ParamSpec(0.08, 0.01, 0.9, "",
        "Ниже — холст ещё почти пуст (эскиз); выше — уже есть подмалёвок и позже."),
    "low_sat_threshold": ParamSpec(0.05, 0.0, 1.0, "",
        "Ниже — малоцветная тональная проработка (подмалёвок); выше — цвет уже введён."),
    "detail_motion_threshold": ParamSpec(0.003, 0.0, 0.5, "",
        "Выше — крупные активные изменения (работа с цветом); ниже — точечные правки (детализация)."),
    "pause_motion_threshold": ParamSpec(0.00015, 0.0, 0.2, "",
        "EMA motion ниже -> устойчивая пауза/подпись (должен быть меньше типичного motion детализации)."),
    "pause_ema_alpha": ParamSpec(0.3, 0.01, 1.0, "",
        "Коэффициент сглаживания motion для детекции паузы — выше -> быстрее реагирует на остановку."),
}


@MethodRegistry.register("creation_stage.classifier", "rule_based", schema=PARAM_SCHEMA)
class RuleBasedStageClassifier:
    def __init__(self, **params):
        defaults = {k: v.default for k, v in PARAM_SCHEMA.items()}
        self.p = {**defaults, **params}
        self.motion_ema = None

    def classify(self, features: dict) -> str:
        f = features
        alpha = self.p["pause_ema_alpha"]
        self.motion_ema = f["motion"] if self.motion_ema is None else (
            alpha * f["motion"] + (1 - alpha) * self.motion_ema
        )

        if self.motion_ema < self.p["pause_motion_threshold"] and f["fill_fraction"] > self.p["low_fill_threshold"]:
            return "пауза/подпись"
        if f["fill_fraction"] < self.p["low_fill_threshold"]:
            return "эскиз"
        if f["saturation"] < self.p["low_sat_threshold"]:
            return "подмалёвок"
        if f["motion"] > self.p["detail_motion_threshold"]:
            return "работа с цветом"
        return "детализация"
