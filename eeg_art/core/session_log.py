"""
core.session_log
-------------------
Минимальный журнал сессии: кадры рендера (PNG) + признаки по каждому окну
(JSON) с единым таймкодом. В дальнейшем сюда же пишутся видео/аудио/gaze
потоки — интерфейс (append_frame / append_features) не меняется.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class SessionLog:
    output_dir: str
    frame_index: int = 0
    features_log: list = field(default_factory=list)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)

    def append_frame(self, image_array: np.ndarray, timestamp: float) -> str:
        """image_array: (H, W) или (H, W, 3), значения в [0, 1]."""
        arr = np.clip(image_array, 0, 1)
        arr8 = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr8)
        path = os.path.join(self.output_dir, "frames", f"frame_{self.frame_index:05d}.png")
        img.save(path)
        self.frame_index += 1
        return path

    def append_stream(self, stream_name: str, image_array: np.ndarray, timestamp: float, index: int) -> str:
        """Архивный вспомогательный поток (например 'room_camera',
        'artwork_camera'), не входящий в основной ряд кадров генерации —
        своя подпапка streams/<stream_name>/, свой счётчик index (обычно
        индекс окна, чтобы имя файла было привязано к общему таймкоду)."""
        stream_dir = os.path.join(self.output_dir, "streams", stream_name)
        os.makedirs(stream_dir, exist_ok=True)
        arr = np.clip(image_array, 0, 1)
        arr8 = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr8)
        path = os.path.join(stream_dir, f"{stream_name}_{index:05d}_t{timestamp:07.2f}.png")
        img.save(path)
        return path

    def append_features(self, timestamp: float, fields: dict) -> None:
        self.features_log.append({"timestamp": timestamp, "fields": _to_jsonable(fields)})

    def flush(self) -> None:
        with open(os.path.join(self.output_dir, "features_log.json"), "w") as f:
            json.dump(self.features_log, f, indent=2)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
