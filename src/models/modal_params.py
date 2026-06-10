from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
import json
import os
from config import TEST_EVENTS_DIR


@dataclass
class ModeShape:
    frequency: float
    damping_ratio: float
    mode_vector: np.ndarray
    mac_value: Optional[float] = None
    damping_quality: str = "good"

    def to_dict(self) -> dict:
        return {
            "frequency": self.frequency,
            "damping_ratio": self.damping_ratio,
            "mode_vector": self.mode_vector.tolist(),
            "mac_value": self.mac_value,
            "damping_quality": self.damping_quality
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ModeShape':
        return cls(
            frequency=data["frequency"],
            damping_ratio=data["damping_ratio"],
            mode_vector=np.array(data["mode_vector"]),
            mac_value=data.get("mac_value"),
            damping_quality=data.get("damping_quality", "good")
        )


@dataclass
class ModalParams:
    event_id: str
    frequencies: np.ndarray
    singular_values: np.ndarray
    mode_shapes: List[ModeShape] = field(default_factory=list)
    psd_matrix: Optional[np.ndarray] = None
    params: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "frequencies": self.frequencies.tolist(),
            "singular_values": self.singular_values.tolist(),
            "mode_shapes": [ms.to_dict() for ms in self.mode_shapes],
            "params": self.params
        }

    def save(self) -> None:
        event_dir = os.path.join(TEST_EVENTS_DIR, self.event_id)
        os.makedirs(event_dir, exist_ok=True)
        filepath = os.path.join(event_dir, "modal_params.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, event_id: str) -> Optional['ModalParams']:
        event_dir = os.path.join(TEST_EVENTS_DIR, event_id)
        filepath = os.path.join(event_dir, "modal_params.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            event_id=data["event_id"],
            frequencies=np.array(data["frequencies"]),
            singular_values=np.array(data["singular_values"]),
            mode_shapes=[ModeShape.from_dict(ms) for ms in data.get("mode_shapes", [])],
            params=data.get("params", {})
        )

    def get_mode_frequencies(self) -> np.ndarray:
        return np.array([ms.frequency for ms in self.mode_shapes])

    def get_mode_damping_ratios(self) -> np.ndarray:
        return np.array([ms.damping_ratio for ms in self.mode_shapes])

    def get_mode_matrix(self) -> np.ndarray:
        if not self.mode_shapes:
            return np.array([])
        return np.column_stack([ms.mode_vector for ms in self.mode_shapes])
