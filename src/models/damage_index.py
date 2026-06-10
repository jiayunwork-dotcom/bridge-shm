from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import numpy as np
import json
import os
from config import TEST_EVENTS_DIR


class DamageType(Enum):
    FREQUENCY_CHANGE = "frequency_change"
    FLEXIBILITY_CHANGE = "flexibility_change"
    CURVATURE_MODE = "curvature_mode"
    MODAL_STRAIN_ENERGY = "modal_strain_energy"


@dataclass
class DamageIndex:
    event_id: str
    baseline_event_id: str
    damage_type: DamageType
    values: np.ndarray
    threshold: Optional[float] = None
    locations: Optional[List[str]] = None
    temperature_compensated: bool = False

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "baseline_event_id": self.baseline_event_id,
            "damage_type": self.damage_type.value,
            "values": self.values.tolist(),
            "threshold": self.threshold,
            "locations": self.locations,
            "temperature_compensated": self.temperature_compensated
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DamageIndex':
        return cls(
            event_id=data["event_id"],
            baseline_event_id=data["baseline_event_id"],
            damage_type=DamageType(data["damage_type"]),
            values=np.array(data["values"]),
            threshold=data.get("threshold"),
            locations=data.get("locations"),
            temperature_compensated=data.get("temperature_compensated", False)
        )

    def save(self) -> None:
        event_dir = os.path.join(TEST_EVENTS_DIR, self.event_id)
        os.makedirs(event_dir, exist_ok=True)
        filepath = os.path.join(event_dir, f"damage_{self.damage_type.value}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, event_id: str, damage_type: DamageType) -> Optional['DamageIndex']:
        event_dir = os.path.join(TEST_EVENTS_DIR, event_id)
        filepath = os.path.join(event_dir, f"damage_{damage_type.value}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_all(cls, event_id: str) -> Dict[DamageType, 'DamageIndex']:
        indices = {}
        for dt in DamageType:
            idx = cls.load(event_id, dt)
            if idx:
                indices[dt] = idx
        return indices

    def get_anomalous_indices(self) -> np.ndarray:
        if self.threshold is None:
            return np.array([])
        return np.where(np.abs(self.values) > self.threshold)[0]
