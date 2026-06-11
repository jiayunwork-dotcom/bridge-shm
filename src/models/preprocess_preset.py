from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
import os
from config import PRESETS_DIR


@dataclass
class PreprocessPreset:
    id: str
    bridge_id: str
    name: str
    detrend_method: str = 'none'
    poly_order: int = 3
    filter_type: str = 'none'
    cutoff_freq: Optional[float] = None
    cutoff_freq2: Optional[float] = None
    target_sr: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bridge_id": self.bridge_id,
            "name": self.name,
            "detrend_method": self.detrend_method,
            "poly_order": self.poly_order,
            "filter_type": self.filter_type,
            "cutoff_freq": self.cutoff_freq,
            "cutoff_freq2": self.cutoff_freq2,
            "target_sr": self.target_sr
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PreprocessPreset':
        return cls(
            id=data["id"],
            bridge_id=data["bridge_id"],
            name=data["name"],
            detrend_method=data.get("detrend_method", "none"),
            poly_order=data.get("poly_order", 3),
            filter_type=data.get("filter_type", "none"),
            cutoff_freq=data.get("cutoff_freq"),
            cutoff_freq2=data.get("cutoff_freq2"),
            target_sr=data.get("target_sr")
        )

    def save(self) -> None:
        bridge_presets_dir = os.path.join(PRESETS_DIR, self.bridge_id)
        os.makedirs(bridge_presets_dir, exist_ok=True)
        filepath = os.path.join(bridge_presets_dir, f"{self.id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, bridge_id: str, preset_id: str) -> Optional['PreprocessPreset']:
        filepath = os.path.join(PRESETS_DIR, bridge_id, f"{preset_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def list_by_bridge(cls, bridge_id: str) -> List['PreprocessPreset']:
        presets = []
        bridge_presets_dir = os.path.join(PRESETS_DIR, bridge_id)
        if not os.path.exists(bridge_presets_dir):
            return presets
        for filename in os.listdir(bridge_presets_dir):
            if filename.endswith('.json'):
                preset_id = filename[:-5]
                preset = cls.load(bridge_id, preset_id)
                if preset:
                    presets.append(preset)
        return presets

    @classmethod
    def delete(cls, bridge_id: str, preset_id: str) -> bool:
        filepath = os.path.join(PRESETS_DIR, bridge_id, f"{preset_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
