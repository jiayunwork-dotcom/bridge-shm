from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import json
import os
from config import BRIDGES_DIR


class SensorType(Enum):
    ACCELERATION = "acceleration"
    STRAIN = "strain"
    DISPLACEMENT = "displacement"
    TEMPERATURE = "temperature"
    WIND_SPEED = "wind_speed"


@dataclass
class Sensor:
    id: str
    name: str
    type: SensorType
    channel: int
    location: Tuple[float, float, float]
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    sampling_rate: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "channel": self.channel,
            "location": list(self.location),
            "direction": list(self.direction),
            "sampling_rate": self.sampling_rate
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Sensor':
        return cls(
            id=data["id"],
            name=data["name"],
            type=SensorType(data["type"]),
            channel=data["channel"],
            location=tuple(data["location"]),
            direction=tuple(data.get("direction", [0.0, 0.0, 1.0])),
            sampling_rate=data.get("sampling_rate")
        )


@dataclass
class Bridge:
    id: str
    name: str
    description: str = ""
    sensors: List[Sensor] = field(default_factory=list)
    baseline_event_id: Optional[str] = None
    alert_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "sensors": [s.to_dict() for s in self.sensors],
            "baseline_event_id": self.baseline_event_id,
            "alert_config": self.alert_config
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Bridge':
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            sensors=[Sensor.from_dict(s) for s in data.get("sensors", [])],
            baseline_event_id=data.get("baseline_event_id"),
            alert_config=data.get("alert_config", {})
        )

    def save(self) -> None:
        filepath = os.path.join(BRIDGES_DIR, f"{self.id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, bridge_id: str) -> Optional['Bridge']:
        filepath = os.path.join(BRIDGES_DIR, f"{bridge_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def list_all(cls) -> List['Bridge']:
        bridges = []
        if not os.path.exists(BRIDGES_DIR):
            return bridges
        for filename in os.listdir(BRIDGES_DIR):
            if filename.endswith('.json'):
                bridge_id = filename[:-5]
                bridge = cls.load(bridge_id)
                if bridge:
                    bridges.append(bridge)
        return bridges

    def get_sensor_by_channel(self, channel: int) -> Optional[Sensor]:
        for sensor in self.sensors:
            if sensor.channel == channel:
                return sensor
        return None

    def get_sensors_by_type(self, sensor_type: SensorType) -> List[Sensor]:
        return [s for s in self.sensors if s.type == sensor_type]
