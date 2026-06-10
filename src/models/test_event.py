from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
from config import TEST_EVENTS_DIR


@dataclass
class TestEventMetadata:
    collection_time: datetime
    weather: str = "unknown"
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    traffic_status: str = "normal"
    operator: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "collection_time": self.collection_time.isoformat(),
            "weather": self.weather,
            "temperature": self.temperature,
            "wind_speed": self.wind_speed,
            "traffic_status": self.traffic_status,
            "operator": self.operator,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TestEventMetadata':
        return cls(
            collection_time=datetime.fromisoformat(data["collection_time"]),
            weather=data.get("weather", "unknown"),
            temperature=data.get("temperature"),
            wind_speed=data.get("wind_speed"),
            traffic_status=data.get("traffic_status", "normal"),
            operator=data.get("operator", ""),
            notes=data.get("notes", "")
        )


@dataclass
class TestEvent:
    id: str
    bridge_id: str
    name: str
    metadata: TestEventMetadata
    sampling_rate: float
    data: pd.DataFrame = field(repr=False)
    channel_names: List[str] = field(default_factory=list)
    preprocessing_params: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bridge_id": self.bridge_id,
            "name": self.name,
            "metadata": self.metadata.to_dict(),
            "sampling_rate": self.sampling_rate,
            "channel_names": self.channel_names,
            "preprocessing_params": self.preprocessing_params
        }

    def save(self) -> None:
        event_dir = os.path.join(TEST_EVENTS_DIR, self.id)
        os.makedirs(event_dir, exist_ok=True)

        metadata_path = os.path.join(event_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        data_path = os.path.join(event_dir, "data.parquet")
        self.data.to_parquet(data_path, index=False)

    @classmethod
    def load(cls, event_id: str) -> Optional['TestEvent']:
        event_dir = os.path.join(TEST_EVENTS_DIR, event_id)
        metadata_path = os.path.join(event_dir, "metadata.json")
        data_path = os.path.join(event_dir, "data.parquet")

        if not os.path.exists(metadata_path) or not os.path.exists(data_path):
            return None

        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.read_parquet(data_path)

        return cls(
            id=data["id"],
            bridge_id=data["bridge_id"],
            name=data["name"],
            metadata=TestEventMetadata.from_dict(data["metadata"]),
            sampling_rate=data["sampling_rate"],
            data=df,
            channel_names=data.get("channel_names", list(df.columns)),
            preprocessing_params=data.get("preprocessing_params", {})
        )

    @classmethod
    def list_by_bridge(cls, bridge_id: str) -> List['TestEvent']:
        events = []
        if not os.path.exists(TEST_EVENTS_DIR):
            return events
        for event_id in os.listdir(TEST_EVENTS_DIR):
            event_dir = os.path.join(TEST_EVENTS_DIR, event_id)
            if os.path.isdir(event_dir):
                event = cls.load(event_id)
                if event and event.bridge_id == bridge_id:
                    events.append(event)
        events.sort(key=lambda e: e.metadata.collection_time)
        return events

    def get_time_vector(self) -> np.ndarray:
        n_samples = len(self.data)
        return np.arange(n_samples) / self.sampling_rate

    def get_channel_data(self, channel_name: str) -> np.ndarray:
        return self.data[channel_name].values

    def clip_by_time(self, start_time: float, end_time: float) -> None:
        time_vec = self.get_time_vector()
        mask = (time_vec >= start_time) & (time_vec <= end_time)
        self.data = self.data[mask].reset_index(drop=True)
