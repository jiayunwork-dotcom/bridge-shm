from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import json
import os
import uuid
import pandas as pd
import numpy as np
from config import UNARCHIVED_DIR


@dataclass
class UnarchivedFile:
    id: str
    bridge_id: str
    filename: str
    sampling_rate: float
    channel_names: List[str]
    n_samples: int
    duration: float
    upload_time: datetime
    event_id: Optional[str] = None
    preprocessing_params: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bridge_id": self.bridge_id,
            "filename": self.filename,
            "sampling_rate": self.sampling_rate,
            "channel_names": self.channel_names,
            "n_samples": self.n_samples,
            "duration": self.duration,
            "upload_time": self.upload_time.isoformat(),
            "event_id": self.event_id,
            "preprocessing_params": self.preprocessing_params
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'UnarchivedFile':
        return cls(
            id=data["id"],
            bridge_id=data["bridge_id"],
            filename=data["filename"],
            sampling_rate=data["sampling_rate"],
            channel_names=data["channel_names"],
            n_samples=data["n_samples"],
            duration=data["duration"],
            upload_time=datetime.fromisoformat(data["upload_time"]),
            event_id=data.get("event_id"),
            preprocessing_params=data.get("preprocessing_params", {})
        )

    def save_metadata(self) -> None:
        bridge_dir = os.path.join(UNARCHIVED_DIR, self.bridge_id)
        os.makedirs(bridge_dir, exist_ok=True)
        meta_path = os.path.join(bridge_dir, f"{self.id}_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_data(self, df: pd.DataFrame) -> None:
        bridge_dir = os.path.join(UNARCHIVED_DIR, self.bridge_id)
        os.makedirs(bridge_dir, exist_ok=True)
        data_path = os.path.join(bridge_dir, f"{self.id}_data.parquet")
        df.to_parquet(data_path, index=False)

    def load_data(self) -> Optional[pd.DataFrame]:
        data_path = os.path.join(UNARCHIVED_DIR, self.bridge_id, f"{self.id}_data.parquet")
        if os.path.exists(data_path):
            return pd.read_parquet(data_path)
        return None

    @classmethod
    def load(cls, bridge_id: str, file_id: str) -> Optional['UnarchivedFile']:
        meta_path = os.path.join(UNARCHIVED_DIR, bridge_id, f"{file_id}_meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def list_by_bridge(cls, bridge_id: str, only_unassigned: bool = False) -> List['UnarchivedFile']:
        files = []
        bridge_dir = os.path.join(UNARCHIVED_DIR, bridge_id)
        if not os.path.exists(bridge_dir):
            return files
        for filename in os.listdir(bridge_dir):
            if filename.endswith('_meta.json'):
                file_id = filename[:-10]
                uf = cls.load(bridge_id, file_id)
                if uf:
                    if only_unassigned and uf.event_id is not None:
                        continue
                    files.append(uf)
        files.sort(key=lambda f: f.upload_time, reverse=True)
        return files

    @classmethod
    def delete(cls, bridge_id: str, file_id: str) -> bool:
        bridge_dir = os.path.join(UNARCHIVED_DIR, bridge_id)
        meta_path = os.path.join(bridge_dir, f"{file_id}_meta.json")
        data_path = os.path.join(bridge_dir, f"{file_id}_data.parquet")
        deleted = False
        if os.path.exists(meta_path):
            os.remove(meta_path)
            deleted = True
        if os.path.exists(data_path):
            os.remove(data_path)
            deleted = True
        return deleted

    def assign_to_event(self, event_id: str) -> None:
        self.event_id = event_id
        self.save_metadata()

    def unassign_from_event(self) -> None:
        self.event_id = None
        self.save_metadata()
