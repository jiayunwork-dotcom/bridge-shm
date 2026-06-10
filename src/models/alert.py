from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime
import json
import os
from config import TEST_EVENTS_DIR


class AlertLevel(Enum):
    YELLOW = "yellow"
    RED = "red"
    INFO = "info"


@dataclass
class Alert:
    id: str
    bridge_id: str
    event_id: str
    level: AlertLevel
    trigger_time: datetime
    metric: str
    current_value: float
    threshold: float
    suggestion: str = ""
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bridge_id": self.bridge_id,
            "event_id": self.event_id,
            "level": self.level.value,
            "trigger_time": self.trigger_time.isoformat(),
            "metric": self.metric,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "suggestion": self.suggestion,
            "acknowledged": self.acknowledged
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Alert':
        return cls(
            id=data["id"],
            bridge_id=data["bridge_id"],
            event_id=data["event_id"],
            level=AlertLevel(data["level"]),
            trigger_time=datetime.fromisoformat(data["trigger_time"]),
            metric=data["metric"],
            current_value=data["current_value"],
            threshold=data["threshold"],
            suggestion=data.get("suggestion", ""),
            acknowledged=data.get("acknowledged", False)
        )

    def save(self) -> None:
        event_dir = os.path.join(TEST_EVENTS_DIR, self.event_id)
        os.makedirs(event_dir, exist_ok=True)
        alerts_file = os.path.join(event_dir, "alerts.json")
        
        alerts = []
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
        
        alert_dicts = [a for a in alerts if a["id"] != self.id]
        alert_dicts.append(self.to_dict())
        
        with open(alerts_file, 'w', encoding='utf-8') as f:
            json.dump(alert_dicts, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_by_event(cls, event_id: str) -> list:
        event_dir = os.path.join(TEST_EVENTS_DIR, event_id)
        alerts_file = os.path.join(event_dir, "alerts.json")
        if not os.path.exists(alerts_file):
            return []
        with open(alerts_file, 'r', encoding='utf-8') as f:
            alerts = json.load(f)
        return [cls.from_dict(a) for a in alerts]

    @classmethod
    def load_by_bridge(cls, bridge_id: str, unacknowledged_only: bool = False) -> list:
        from .test_event import TestEvent
        events = TestEvent.list_by_bridge(bridge_id)
        all_alerts = []
        for event in events:
            alerts = cls.load_by_event(event.id)
            for alert in alerts:
                if not unacknowledged_only or not alert.acknowledged:
                    all_alerts.append(alert)
        all_alerts.sort(key=lambda a: a.trigger_time, reverse=True)
        return all_alerts
