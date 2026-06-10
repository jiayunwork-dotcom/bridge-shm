import numpy as np
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from src.models.alert import Alert, AlertLevel
from src.models.modal_params import ModalParams
from src.models.test_event import TestEvent
from src.models.bridge import Bridge


class AlertCondition(Enum):
    FREQUENCY_DROP = "frequency_drop"
    DAMPING_INCREASE = "damping_increase"
    DAMAGE_INDEX_EXCEED = "damage_index_exceed"


@dataclass
class AlertRule:
    id: str
    name: str
    bridge_id: str
    condition: AlertCondition
    mode_index: int
    threshold: float
    level: AlertLevel
    suggestion: str = ""
    enabled: bool = True
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "bridge_id": self.bridge_id,
            "condition": self.condition.value,
            "mode_index": self.mode_index,
            "threshold": self.threshold,
            "level": self.level.value,
            "suggestion": self.suggestion,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AlertRule':
        return cls(
            id=data["id"],
            name=data["name"],
            bridge_id=data["bridge_id"],
            condition=AlertCondition(data["condition"]),
            mode_index=data["mode_index"],
            threshold=data["threshold"],
            level=AlertLevel(data["level"]),
            suggestion=data.get("suggestion", ""),
            enabled=data.get("enabled", True)
        )


def load_alert_rules(bridge_id: str) -> List[AlertRule]:
    from config import BRIDGES_DIR
    import os
    import json
    
    filepath = os.path.join(BRIDGES_DIR, f"{bridge_id}_alert_rules.json")
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        rules_data = json.load(f)
    
    return [AlertRule.from_dict(r) for r in rules_data]


def save_alert_rules(bridge_id: str, rules: List[AlertRule]) -> None:
    from config import BRIDGES_DIR
    import os
    import json
    
    filepath = os.path.join(BRIDGES_DIR, f"{bridge_id}_alert_rules.json")
    rules_data = [r.to_dict() for r in rules]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)


def create_alert(
    bridge_id: str,
    event_id: str,
    level: AlertLevel,
    metric: str,
    current_value: float,
    threshold: float,
    suggestion: str = ""
) -> Alert:
    return Alert(
        id=str(uuid.uuid4()),
        bridge_id=bridge_id,
        event_id=event_id,
        level=level,
        trigger_time=datetime.now(),
        metric=metric,
        current_value=current_value,
        threshold=threshold,
        suggestion=suggestion
    )


def evaluate_alert_rules(
    bridge: Bridge,
    current_params: ModalParams,
    baseline_params: Optional[ModalParams] = None,
    damage_indices: Optional[Dict] = None
) -> List[Alert]:
    alerts = []
    rules = load_alert_rules(bridge.id)
    
    current_freqs = current_params.get_mode_frequencies()
    current_damping = current_params.get_mode_damping_ratios()
    
    for rule in rules:
        if not rule.enabled:
            continue
        
        if rule.mode_index >= len(current_freqs):
            continue
        
        alert = None
        
        if rule.condition == AlertCondition.FREQUENCY_DROP and baseline_params is not None:
            baseline_freqs = baseline_params.get_mode_frequencies()
            if rule.mode_index < len(baseline_freqs) and baseline_freqs[rule.mode_index] > 0:
                change_rate = (current_freqs[rule.mode_index] - baseline_freqs[rule.mode_index]) / baseline_freqs[rule.mode_index] * 100
                if change_rate < -abs(rule.threshold):
                    alert = create_alert(
                        bridge_id=bridge.id,
                        event_id=current_params.event_id,
                        level=rule.level,
                        metric=f"第{rule.mode_index+1}阶频率变化率",
                        current_value=change_rate,
                        threshold=-abs(rule.threshold),
                        suggestion=rule.suggestion or f"第{rule.mode_index+1}阶频率下降超过{abs(rule.threshold)}%，建议检查桥梁结构"
                    )
        
        elif rule.condition == AlertCondition.DAMPING_INCREASE:
            damping = current_damping[rule.mode_index] * 100
            if damping > rule.threshold:
                alert = create_alert(
                    bridge_id=bridge.id,
                    event_id=current_params.event_id,
                    level=rule.level,
                    metric=f"第{rule.mode_index+1}阶阻尼比",
                    current_value=damping,
                    threshold=rule.threshold,
                    suggestion=rule.suggestion or f"第{rule.mode_index+1}阶阻尼比超过{rule.threshold}%，建议检查结构阻尼特性"
                )
        
        elif rule.condition == AlertCondition.DAMAGE_INDEX_EXCEED and damage_indices is not None:
            from src.models.damage_index import DamageType
            for dt, di in damage_indices.items():
                if rule.mode_index < len(di.values):
                    if abs(di.values[rule.mode_index]) > abs(rule.threshold):
                        alert = create_alert(
                            bridge_id=bridge.id,
                            event_id=current_params.event_id,
                            level=rule.level,
                            metric=f"{dt.value} - 第{rule.mode_index+1}测点",
                            current_value=di.values[rule.mode_index],
                            threshold=abs(rule.threshold),
                            suggestion=rule.suggestion or f"损伤指标{dt.value}超过阈值，建议检查对应位置"
                        )
                        break
        
        if alert is not None:
            alerts.append(alert)
            alert.save()
    
    return alerts


def check_all_alerts(
    bridge: Bridge,
    events: List[TestEvent],
    modal_params_list: List[ModalParams],
    baseline_params: Optional[ModalParams] = None
) -> List[Alert]:
    all_alerts = []
    
    for event, params in zip(events, modal_params_list):
        from src.models.damage_index import DamageIndex
        damage_indices = DamageIndex.load_all(event.id) if event.id else {}
        
        if baseline_params is not None:
            params.event_id = event.id
            alerts = evaluate_alert_rules(
                bridge, params, baseline_params, damage_indices
            )
            all_alerts.extend(alerts)
    
    return all_alerts


def list_all_events() -> List[TestEvent]:
    from .test_event import TestEvent
    from config import TEST_EVENTS_DIR
    import os
    
    events = []
    if not os.path.exists(TEST_EVENTS_DIR):
        return events
    for event_id in os.listdir(TEST_EVENTS_DIR):
        event_dir = os.path.join(TEST_EVENTS_DIR, event_id)
        if os.path.isdir(event_dir):
            event = TestEvent.load(event_id)
            if event:
                events.append(event)
    events.sort(key=lambda e: e.metadata.collection_time)
    return events
