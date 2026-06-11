from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
import json
import os
import uuid
from config import BRIDGES_DIR


class MetricType(Enum):
    RMS_AMPLITUDE = "rms_amplitude"
    PEAK_TO_PEAK = "peak_to_peak"
    FREQUENCY_OFFSET = "frequency_offset"
    BASELINE_DRIFT = "baseline_drift"


class ComparisonType(Enum):
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    OUT_OF_RANGE = "out_of_range"


class PriorityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertStatus(Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    IGNORED = "ignored"


class LogicOperator(Enum):
    AND = "and"
    OR = "or"


class AuditOperationType(Enum):
    ACKNOWLEDGE = "acknowledge"
    IGNORE = "ignore"
    DELETE = "delete"
    BATCH_ACKNOWLEDGE = "batch_acknowledge"
    BATCH_IGNORE = "batch_ignore"
    BATCH_DELETE = "batch_delete"


METRIC_LABELS = {
    "rms_amplitude": "均方根幅值",
    "peak_to_peak": "峰峰值",
    "frequency_offset": "频率偏移量",
    "baseline_drift": "基线漂移量"
}

COMPARISON_LABELS = {
    "greater_than": "大于",
    "less_than": "小于",
    "out_of_range": "超出范围"
}

PRIORITY_COLORS = {
    "high": "danger",
    "medium": "warning",
    "low": "info"
}

PRIORITY_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低"
}

STATUS_LABELS = {
    "pending": "待处理",
    "acknowledged": "已确认",
    "ignored": "已忽略"
}

STATUS_COLORS = {
    "pending": "secondary",
    "acknowledged": "success",
    "ignored": "muted"
}

LOGIC_LABELS = {
    "and": "AND (全部满足)",
    "or": "OR (任一满足)"
}

AUDIT_OPERATION_LABELS = {
    "acknowledge": "确认",
    "ignore": "忽略",
    "delete": "删除",
    "batch_acknowledge": "批量确认",
    "batch_ignore": "批量忽略",
    "batch_delete": "批量删除"
}

AUDIT_OPERATION_COLORS = {
    "acknowledge": "success",
    "ignore": "secondary",
    "delete": "danger",
    "batch_acknowledge": "success",
    "batch_ignore": "secondary",
    "batch_delete": "danger"
}


@dataclass
class RuleCondition:
    id: str
    sensor_channels: List[int]
    metric_type: MetricType
    comparison: ComparisonType
    threshold: float
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    duration_seconds: int = 5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sensor_channels": self.sensor_channels,
            "metric_type": self.metric_type.value,
            "comparison": self.comparison.value,
            "threshold": self.threshold,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
            "duration_seconds": self.duration_seconds
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RuleCondition':
        return cls(
            id=data.get("id", "cond_" + uuid.uuid4().hex[:6]),
            sensor_channels=data.get("sensor_channels", []),
            metric_type=MetricType(data["metric_type"]),
            comparison=ComparisonType(data["comparison"]),
            threshold=data.get("threshold", 0.0),
            threshold_min=data.get("threshold_min"),
            threshold_max=data.get("threshold_max"),
            duration_seconds=data.get("duration_seconds", 5)
        )


@dataclass
class AlertRule:
    id: str
    bridge_id: str
    name: str
    sensor_channels: List[int]
    metric_type: MetricType
    comparison: ComparisonType
    threshold: float
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    duration_seconds: int = 5
    priority: PriorityLevel = PriorityLevel.MEDIUM
    linked_event_id: Optional[str] = None
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    is_composite: bool = False
    conditions: List[RuleCondition] = field(default_factory=list)
    logic_operator: LogicOperator = LogicOperator.AND

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bridge_id": self.bridge_id,
            "name": self.name,
            "sensor_channels": self.sensor_channels,
            "metric_type": self.metric_type.value,
            "comparison": self.comparison.value,
            "threshold": self.threshold,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
            "duration_seconds": self.duration_seconds,
            "priority": self.priority.value,
            "linked_event_id": self.linked_event_id,
            "enabled": self.enabled,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "created_at": self.created_at.isoformat(),
            "is_composite": self.is_composite,
            "conditions": [c.to_dict() for c in self.conditions],
            "logic_operator": self.logic_operator.value
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AlertRule':
        conditions_raw = data.get("conditions", [])
        conditions = [RuleCondition.from_dict(c) for c in conditions_raw] if conditions_raw else []
        return cls(
            id=data["id"],
            bridge_id=data["bridge_id"],
            name=data["name"],
            sensor_channels=data.get("sensor_channels", []),
            metric_type=MetricType(data["metric_type"]),
            comparison=ComparisonType(data["comparison"]),
            threshold=data.get("threshold", 0.0),
            threshold_min=data.get("threshold_min"),
            threshold_max=data.get("threshold_max"),
            duration_seconds=data.get("duration_seconds", 5),
            priority=PriorityLevel(data.get("priority", "medium")),
            linked_event_id=data.get("linked_event_id"),
            enabled=data.get("enabled", True),
            last_triggered=datetime.fromisoformat(data["last_triggered"]) if data.get("last_triggered") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            is_composite=data.get("is_composite", False),
            conditions=conditions,
            logic_operator=LogicOperator(data.get("logic_operator", "and"))
        )


@dataclass
class AnomalyAlertEvent:
    id: str
    bridge_id: str
    rule_id: str
    rule_name: str
    trigger_time: datetime
    sensor_channel: int
    metric_value: float
    metric_type: MetricType
    priority: PriorityLevel
    status: AlertStatus = AlertStatus.PENDING
    unarchived_file_id: Optional[str] = None
    trigger_offset_seconds: float = 0.0
    processing_notes: str = ""
    acknowledged_at: Optional[datetime] = None
    ignored_at: Optional[datetime] = None
    is_composite: bool = False
    skipped_condition_ids: List[str] = field(default_factory=list)
    triggered_condition_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bridge_id": self.bridge_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "trigger_time": self.trigger_time.isoformat(),
            "sensor_channel": self.sensor_channel,
            "metric_value": self.metric_value,
            "metric_type": self.metric_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "unarchived_file_id": self.unarchived_file_id,
            "trigger_offset_seconds": self.trigger_offset_seconds,
            "processing_notes": self.processing_notes,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "ignored_at": self.ignored_at.isoformat() if self.ignored_at else None,
            "is_composite": self.is_composite,
            "skipped_condition_ids": self.skipped_condition_ids,
            "triggered_condition_ids": self.triggered_condition_ids
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AnomalyAlertEvent':
        return cls(
            id=data["id"],
            bridge_id=data["bridge_id"],
            rule_id=data["rule_id"],
            rule_name=data["rule_name"],
            trigger_time=datetime.fromisoformat(data["trigger_time"]),
            sensor_channel=data["sensor_channel"],
            metric_value=data["metric_value"],
            metric_type=MetricType(data["metric_type"]),
            priority=PriorityLevel(data["priority"]),
            status=AlertStatus(data.get("status", "pending")),
            unarchived_file_id=data.get("unarchived_file_id"),
            trigger_offset_seconds=data.get("trigger_offset_seconds", 0.0),
            processing_notes=data.get("processing_notes", ""),
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None,
            ignored_at=datetime.fromisoformat(data["ignored_at"]) if data.get("ignored_at") else None,
            is_composite=data.get("is_composite", False),
            skipped_condition_ids=data.get("skipped_condition_ids", []),
            triggered_condition_ids=data.get("triggered_condition_ids", [])
        )


@dataclass
class AlertAuditLog:
    id: str
    bridge_id: str
    operation_type: AuditOperationType
    event_ids: List[str]
    status_before: Optional[AlertStatus] = None
    status_after: Optional[AlertStatus] = None
    operation_time: datetime = field(default_factory=datetime.now)
    operator: str = "system"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bridge_id": self.bridge_id,
            "operation_type": self.operation_type.value,
            "event_ids": self.event_ids,
            "status_before": self.status_before.value if self.status_before else None,
            "status_after": self.status_after.value if self.status_after else None,
            "operation_time": self.operation_time.isoformat(),
            "operator": self.operator
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AlertAuditLog':
        return cls(
            id=data["id"],
            bridge_id=data["bridge_id"],
            operation_type=AuditOperationType(data["operation_type"]),
            event_ids=data.get("event_ids", []),
            status_before=AlertStatus(data["status_before"]) if data.get("status_before") else None,
            status_after=AlertStatus(data["status_after"]) if data.get("status_after") else None,
            operation_time=datetime.fromisoformat(data["operation_time"]),
            operator=data.get("operator", "system")
        )


def _get_alert_rules_path(bridge_id: str) -> str:
    return os.path.join(BRIDGES_DIR, f"{bridge_id}_anomaly_rules.json")


def _get_alert_events_path(bridge_id: str) -> str:
    return os.path.join(BRIDGES_DIR, f"{bridge_id}_anomaly_events.json")


def _get_audit_logs_path(bridge_id: str) -> str:
    return os.path.join(BRIDGES_DIR, f"{bridge_id}_alert_audit_logs.json")


def load_alert_rules(bridge_id: str) -> List[AlertRule]:
    filepath = _get_alert_rules_path(bridge_id)
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        return [AlertRule.from_dict(r) for r in rules_data]
    except (json.JSONDecodeError, KeyError, ValueError):
        return []


def save_alert_rules(bridge_id: str, rules: List[AlertRule]) -> None:
    filepath = _get_alert_rules_path(bridge_id)
    rules_data = [r.to_dict() for r in rules]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)


def add_alert_rule(bridge_id: str, rule: AlertRule) -> None:
    rules = load_alert_rules(bridge_id)
    rules.append(rule)
    save_alert_rules(bridge_id, rules)


def update_alert_rule(bridge_id: str, rule_id: str, updates: Dict[str, Any]) -> bool:
    rules = load_alert_rules(bridge_id)
    for i, r in enumerate(rules):
        if r.id == rule_id:
            for k, v in updates.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            save_alert_rules(bridge_id, rules)
            return True
    return False


def delete_alert_rule(bridge_id: str, rule_id: str) -> bool:
    rules = load_alert_rules(bridge_id)
    new_rules = [r for r in rules if r.id != rule_id]
    if len(new_rules) != len(rules):
        save_alert_rules(bridge_id, new_rules)
        return True
    return False


def load_alert_events(bridge_id: str) -> List[AnomalyAlertEvent]:
    filepath = _get_alert_events_path(bridge_id)
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            events_data = json.load(f)
        return [AnomalyAlertEvent.from_dict(e) for e in events_data]
    except (json.JSONDecodeError, KeyError, ValueError):
        return []


def save_alert_events(bridge_id: str, events: List[AnomalyAlertEvent]) -> None:
    filepath = _get_alert_events_path(bridge_id)
    events_data = [e.to_dict() for e in events]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(events_data, f, indent=2, ensure_ascii=False)


def add_alert_event(bridge_id: str, event: AnomalyAlertEvent) -> None:
    events = load_alert_events(bridge_id)
    events.append(event)
    save_alert_events(bridge_id, events)


def update_alert_event_status(
    bridge_id: str, event_id: str, status: AlertStatus, notes: str = "",
    write_audit: bool = True
) -> bool:
    events = load_alert_events(bridge_id)
    now = datetime.now()
    status_before = None
    updated = False
    for e in events:
        if e.id == event_id:
            status_before = e.status
            e.status = status
            if notes:
                e.processing_notes = notes
            if status == AlertStatus.ACKNOWLEDGED:
                e.acknowledged_at = now
            elif status == AlertStatus.IGNORED:
                e.ignored_at = now
            save_alert_events(bridge_id, events)
            updated = True
            break

    if updated and write_audit:
        op_type = AuditOperationType.ACKNOWLEDGE if status == AlertStatus.ACKNOWLEDGED else AuditOperationType.IGNORE
        add_alert_audit_log(bridge_id, op_type, [event_id], status_before, status)

    return updated


def get_events_in_window(
    bridge_id: str, window_hours: int = 24
) -> List[AnomalyAlertEvent]:
    events = load_alert_events(bridge_id)
    cutoff = datetime.now() - timedelta(hours=window_hours)
    return [e for e in events if e.trigger_time >= cutoff]


def generate_rule_id() -> str:
    return "rule_" + uuid.uuid4().hex[:8]


def generate_event_id() -> str:
    return "evt_" + uuid.uuid4().hex[:12]


def generate_condition_id() -> str:
    return "cond_" + uuid.uuid4().hex[:6]


def generate_audit_log_id() -> str:
    return "audit_" + uuid.uuid4().hex[:10]


def load_alert_audit_logs(bridge_id: str) -> List[AlertAuditLog]:
    filepath = _get_audit_logs_path(bridge_id)
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            logs_data = json.load(f)
        return [AlertAuditLog.from_dict(l) for l in logs_data]
    except (json.JSONDecodeError, KeyError, ValueError):
        return []


def save_alert_audit_logs(bridge_id: str, logs: List[AlertAuditLog]) -> None:
    filepath = _get_audit_logs_path(bridge_id)
    logs_data = [l.to_dict() for l in logs]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(logs_data, f, indent=2, ensure_ascii=False)


def add_alert_audit_log(
    bridge_id: str,
    operation_type: AuditOperationType,
    event_ids: List[str],
    status_before: Optional[AlertStatus] = None,
    status_after: Optional[AlertStatus] = None,
    operator: str = "system"
) -> AlertAuditLog:
    logs = load_alert_audit_logs(bridge_id)
    log = AlertAuditLog(
        id=generate_audit_log_id(),
        bridge_id=bridge_id,
        operation_type=operation_type,
        event_ids=event_ids,
        status_before=status_before,
        status_after=status_after,
        operator=operator
    )
    logs.append(log)
    save_alert_audit_logs(bridge_id, logs)
    return log


def batch_update_alert_event_status(
    bridge_id: str, event_ids: List[str], status: AlertStatus, notes: str = ""
) -> int:
    events = load_alert_events(bridge_id)
    now = datetime.now()
    updated_count = 0
    status_before = None
    for e in events:
        if e.id in event_ids:
            if status_before is None:
                status_before = e.status
            e.status = status
            if notes:
                e.processing_notes = notes
            if status == AlertStatus.ACKNOWLEDGED:
                e.acknowledged_at = now
            elif status == AlertStatus.IGNORED:
                e.ignored_at = now
            updated_count += 1
    save_alert_events(bridge_id, events)

    if updated_count > 0:
        op_type = AuditOperationType.BATCH_ACKNOWLEDGE if status == AlertStatus.ACKNOWLEDGED else AuditOperationType.BATCH_IGNORE
        add_alert_audit_log(bridge_id, op_type, event_ids, status_before, status)

    return updated_count


def delete_alert_event(bridge_id: str, event_id: str, write_audit: bool = True) -> bool:
    events = load_alert_events(bridge_id)
    status_before = None
    new_events = []
    deleted = False
    for e in events:
        if e.id == event_id:
            status_before = e.status
            deleted = True
        else:
            new_events.append(e)
    if deleted:
        save_alert_events(bridge_id, new_events)
        if write_audit:
            add_alert_audit_log(bridge_id, AuditOperationType.DELETE, [event_id], status_before, None)
    return deleted


def batch_delete_alert_events(bridge_id: str, event_ids: List[str]) -> int:
    events = load_alert_events(bridge_id)
    status_before = None
    new_events = []
    deleted_count = 0
    for e in events:
        if e.id in event_ids:
            if status_before is None:
                status_before = e.status
            deleted_count += 1
        else:
            new_events.append(e)
    save_alert_events(bridge_id, new_events)

    if deleted_count > 0:
        add_alert_audit_log(bridge_id, AuditOperationType.BATCH_DELETE, event_ids, status_before, None)

    return deleted_count


def get_audit_logs_in_window(
    bridge_id: str, window_hours: Optional[int] = None
) -> List[AlertAuditLog]:
    logs = load_alert_audit_logs(bridge_id)
    if window_hours is not None:
        cutoff = datetime.now() - timedelta(hours=window_hours)
        logs = [l for l in logs if l.operation_time >= cutoff]
    logs.sort(key=lambda l: l.operation_time, reverse=True)
    return logs
