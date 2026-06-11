import dash
from dash import html, dcc, Input, Output, callback, State, ALL, ctx, no_update, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta
from collections import Counter
import base64
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge import Bridge
from src.models.test_event import TestEvent
from src.models.anomaly_alert import (
    AlertRule, AnomalyAlertEvent, RuleCondition, LogicOperator,
    AlertAuditLog, AuditOperationType,
    MetricType, ComparisonType, PriorityLevel, AlertStatus,
    METRIC_LABELS, COMPARISON_LABELS,
    PRIORITY_COLORS, PRIORITY_LABELS,
    STATUS_LABELS, STATUS_COLORS,
    LOGIC_LABELS, AUDIT_OPERATION_LABELS, AUDIT_OPERATION_COLORS,
    load_alert_rules, add_alert_rule, update_alert_rule, delete_alert_rule, save_alert_rules,
    load_alert_events, update_alert_event_status, get_events_in_window,
    batch_update_alert_event_status, delete_alert_event, batch_delete_alert_events,
    load_alert_audit_logs, get_audit_logs_in_window,
    generate_rule_id, generate_condition_id
)
from src.monitoring.anomaly_detector import (
    AnomalyDetector, get_waveform_around_trigger, get_spectrum_around_trigger,
    DEFAULT_EVAL_INTERVAL
)

dash.register_page(__name__, path='/alert-management')

_metric_options = [
    {"label": v, "value": k} for k, v in METRIC_LABELS.items()
]
_comparison_options = [
    {"label": v, "value": k} for k, v in COMPARISON_LABELS.items()
]
_priority_options = [
    {"label": PRIORITY_LABELS[k], "value": k} for k in ["high", "medium", "low"]
]
_window_options = [
    {"label": "过去24小时", "value": 24},
    {"label": "过去7天", "value": 168},
    {"label": "过去30天", "value": 720},
]
_status_filter_options = [
    {"label": "待处理", "value": "pending"},
    {"label": "已确认", "value": "acknowledged"},
    {"label": "已忽略", "value": "ignored"},
]
_priority_filter_options = [
    {"label": "高优先级", "value": "high"},
    {"label": "中优先级", "value": "medium"},
    {"label": "低优先级", "value": "low"},
]
_logic_options = [
    {"label": "AND (全部满足)", "value": "and"},
    {"label": "OR (任一满足)", "value": "or"},
]
_audit_op_filter_options = [
    {"label": "全部操作", "value": "all"},
    {"label": "确认", "value": "acknowledge"},
    {"label": "忽略", "value": "ignore"},
    {"label": "删除", "value": "delete"},
    {"label": "批量确认", "value": "batch_acknowledge"},
    {"label": "批量忽略", "value": "batch_ignore"},
    {"label": "批量删除", "value": "batch_delete"},
]
_audit_window_options = [
    {"label": "全部日志", "value": 0},
    {"label": "过去24小时", "value": 24},
    {"label": "过去7天", "value": 168},
    {"label": "过去30天", "value": 720},
]

_detector = AnomalyDetector()
_rules_refresh_counter = 0


def _build_condition_card(cond_idx, condition_data=None):
    prefix = f"cond-{cond_idx}"
    default_channels = condition_data.get("sensor_channels", []) if condition_data else []
    default_metric = condition_data.get("metric_type", "rms_amplitude") if condition_data else "rms_amplitude"
    default_comparison = condition_data.get("comparison", "greater_than") if condition_data else "greater_than"
    default_threshold = condition_data.get("threshold", 1.0) if condition_data else 1.0
    default_th_min = condition_data.get("threshold_min", -1.0) if condition_data else -1.0
    default_th_max = condition_data.get("threshold_max", 1.0) if condition_data else 1.0
    default_duration = condition_data.get("duration_seconds", 5) if condition_data else 5

    return dbc.Card([
        dbc.CardHeader([
            html.Strong(f"条件 {cond_idx + 1}"),
            dbc.Button(
                "删除",
                id={"type": "remove-condition-btn", "index": cond_idx},
                color="outline-danger", size="sm", className="float-end"
            ),
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("目标传感器通道*"),
                    dcc.Dropdown(
                        id={"type": f"{prefix}-sensor-channels", "index": cond_idx},
                        multi=True,
                        value=default_channels,
                        placeholder="从桥梁传感器中选择..."
                    ),
                ], width=12, className="mb-2"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("监测指标*"),
                    dcc.Dropdown(
                        id={"type": f"{prefix}-metric-type", "index": cond_idx},
                        options=_metric_options,
                        value=default_metric,
                        clearable=False
                    ),
                ], width=6, className="mb-2"),
                dbc.Col([
                    html.Label("触发条件*"),
                    dcc.Dropdown(
                        id={"type": f"{prefix}-comparison", "index": cond_idx},
                        options=_comparison_options,
                        value=default_comparison,
                        clearable=False
                    ),
                ], width=6, className="mb-2"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("阈值*"),
                    dbc.Input(
                        id={"type": f"{prefix}-threshold", "index": cond_idx},
                        type="number", step=0.001, value=default_threshold
                    ),
                ], width=12, className="mb-2", id={"type": f"{prefix}-single-threshold-col", "index": cond_idx}),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("最小值(下限)*"),
                    dbc.Input(
                        id={"type": f"{prefix}-threshold-min", "index": cond_idx},
                        type="number", step=0.001, value=default_th_min
                    ),
                ], width=6, className="mb-2", id={"type": f"{prefix}-min-threshold-col", "index": cond_idx}, style={"display": "none"}),
                dbc.Col([
                    html.Label("最大值(上限)*"),
                    dbc.Input(
                        id={"type": f"{prefix}-threshold-max", "index": cond_idx},
                        type="number", step=0.001, value=default_th_max
                    ),
                ], width=6, className="mb-2", id={"type": f"{prefix}-max-threshold-col", "index": cond_idx}, style={"display": "none"}),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("持续时长要求(秒)*"),
                    dbc.Input(
                        id={"type": f"{prefix}-duration", "index": cond_idx},
                        type="number", min=1, max=3600, step=1, value=default_duration
                    ),
                    html.Small("连续N秒满足条件才触发", className="text-muted"),
                ], width=12, className="mb-2"),
            ]),
        ]),
    ], className="mb-2", id={"type": "condition-card", "index": cond_idx})


layout = dbc.Container([
    html.H2("传感器异常检测与告警管理", className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("选择桥梁:", className="me-2"),
            dcc.Dropdown(id="alert-mgmt-bridge-selector", placeholder="请选择桥梁"),
        ], width=12, className="mb-3"),
    ]),

    dbc.Row([
        dbc.Col(width=5, children=[
            dbc.Card([
                dbc.CardHeader([
                    html.H5("告警规则配置", className="m-0"),
                ]),
                dbc.CardBody([
                    dbc.Card([
                        dbc.CardHeader(html.Strong("新增/编辑规则")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("规则名称*"),
                                    dbc.Input(id="rule-name", placeholder="例如: 通道1 RMS超限预警"),
                                ], width=12, className="mb-2"),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Switch(
                                        id="rule-composite-mode",
                                        label="复合模式(多条件联合判断)",
                                        value=False,
                                        className="mb-3"
                                    ),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("逻辑关系*"),
                                    dcc.Dropdown(
                                        id="rule-logic-operator",
                                        options=_logic_options,
                                        value="and",
                                        clearable=False
                                    ),
                                ], width=12, className="mb-2", id="logic-operator-col", style={"display": "none"}),
                            ]),

                            html.Div(id="simple-rule-section", children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("目标传感器通道*"),
                                        dcc.Dropdown(
                                            id="rule-sensor-channels",
                                            multi=True,
                                            placeholder="从桥梁传感器中选择..."
                                        ),
                                    ], width=12, className="mb-2"),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("监测指标*"),
                                        dcc.Dropdown(
                                            id="rule-metric-type",
                                            options=_metric_options,
                                            value="rms_amplitude",
                                            clearable=False
                                        ),
                                    ], width=6, className="mb-2"),
                                    dbc.Col([
                                        html.Label("触发条件*"),
                                        dcc.Dropdown(
                                            id="rule-comparison",
                                            options=_comparison_options,
                                            value="greater_than",
                                            clearable=False
                                        ),
                                    ], width=6, className="mb-2"),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("阈值*"),
                                        dbc.Input(id="rule-threshold", type="number", step=0.001, value=1.0),
                                    ], width=12, className="mb-2", id="single-threshold-col"),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("最小值(下限)*"),
                                        dbc.Input(id="rule-threshold-min", type="number", step=0.001, value=-1.0),
                                    ], width=6, className="mb-2", id="min-threshold-col", style={"display": "none"}),
                                    dbc.Col([
                                        html.Label("最大值(上限)*"),
                                        dbc.Input(id="rule-threshold-max", type="number", step=0.001, value=1.0),
                                    ], width=6, className="mb-2", id="max-threshold-col", style={"display": "none"}),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("持续时长要求(秒)*"),
                                        dbc.Input(
                                            id="rule-duration",
                                            type="number", min=1, max=3600, step=1, value=5
                                        ),
                                        html.Small("连续N秒满足条件才触发，防止瞬时噪声误报", className="text-muted"),
                                    ], width=6, className="mb-2"),
                                    dbc.Col([
                                        html.Label("规则优先级*"),
                                        dcc.Dropdown(
                                            id="rule-priority",
                                            options=_priority_options,
                                            value="medium",
                                            clearable=False
                                        ),
                                    ], width=6, className="mb-2"),
                                ]),
                            ]),

                            html.Div(id="composite-rule-section", style={"display": "none"}, children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("规则优先级*"),
                                        dcc.Dropdown(
                                            id="rule-priority-composite",
                                            options=_priority_options,
                                            value="medium",
                                            clearable=False
                                        ),
                                    ], width=6, className="mb-2"),
                                    dbc.Col([
                                        dbc.Button(
                                            "+ 添加条件", id="add-condition-btn",
                                            color="outline-primary", size="sm", className="w-100 mt-4"
                                        ),
                                    ], width=6, className="mb-2"),
                                ]),
                                html.Div(id="condition-cards-container", children=[]),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    html.Label("关联测试事件(可选)"),
                                    dcc.Dropdown(
                                        id="rule-linked-event",
                                        placeholder="选择已有的测试事件..."
                                    ),
                                ], width=12, className="mb-3"),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "保存规则", id="save-rule-btn", color="primary", className="w-100 me-2"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    dbc.Button(
                                        "重置表单", id="reset-rule-btn", color="secondary", className="w-100"
                                    ),
                                ], width=6),
                            ]),
                            dcc.Store(id="editing-rule-id", data=None),
                            dcc.Store(id="condition-counter", data=0),
                            dcc.Store(id="conditions-data-store", data=[]),
                        ]),
                    ], className="mb-3"),

                    html.H6("规则列表", className="mt-4 mb-2"),
                    html.Div(id="rule-card-list", children=html.P(
                        "请先选择桥梁", className="text-muted text-center py-4"
                    )),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "导出全部规则", id="export-rules-btn",
                                color="outline-info", className="w-100"
                            ),
                        ], width=6),
                        dbc.Col([
                            dcc.Upload(
                                id="upload-rules-json",
                                children=dbc.Button(
                                    "导入规则",
                                    color="outline-success", className="w-100"
                                ),
                                accept=".json"
                            ),
                        ], width=6),
                    ]),
                    dcc.Download(id="download-rules-json"),
                ]),
            ]),
        ]),

        dbc.Col(width=7, children=[
            dbc.Card([
                dbc.CardHeader([
                    html.H5("告警事件看板", className="m-0"),
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("统计时间窗口:"),
                            dcc.Dropdown(
                                id="stats-time-window",
                                options=_window_options,
                                value=24,
                                clearable=False,
                                style={"minWidth": "160px"}
                            ),
                        ], width=4, className="mb-3"),
                        dbc.Col([
                            html.Div(
                                [
                                    html.Label("后台评估引擎:"),
                                    dbc.Badge(
                                        "运行中", id="engine-status-badge",
                                        color="success", pill=True, className="ms-2"
                                    ),
                                    html.Span(
                                        id="eval-last-run",
                                        className="ms-3 text-muted small"
                                    ),
                                ],
                                className="d-flex align-items-center pt-2"
                            ),
                        ], width=8, className="mb-3"),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(id="stat-total-count", children="0", className="text-center"),
                                    html.P("告警总数", className="text-center text-muted small m-0"),
                                ]),
                            ]),
                        ], width=2),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dcc.Graph(id="priority-pie-chart", config={"displayModeBar": False}, style={"height": "140px"}),
                                ], className="p-0"),
                            ]),
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dcc.Graph(id="trend-line-chart", config={"displayModeBar": False}, style={"height": "140px"}),
                                ], className="p-0"),
                            ]),
                        ], width=6),
                    ], className="mb-3 g-2"),

                    html.Hr(),

                    dbc.Tabs(id="event-tabs", active_tab="events-tab", children=[
                        dbc.Tab(label="告警历史", tab_id="events-tab", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("按优先级筛选:"),
                                    dbc.Checklist(
                                        id="priority-filter",
                                        options=_priority_filter_options,
                                        value=["high", "medium", "low"],
                                        inline=True,
                                        switch=True,
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("按处理状态筛选:"),
                                    dbc.Checklist(
                                        id="status-filter",
                                        options=_status_filter_options,
                                        value=["pending", "acknowledged", "ignored"],
                                        inline=True,
                                        switch=True,
                                    ),
                                ], width=6),
                            ], className="mb-2 mt-2"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Checklist(
                                        id="select-all-events",
                                        options=[{"label": "全选", "value": "all"}],
                                        value=[],
                                        switch=True,
                                        className="mb-2"
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Div(id="batch-operation-bar", className="d-flex gap-2 mb-2", children=[
                                        dbc.Button(
                                            "批量确认", id="batch-ack-btn",
                                            color="success", size="sm", disabled=True
                                        ),
                                        dbc.Button(
                                            "批量忽略", id="batch-ignore-btn",
                                            color="secondary", size="sm", disabled=True
                                        ),
                                        dbc.Button(
                                            "批量删除", id="batch-delete-btn",
                                            color="danger", size="sm", disabled=True
                                        ),
                                    ]),
                                ], width=9),
                            ]),

                            html.Div(id="alert-event-list", children=html.P(
                                "暂无告警事件", className="text-muted text-center py-4"
                            )),
                        ]),
                        dbc.Tab(label="审计日志", tab_id="audit-tab", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("按操作类型筛选:"),
                                    dcc.Dropdown(
                                        id="audit-op-filter",
                                        options=_audit_op_filter_options,
                                        value="all",
                                        clearable=False,
                                        className="mb-2 mt-2"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("时间范围:"),
                                    dcc.Dropdown(
                                        id="audit-window-filter",
                                        options=_audit_window_options,
                                        value=0,
                                        clearable=False,
                                        className="mb-2 mt-2"
                                    ),
                                ], width=6),
                            ], className="mt-2"),
                            html.Div(id="audit-log-list", children=html.P(
                                "暂无审计日志", className="text-muted text-center py-4"
                            )),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="alert-detail-title")),
        dbc.ModalBody(id="alert-detail-body"),
        dbc.ModalFooter([
            dbc.Button("关闭", id="close-detail-modal", color="secondary"),
        ]),
    ], id="alert-detail-modal", is_open=False, size="xl"),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="batch-confirm-title")),
        dbc.ModalBody(id="batch-confirm-body"),
        dbc.ModalFooter([
            dbc.Button("取消", id="batch-cancel-btn", color="secondary"),
            dbc.Button("确认", id="batch-confirm-btn", color="primary"),
        ]),
    ], id="batch-confirm-modal", is_open=False),
    dcc.Store(id="batch-operation-type", data=None),
    dcc.Store(id="selected-event-ids", data=[]),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="import-result-title")),
        dbc.ModalBody(id="import-result-body"),
        dbc.ModalFooter([
            dbc.Button("关闭", id="close-import-result-btn", color="primary"),
        ]),
    ], id="import-result-modal", is_open=False),

    dcc.Interval(
        id="evaluation-interval",
        interval=DEFAULT_EVAL_INTERVAL * 1000,
        n_intervals=0
    ),
    dcc.Store(id="last-eval-time", data=None),
    dcc.Store(id="bridge-list-refresh", data=1),
    html.Div(id="alert-mgmt-notifications"),
], fluid=True)


@callback(
    Output("alert-mgmt-bridge-selector", "options"),
    Input("bridge-list-refresh", "data"),
    Input("current-bridge-store", "data"),
)
def update_bridge_selector(_, store_data):
    bridges = Bridge.list_all()
    return [{"label": f"{b.name} ({b.id})", "value": b.id} for b in bridges]


@callback(
    Output("alert-mgmt-bridge-selector", "value"),
    Input("current-bridge-store", "data"),
    Input("alert-mgmt-bridge-selector", "options"),
    State("alert-mgmt-bridge-selector", "value"),
)
def sync_bridge_selector(store_data, options, current_value):
    if store_data and store_data.get("id"):
        target_id = store_data["id"]
        if current_value != target_id:
            valid_ids = [o["value"] for o in options] if options else []
            if target_id in valid_ids:
                return target_id
    return current_value if current_value is not None else None


@callback(
    Output("rule-sensor-channels", "options"),
    Output("rule-linked-event", "options"),
    Input("alert-mgmt-bridge-selector", "value"),
)
def load_sensor_and_event_options(bridge_id):
    if not bridge_id:
        return [], []
    bridge = Bridge.load(bridge_id)
    sensor_opts = []
    if bridge:
        sensor_opts = [
            {
                "label": f"CH{s.channel} - {s.name} ({s.type.value})",
                "value": s.channel
            }
            for s in bridge.sensors
        ]
    events = TestEvent.list_by_bridge(bridge_id)
    event_opts = [
        {
            "label": f"{e.name} ({e.metadata.collection_time.strftime('%Y-%m-%d %H:%M')})",
            "value": e.id
        }
        for e in events
    ]
    return sensor_opts, event_opts


@callback(
    Output("simple-rule-section", "style"),
    Output("composite-rule-section", "style"),
    Output("logic-operator-col", "style"),
    Input("rule-composite-mode", "value"),
)
def toggle_composite_mode(is_composite):
    if is_composite:
        return {"display": "none"}, {}, {}
    else:
        return {}, {"display": "none"}, {"display": "none"}


@callback(
    Output("single-threshold-col", "style"),
    Output("min-threshold-col", "style"),
    Output("max-threshold-col", "style"),
    Input("rule-comparison", "value"),
)
def update_threshold_inputs(comparison):
    if comparison == "out_of_range":
        return {"display": "none"}, {}, {}
    else:
        return {}, {"display": "none"}, {"display": "none"}


@callback(
    Output("condition-cards-container", "children"),
    Output("condition-counter", "data"),
    Output("conditions-data-store", "data"),
    Input("add-condition-btn", "n_clicks"),
    Input({"type": "remove-condition-btn", "index": ALL}, "n_clicks"),
    State("condition-cards-container", "children"),
    State("condition-counter", "data"),
    State("conditions-data-store", "data"),
    State("alert-mgmt-bridge-selector", "value"),
    prevent_initial_call=True,
)
def manage_conditions(add_clicks, remove_clicks, existing_cards, counter, stored_conds, bridge_id):
    trigger = ctx.triggered_id
    cards = existing_cards or []
    conds = stored_conds or []

    if trigger == "add-condition-btn":
        if counter >= 5:
            return no_update, no_update, no_update
        new_idx = counter
        new_card = _build_condition_card(new_idx)
        cards.append(new_card)
        conds.append({
            "id": generate_condition_id(),
            "sensor_channels": [],
            "metric_type": "rms_amplitude",
            "comparison": "greater_than",
            "threshold": 1.0,
            "threshold_min": -1.0,
            "threshold_max": 1.0,
            "duration_seconds": 5
        })
        return cards, counter + 1, conds
    else:
        try:
            trigger_obj = json.loads(trigger)
            if trigger_obj.get("type") == "remove-condition-btn":
                remove_idx = trigger_obj["index"]
                if counter <= 1:
                    return no_update, no_update, no_update
                new_cards = []
                new_conds = []
                new_i = 0
                for i, (card, cond) in enumerate(zip(cards, conds)):
                    if i != remove_idx:
                        new_cards.append(card)
                        new_conds.append(cond)
                        new_i += 1
                return new_cards, new_i, new_conds
        except (json.JSONDecodeError, KeyError):
            pass

    return no_update, no_update, no_update


@callback(
    Output("editing-rule-id", "data"),
    Output("rule-name", "value"),
    Output("rule-sensor-channels", "value"),
    Output("rule-metric-type", "value"),
    Output("rule-comparison", "value"),
    Output("rule-threshold", "value"),
    Output("rule-threshold-min", "value"),
    Output("rule-threshold-max", "value"),
    Output("rule-duration", "value"),
    Output("rule-priority", "value"),
    Output("rule-priority-composite", "value"),
    Output("rule-linked-event", "value"),
    Output("save-rule-btn", "children"),
    Output("rule-composite-mode", "value"),
    Output("rule-logic-operator", "value"),
    Output("condition-cards-container", "children"),
    Output("condition-counter", "data"),
    Output("conditions-data-store", "data"),
    Input("reset-rule-btn", "n_clicks"),
    Input({"type": "rule-edit-btn", "index": ALL}, "n_clicks"),
    State("alert-mgmt-bridge-selector", "value"),
    prevent_initial_call=True,
)
def reset_or_edit_rule_form(reset_clicks, edit_clicks_list, bridge_id):
    trigger = ctx.triggered_id
    if trigger == "reset-rule-btn" or not any(edit_clicks_list):
        return (
            None, None, None, "rms_amplitude", "greater_than",
            1.0, -1.0, 1.0, 5, "medium", "medium", None, "保存规则",
            False, "and", [], 0, []
        )

    rule_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    if not bridge_id:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    rules = load_alert_rules(bridge_id)
    target = next((r for r in rules if r.id == rule_id), None)
    if not target:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    cond_cards = []
    cond_data_list = []
    if target.is_composite and target.conditions:
        for i, cond in enumerate(target.conditions):
            cond_cards.append(_build_condition_card(i, {
                "sensor_channels": cond.sensor_channels,
                "metric_type": cond.metric_type.value,
                "comparison": cond.comparison.value,
                "threshold": cond.threshold,
                "threshold_min": cond.threshold_min,
                "threshold_max": cond.threshold_max,
                "duration_seconds": cond.duration_seconds
            }))
            cond_data_list.append({
                "id": cond.id,
                "sensor_channels": cond.sensor_channels,
                "metric_type": cond.metric_type.value,
                "comparison": cond.comparison.value,
                "threshold": cond.threshold,
                "threshold_min": cond.threshold_min,
                "threshold_max": cond.threshold_max,
                "duration_seconds": cond.duration_seconds
            })

    return (
        target.id,
        target.name,
        target.sensor_channels,
        target.metric_type.value,
        target.comparison.value,
        target.threshold,
        target.threshold_min,
        target.threshold_max,
        target.duration_seconds,
        target.priority.value,
        target.priority.value,
        target.linked_event_id,
        "更新规则",
        target.is_composite,
        target.logic_operator.value,
        cond_cards,
        len(cond_cards) if cond_cards else 0,
        cond_data_list
    )


@callback(
    Output("alert-mgmt-notifications", "children", allow_duplicate=True),
    Output("rule-card-list", "children", allow_duplicate=True),
    Output("editing-rule-id", "data", allow_duplicate=True),
    Output("rule-name", "value", allow_duplicate=True),
    Output("rule-sensor-channels", "value", allow_duplicate=True),
    Output("rule-metric-type", "value", allow_duplicate=True),
    Output("rule-comparison", "value", allow_duplicate=True),
    Output("rule-threshold", "value", allow_duplicate=True),
    Output("rule-threshold-min", "value", allow_duplicate=True),
    Output("rule-threshold-max", "value", allow_duplicate=True),
    Output("rule-duration", "value", allow_duplicate=True),
    Output("rule-priority", "value", allow_duplicate=True),
    Output("rule-priority-composite", "value", allow_duplicate=True),
    Output("rule-linked-event", "value", allow_duplicate=True),
    Output("save-rule-btn", "children", allow_duplicate=True),
    Output("rule-composite-mode", "value", allow_duplicate=True),
    Output("rule-logic-operator", "value", allow_duplicate=True),
    Output("condition-cards-container", "children", allow_duplicate=True),
    Output("condition-counter", "data", allow_duplicate=True),
    Output("conditions-data-store", "data", allow_duplicate=True),
    Input("save-rule-btn", "n_clicks"),
    State("alert-mgmt-bridge-selector", "value"),
    State("editing-rule-id", "data"),
    State("rule-name", "value"),
    State("rule-sensor-channels", "value"),
    State("rule-metric-type", "value"),
    State("rule-comparison", "value"),
    State("rule-threshold", "value"),
    State("rule-threshold-min", "value"),
    State("rule-threshold-max", "value"),
    State("rule-duration", "value"),
    State("rule-priority", "value"),
    State("rule-priority-composite", "value"),
    State("rule-linked-event", "value"),
    State("rule-composite-mode", "value"),
    State("rule-logic-operator", "value"),
    State("conditions-data-store", "data"),
    State("condition-cards-container", "children"),
    State({"type": "cond-0-sensor-channels", "index": ALL}, "value"),
    State({"type": "cond-0-metric-type", "index": ALL}, "value"),
    State({"type": "cond-0-comparison", "index": ALL}, "value"),
    State({"type": "cond-0-threshold", "index": ALL}, "value"),
    State({"type": "cond-0-threshold-min", "index": ALL}, "value"),
    State({"type": "cond-0-threshold-max", "index": ALL}, "value"),
    State({"type": "cond-0-duration", "index": ALL}, "value"),
    State({"type": "cond-1-sensor-channels", "index": ALL}, "value"),
    State({"type": "cond-1-metric-type", "index": ALL}, "value"),
    State({"type": "cond-1-comparison", "index": ALL}, "value"),
    State({"type": "cond-1-threshold", "index": ALL}, "value"),
    State({"type": "cond-1-threshold-min", "index": ALL}, "value"),
    State({"type": "cond-1-threshold-max", "index": ALL}, "value"),
    State({"type": "cond-1-duration", "index": ALL}, "value"),
    State({"type": "cond-2-sensor-channels", "index": ALL}, "value"),
    State({"type": "cond-2-metric-type", "index": ALL}, "value"),
    State({"type": "cond-2-comparison", "index": ALL}, "value"),
    State({"type": "cond-2-threshold", "index": ALL}, "value"),
    State({"type": "cond-2-threshold-min", "index": ALL}, "value"),
    State({"type": "cond-2-threshold-max", "index": ALL}, "value"),
    State({"type": "cond-2-duration", "index": ALL}, "value"),
    State({"type": "cond-3-sensor-channels", "index": ALL}, "value"),
    State({"type": "cond-3-metric-type", "index": ALL}, "value"),
    State({"type": "cond-3-comparison", "index": ALL}, "value"),
    State({"type": "cond-3-threshold", "index": ALL}, "value"),
    State({"type": "cond-3-threshold-min", "index": ALL}, "value"),
    State({"type": "cond-3-threshold-max", "index": ALL}, "value"),
    State({"type": "cond-3-duration", "index": ALL}, "value"),
    State({"type": "cond-4-sensor-channels", "index": ALL}, "value"),
    State({"type": "cond-4-metric-type", "index": ALL}, "value"),
    State({"type": "cond-4-comparison", "index": ALL}, "value"),
    State({"type": "cond-4-threshold", "index": ALL}, "value"),
    State({"type": "cond-4-threshold-min", "index": ALL}, "value"),
    State({"type": "cond-4-threshold-max", "index": ALL}, "value"),
    State({"type": "cond-4-duration", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def save_rule(
    n_clicks, bridge_id, editing_id, name, channels, metric_type,
    comparison, threshold, threshold_min, threshold_max, duration,
    priority, priority_composite, linked_event, is_composite, logic_operator,
    stored_conditions, condition_cards,
    *all_condition_vals
):
    if not bridge_id:
        return dbc.Alert("请先选择桥梁", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    if not name:
        return dbc.Alert("请填写规则名称", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    bridge = Bridge.load(bridge_id)
    valid_channels = [s.channel for s in bridge.sensors] if bridge else []
    effective_priority = priority_composite if is_composite else priority

    num_fields_per_condition = 7
    max_conditions = 5
    collected_conditions = []
    if is_composite:
        num_from_store = len(stored_conditions) if stored_conditions else 0
        num_conditions = min(max(num_from_store, 1), max_conditions)

        for i in range(num_conditions):
            base_idx = i * num_fields_per_condition
            cond_channels = None
            cond_metric = None
            cond_comparison = None
            cond_threshold = None
            cond_th_min = None
            cond_th_max = None
            cond_duration = None

            if base_idx + 6 < len(all_condition_vals):
                vals = all_condition_vals[base_idx:base_idx + num_fields_per_condition]
                if vals[0]:
                    cond_channels = vals[0][0] if isinstance(vals[0], list) and len(vals[0]) > 0 else vals[0]
                if vals[1]:
                    cond_metric = vals[1][0] if isinstance(vals[1], list) and len(vals[1]) > 0 else vals[1]
                if vals[2]:
                    cond_comparison = vals[2][0] if isinstance(vals[2], list) and len(vals[2]) > 0 else vals[2]
                if vals[3]:
                    cond_threshold = vals[3][0] if isinstance(vals[3], list) and len(vals[3]) > 0 else vals[3]
                if vals[4]:
                    cond_th_min = vals[4][0] if isinstance(vals[4], list) and len(vals[4]) > 0 else vals[4]
                if vals[5]:
                    cond_th_max = vals[5][0] if isinstance(vals[5], list) and len(vals[5]) > 0 else vals[5]
                if vals[6]:
                    cond_duration = vals[6][0] if isinstance(vals[6], list) and len(vals[6]) > 0 else vals[6]

            base_cond = stored_conditions[i] if stored_conditions and i < len(stored_conditions) else {}

            final_channels = cond_channels if cond_channels is not None else base_cond.get("sensor_channels", [])
            filtered_ch = [c for c in final_channels if c in valid_channels]
            collected_conditions.append({
                "id": base_cond.get("id", generate_condition_id()),
                "sensor_channels": filtered_ch,
                "metric_type": cond_metric if cond_metric is not None else base_cond.get("metric_type", "rms_amplitude"),
                "comparison": cond_comparison if cond_comparison is not None else base_cond.get("comparison", "greater_than"),
                "threshold": cond_threshold if cond_threshold is not None else base_cond.get("threshold", 1.0),
                "threshold_min": cond_th_min if cond_th_min is not None else base_cond.get("threshold_min", -1.0),
                "threshold_max": cond_th_max if cond_th_max is not None else base_cond.get("threshold_max", 1.0),
                "duration_seconds": max(1, int(cond_duration if cond_duration is not None else base_cond.get("duration_seconds", 5)))
            })

        collected_conditions = [c for c in collected_conditions if c.get("sensor_channels")]
        if not collected_conditions:
            return dbc.Alert("复合模式下请添加至少一个条件并选择传感器通道", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    else:
        if not channels or len(channels) == 0:
            return dbc.Alert("请选择至少一个传感器通道", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        threshold_value = threshold
        t_min = threshold_min
        t_max = threshold_max
        if comparison == "out_of_range":
            if t_min is None or t_max is None:
                return dbc.Alert("请填写范围的最小和最大值", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
            threshold_value = t_max
        else:
            if threshold is None:
                return dbc.Alert("请填写阈值", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    duration_val = max(1, int(duration or 5))
    filtered_channels = [c for c in (channels or []) if c in valid_channels]

    if is_composite:
        rule_conditions = []
        for cd in collected_conditions:
            comp = cd["comparison"]
            tc = cd["threshold"]
            tmin = cd["threshold_min"] if comp == "out_of_range" else None
            tmax = cd["threshold_max"] if comp == "out_of_range" else None
            if comp == "out_of_range":
                tc = tmax
            rule_conditions.append(RuleCondition(
                id=cd["id"],
                sensor_channels=cd["sensor_channels"],
                metric_type=MetricType(cd["metric_type"]),
                comparison=ComparisonType(comp),
                threshold=tc,
                threshold_min=tmin,
                threshold_max=tmax,
                duration_seconds=cd["duration_seconds"]
            ))
    else:
        rule_conditions = []

    if editing_id:
        rules = load_alert_rules(bridge_id)
        for r in rules:
            if r.id == editing_id:
                r.name = name
                r.sensor_channels = filtered_channels
                r.metric_type = MetricType(metric_type)
                r.comparison = ComparisonType(comparison)
                if comparison == "out_of_range":
                    r.threshold = threshold_max
                    r.threshold_min = threshold_min
                    r.threshold_max = threshold_max
                else:
                    r.threshold = threshold
                    r.threshold_min = None
                    r.threshold_max = None
                r.duration_seconds = duration_val
                r.priority = PriorityLevel(effective_priority)
                r.linked_event_id = linked_event
                r.is_composite = is_composite
                r.conditions = rule_conditions
                r.logic_operator = LogicOperator(logic_operator)
                save_alert_rules(bridge_id, rules)
                break
        msg = dbc.Alert(f"规则 '{name}' 更新成功", color="success", duration=3000)
    else:
        tc = threshold_max if comparison == "out_of_range" else threshold
        tmin = threshold_min if comparison == "out_of_range" else None
        tmax = threshold_max if comparison == "out_of_range" else None
        new_rule = AlertRule(
            id=generate_rule_id(),
            bridge_id=bridge_id,
            name=name,
            sensor_channels=filtered_channels,
            metric_type=MetricType(metric_type),
            comparison=ComparisonType(comparison),
            threshold=tc,
            threshold_min=tmin,
            threshold_max=tmax,
            duration_seconds=duration_val,
            priority=PriorityLevel(effective_priority),
            linked_event_id=linked_event,
            is_composite=is_composite,
            conditions=rule_conditions,
            logic_operator=LogicOperator(logic_operator)
        )
        add_alert_rule(bridge_id, new_rule)
        msg = dbc.Alert(f"规则 '{name}' 创建成功", color="success", duration=3000)

    rule_cards = _render_rule_cards(bridge_id)
    return (
        msg, rule_cards,
        None, None, None, "rms_amplitude", "greater_than",
        1.0, -1.0, 1.0, 5, "medium", "medium", None, "保存规则",
        False, "and", [], 0, []
    )


@callback(
    Output("alert-mgmt-notifications", "children", allow_duplicate=True),
    Output("rule-card-list", "children", allow_duplicate=True),
    Input({"type": "rule-toggle-btn", "index": ALL}, "value"),
    Input({"type": "rule-delete-btn", "index": ALL}, "n_clicks"),
    State("alert-mgmt-bridge-selector", "value"),
    prevent_initial_call=True,
)
def toggle_or_delete_rule(toggle_vals, delete_clicks, bridge_id):
    if not bridge_id:
        return no_update, no_update
    prop_id_full = ctx.triggered[0]["prop_id"]
    trigger_id_dict = json.loads(prop_id_full.split(".")[0])
    action_type = trigger_id_dict["type"]
    rule_id = trigger_id_dict["index"]

    if action_type == "rule-toggle-btn":
        rules = load_alert_rules(bridge_id)
        for r in rules:
            if r.id == rule_id:
                new_val = ctx.triggered[0]["value"] if ctx.triggered else r.enabled
                r.enabled = bool(new_val)
                break
        save_alert_rules(bridge_id, rules)
        msg = dbc.Alert("规则状态已更新", color="info", duration=2000)
    elif action_type == "rule-delete-btn":
        delete_alert_rule(bridge_id, rule_id)
        msg = dbc.Alert("规则已删除", color="success", duration=2000)
    else:
        return no_update, no_update

    return msg, _render_rule_cards(bridge_id)


def _render_rule_cards(bridge_id):
    if not bridge_id:
        return html.P("请先选择桥梁", className="text-muted text-center py-4")
    rules = load_alert_rules(bridge_id)
    if not rules:
        return html.P("暂无规则，请在上方表单创建", className="text-muted text-center py-4")

    cards = []
    for r in rules:
        priority_label = PRIORITY_LABELS.get(r.priority.value, r.priority.value)
        priority_color = PRIORITY_COLORS.get(r.priority.value, "secondary")
        last_trigger_str = r.last_triggered.strftime("%Y-%m-%d %H:%M:%S") if r.last_triggered else "从未触发"

        if r.is_composite and r.conditions:
            logic_label = LOGIC_LABELS.get(r.logic_operator.value, r.logic_operator.value)
            info_line = [
                html.Span(f"条件数: {len(r.conditions)}", className="me-3"),
                html.Span(f"逻辑: {logic_label}", className="me-3"),
            ]
        else:
            metric_label = METRIC_LABELS.get(r.metric_type.value, r.metric_type.value)
            comp_label = COMPARISON_LABELS.get(r.comparison.value, r.comparison.value)
            if r.comparison.value == "out_of_range":
                threshold_str = f"[{r.threshold_min} ~ {r.threshold_max}]"
            else:
                threshold_str = f"{r.threshold}"
            info_line = [
                html.Span(f"{metric_label} {comp_label} {threshold_str}", className="me-3"),
                html.Span(f"通道数: {len(r.sensor_channels)}", className="me-3"),
                html.Span(f"持续: {r.duration_seconds}s", className="me-3"),
            ]

        badges = [
            dbc.Badge(
                priority_label,
                color=priority_color,
                pill=True,
                className="ms-2"
            ),
            dbc.Badge(
                "启用" if r.enabled else "禁用",
                color="success" if r.enabled else "secondary",
                className="ms-1",
                style={"fontSize": "0.7rem"}
            ),
        ]
        if r.is_composite:
            badges.insert(1, dbc.Badge(
                "复合规则", color="info", className="ms-1", style={"fontSize": "0.7rem"}
            ))

        cards.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Strong(r.name, className="fs-6"),
                            html.Span(badges),
                        ], width=12, className="mb-2"),
                    ]),
                    html.Div(info_line, className="small text-muted mb-1"),
                    html.Div([
                        html.Small(f"上次触发: {last_trigger_str}"),
                    ], className="small text-muted"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Switch(
                                id={"type": "rule-toggle-btn", "index": r.id},
                                label="启用",
                                value=r.enabled,
                                className="mt-2"
                            ),
                        ], width=4),
                        dbc.Col([
                            dbc.Button(
                                "编辑", id={"type": "rule-edit-btn", "index": r.id},
                                color="outline-primary", size="sm", className="w-100 mt-2"
                            ),
                        ], width=4),
                        dbc.Col([
                            dbc.Button(
                                "删除", id={"type": "rule-delete-btn", "index": r.id},
                                color="outline-danger", size="sm", className="w-100 mt-2"
                            ),
                        ], width=4),
                    ]),
                ])
            ], className="mb-2")
        )
    return cards


def _compute_stats_figures(bridge_id, window_hours, priority_filter, status_filter):
    total_count = "0"
    pie_fig = go.Figure()
    trend_fig = go.Figure()
    event_list_html = html.P("请先选择桥梁", className="text-muted text-center py-4")

    if bridge_id:
        event_list_html = _render_event_list(
            bridge_id, priority_filter or ["high", "medium", "low"],
            status_filter or ["pending", "acknowledged", "ignored"]
        )

        stats_events = get_events_in_window(bridge_id, window_hours or 24)
        total_count = str(len(stats_events))

        if stats_events:
            prio_counts = Counter(e.priority.value for e in stats_events)
            pie_labels = [PRIORITY_LABELS.get(k, k) for k in prio_counts.keys()]
            pie_vals = list(prio_counts.values())
            pie_colors = [
                {"high": "#dc3545", "medium": "#ffc107", "low": "#0dcaf0"}.get(k, "#6c757d")
                for k in prio_counts.keys()
            ]
            pie_fig = go.Figure(data=[go.Pie(
                labels=pie_labels, values=pie_vals,
                marker=dict(colors=pie_colors),
                textinfo="label+percent", showlegend=False,
                hole=0.5
            )])
            pie_fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))

            if window_hours <= 24:
                bucket_minutes = 60
                time_format = "%H:%M"
                label = "小时"
            elif window_hours <= 168:
                bucket_minutes = 360
                time_format = "%m-%d"
                label = "天"
            else:
                bucket_minutes = 1440
                time_format = "%m-%d"
                label = "天"

            buckets = {}
            for e in stats_events:
                if bucket_minutes < 1440:
                    bucket_dt = e.trigger_time.replace(
                        minute=(e.trigger_time.minute // bucket_minutes) * bucket_minutes,
                        second=0, microsecond=0
                    )
                else:
                    bucket_dt = e.trigger_time.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                bucket_key = bucket_dt.strftime(time_format)
                buckets[bucket_key] = buckets.get(bucket_key, 0) + 1

            sorted_keys = sorted(buckets.keys())
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(
                x=sorted_keys, y=[buckets[k] for k in sorted_keys],
                mode="lines+markers", name="告警数",
                line=dict(color="#0d6efd", width=2)
            ))
            trend_fig.update_layout(
                margin=dict(l=30, r=5, t=15, b=20),
                xaxis_title=label, yaxis_title="数量",
                showlegend=False, height=140
            )
        else:
            pie_fig = go.Figure()
            pie_fig.update_layout(
                margin=dict(l=5, r=5, t=5, b=5),
                annotations=[dict(text="暂无数据", showarrow=False, font=dict(size=10, color="gray"))]
            )
            trend_fig = go.Figure()
            trend_fig.update_layout(
                margin=dict(l=30, r=5, t=15, b=20), height=140,
                annotations=[dict(text="暂无数据", showarrow=False, font=dict(size=10, color="gray"))]
            )

    rule_cards = _render_rule_cards(bridge_id)
    return rule_cards, event_list_html, total_count, pie_fig, trend_fig


@callback(
    Output("rule-card-list", "children", allow_duplicate=True),
    Output("alert-event-list", "children", allow_duplicate=True),
    Output("stat-total-count", "children", allow_duplicate=True),
    Output("priority-pie-chart", "figure", allow_duplicate=True),
    Output("trend-line-chart", "figure", allow_duplicate=True),
    Input("alert-mgmt-bridge-selector", "value"),
    Input("priority-filter", "value"),
    Input("status-filter", "value"),
    Input("stats-time-window", "value"),
    prevent_initial_call='initial_duplicate',
)
def refresh_stats_and_lists(bridge_id, priority_filter, status_filter, window_hours):
    return _compute_stats_figures(bridge_id, window_hours, priority_filter, status_filter)


@callback(
    Output("last-eval-time", "data"),
    Output("rule-card-list", "children", allow_duplicate=True),
    Output("alert-event-list", "children", allow_duplicate=True),
    Output("stat-total-count", "children"),
    Output("priority-pie-chart", "figure"),
    Output("trend-line-chart", "figure"),
    Output("eval-last-run", "children"),
    Output("engine-status-badge", "children"),
    Input("evaluation-interval", "n_intervals"),
    State("alert-mgmt-bridge-selector", "value"),
    State("priority-filter", "value"),
    State("status-filter", "value"),
    State("stats-time-window", "value"),
    prevent_initial_call=True,
)
def evaluation_loop(
    n_intervals, bridge_id,
    priority_filter_val, status_filter_val, window_hours
):
    global _rules_refresh_counter
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_run_label = f"上次评估: {now_str}"
    engine_label = "运行中"

    try:
        if bridge_id:
            bridge = Bridge.load(bridge_id)
            if bridge:
                _detector.evaluate_all_rules(bridge, DEFAULT_EVAL_INTERVAL)
    except Exception as e:
        print(f"[评估引擎] 出错: {e}")
        engine_label = "异常"

    rules_html, events_html, total_count, pie_fig, trend_fig = _compute_stats_figures(
        bridge_id, window_hours or 24,
        priority_filter_val or ["high", "medium", "low"],
        status_filter_val or ["pending", "acknowledged", "ignored"]
    )

    return (
        now_str, rules_html, events_html,
        total_count, pie_fig, trend_fig, last_run_label, engine_label
    )


@callback(
    Output("alert-event-list", "children", allow_duplicate=True),
    Input("priority-filter", "value"),
    Input("status-filter", "value"),
    Input("alert-mgmt-bridge-selector", "value"),
    prevent_initial_call=True,
)
def filter_event_list(priority_filter_val, status_filter_val, bridge_id):
    if not bridge_id:
        return html.P("请先选择桥梁", className="text-muted text-center py-4")
    return _render_event_list(
        bridge_id,
        priority_filter_val or ["high", "medium", "low"],
        status_filter_val or ["pending", "acknowledged", "ignored"]
    )


def _render_event_list(bridge_id, priority_filter, status_filter):
    events = load_alert_events(bridge_id)
    if not events:
        return html.P("暂无告警事件", className="text-muted text-center py-4")

    events.sort(key=lambda e: e.trigger_time, reverse=True)
    filtered = [
        e for e in events
        if e.priority.value in priority_filter and e.status.value in status_filter
    ]
    if not filtered:
        return html.P("当前筛选条件下无匹配事件", className="text-muted text-center py-4")

    rows = []
    for e in filtered:
        prio_color = PRIORITY_COLORS.get(e.priority.value, "secondary")
        prio_label = PRIORITY_LABELS.get(e.priority.value, e.priority.value)
        status_color = STATUS_COLORS.get(e.status.value, "secondary")
        status_label = STATUS_LABELS.get(e.status.value, e.status.value)
        metric_label = METRIC_LABELS.get(e.metric_type.value, e.metric_type.value)

        action_btns = []
        if e.status.value == "pending":
            action_btns.append(dbc.Button(
                "确认",
                id={"type": "event-ack-btn", "index": e.id},
                size="sm", color="success", className="me-1"
            ))
            action_btns.append(dbc.Button(
                "忽略",
                id={"type": "event-ignore-btn", "index": e.id},
                size="sm", color="outline-secondary", className="me-1"
            ))
            action_btns.append(dbc.Button(
                "删除",
                id={"type": "event-delete-btn", "index": e.id},
                size="sm", color="outline-danger"
            ))
        else:
            action_btns.append(html.Span(
                status_label, className=f"text-{status_color} small me-2"
            ))
            action_btns.append(dbc.Button(
                "删除",
                id={"type": "event-delete-btn", "index": e.id},
                size="sm", color="outline-danger"
            ))

        extra_badges = []
        if e.is_composite:
            extra_badges.append(dbc.Badge("复合", color="info", className="ms-1"))
            if e.skipped_condition_ids:
                extra_badges.append(dbc.Badge(
                    f"跳过{len(e.skipped_condition_ids)}条件",
                    color="warning", className="ms-1", style={"fontSize": "0.65rem"}
                ))

        rows.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Checklist(
                                id={"type": "event-checkbox", "index": e.id},
                                options=[{"label": "", "value": e.id}],
                                value=[],
                                className="mt-1"
                            ),
                        ], width=1),
                        dbc.Col([
                            html.Strong(e.rule_name),
                            dbc.Badge(
                                prio_label, color=prio_color, pill=True, className="ms-2"
                            ),
                            dbc.Badge(
                                f"CH{e.sensor_channel}", color="light",
                                text_color="dark", className="ms-1"
                            ),
                            html.Span(extra_badges),
                        ], width=6),
                        dbc.Col([
                            html.Small(
                                e.trigger_time.strftime("%Y-%m-%d %H:%M:%S"),
                                className="text-muted"
                            ),
                        ], width=5, className="text-end"),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Span(
                                f"{metric_label}: {e.metric_value:.4f}",
                                className="me-3 small"
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Div(action_btns, className="d-flex justify-content-end"),
                        ], width=6),
                    ], className="mt-1"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "查看详情",
                                id={"type": "event-detail-btn", "index": e.id},
                                size="sm", color="link", n_clicks=0,
                                style={"padding": 0}
                            ),
                        ], width=12),
                    ]),
                ])
            ], className="mb-2")
        )
    return html.Div(rows, style={"maxHeight": "500px", "overflowY": "auto"})


@callback(
    Output("selected-event-ids", "data"),
    Output("select-all-events", "value"),
    Output("batch-ack-btn", "disabled"),
    Output("batch-ignore-btn", "disabled"),
    Output("batch-delete-btn", "disabled"),
    Input({"type": "event-checkbox", "index": ALL}, "value"),
    Input("select-all-events", "value"),
    State("alert-mgmt-bridge-selector", "value"),
    State("priority-filter", "value"),
    State("status-filter", "value"),
    State("selected-event-ids", "data"),
    prevent_initial_call=True,
)
def update_selected_events(checkbox_values, select_all_val, bridge_id, p_filter, s_filter, current_selected):
    trigger = ctx.triggered_id
    selected = current_selected or []

    if not bridge_id:
        return [], [], True, True, True

    events = load_alert_events(bridge_id)
    events.sort(key=lambda e: e.trigger_time, reverse=True)
    filtered = [
        e for e in events
        if e.priority.value in (p_filter or ["high", "medium", "low"])
        and e.status.value in (s_filter or ["pending", "acknowledged", "ignored"])
    ]
    filtered_ids = [e.id for e in filtered]

    if trigger == "select-all-events":
        if "all" in select_all_val:
            selected = filtered_ids[:]
            return selected, ["all"], False, False, False
        else:
            return [], [], True, True, True

    if isinstance(trigger, dict):
        try:
            if trigger.get("type") == "event-checkbox":
                event_id = trigger.get("index")
                checkbox_val = ctx.triggered[0].get("value", [])
                if event_id in checkbox_val:
                    if event_id not in selected:
                        selected.append(event_id)
                else:
                    if event_id in selected:
                        selected.remove(event_id)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    elif isinstance(trigger, str) and trigger.startswith("{"):
        try:
            trigger_obj = json.loads(trigger)
            if trigger_obj.get("type") == "event-checkbox":
                event_id = trigger_obj.get("index")
                triggered_val = ctx.triggered[0].get("value", [])
                if event_id in triggered_val:
                    if event_id not in selected:
                        selected.append(event_id)
                else:
                    if event_id in selected:
                        selected.remove(event_id)
        except (json.JSONDecodeError, KeyError):
            pass

    has_selection = len(selected) > 0
    all_selected = len(selected) > 0 and len(selected) == len(filtered_ids)
    select_all_state = ["all"] if all_selected else []
    return selected, select_all_state, not has_selection, not has_selection, not has_selection


@callback(
    Output("batch-confirm-modal", "is_open"),
    Output("batch-confirm-title", "children"),
    Output("batch-confirm-body", "children"),
    Output("batch-operation-type", "data"),
    Input("batch-ack-btn", "n_clicks"),
    Input("batch-ignore-btn", "n_clicks"),
    Input("batch-delete-btn", "n_clicks"),
    Input("batch-cancel-btn", "n_clicks"),
    Input("batch-confirm-btn", "n_clicks"),
    State("selected-event-ids", "data"),
    State("batch-confirm-modal", "is_open"),
    prevent_initial_call=True,
)
def handle_batch_confirm_modal(ack_clicks, ignore_clicks, delete_clicks, cancel_clicks, confirm_clicks, selected_ids, is_open):
    trigger = ctx.triggered_id

    if trigger == "batch-cancel-btn":
        return False, no_update, no_update, no_update

    if trigger == "batch-ack-btn":
        count = len(selected_ids or [])
        return True, "批量确认告警", html.Div([
            html.P(f"即将确认 {count} 条告警事件。"),
            html.P("状态流转: 待处理 → 已确认", className="text-muted"),
        ]), "acknowledge"

    if trigger == "batch-ignore-btn":
        count = len(selected_ids or [])
        return True, "批量忽略告警", html.Div([
            html.P(f"即将忽略 {count} 条告警事件。"),
            html.P("状态流转: 待处理 → 已忽略", className="text-muted"),
        ]), "ignore"

    if trigger == "batch-delete-btn":
        count = len(selected_ids or [])
        return True, "批量删除告警", html.Div([
            html.P(f"即将删除 {count} 条告警事件。"),
            html.P("此操作不可恢复！", className="text-danger"),
        ]), "delete"

    return no_update, no_update, no_update, no_update


@callback(
    Output("alert-mgmt-notifications", "children", allow_duplicate=True),
    Output("alert-event-list", "children", allow_duplicate=True),
    Output("selected-event-ids", "data", allow_duplicate=True),
    Output("select-all-events", "value", allow_duplicate=True),
    Output("batch-ack-btn", "disabled", allow_duplicate=True),
    Output("batch-ignore-btn", "disabled", allow_duplicate=True),
    Output("batch-delete-btn", "disabled", allow_duplicate=True),
    Output("batch-confirm-modal", "is_open", allow_duplicate=True),
    Input("batch-confirm-btn", "n_clicks"),
    State("batch-operation-type", "data"),
    State("selected-event-ids", "data"),
    State("alert-mgmt-bridge-selector", "value"),
    State("priority-filter", "value"),
    State("status-filter", "value"),
    prevent_initial_call=True,
)
def execute_batch_operation(confirm_clicks, op_type, selected_ids, bridge_id, p_filter, s_filter):
    if not confirm_clicks or not bridge_id or not selected_ids:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    count = 0
    msg = ""
    if op_type == "acknowledge":
        count = batch_update_alert_event_status(bridge_id, selected_ids, AlertStatus.ACKNOWLEDGED)
        msg = f"已成功确认 {count} 条告警"
    elif op_type == "ignore":
        count = batch_update_alert_event_status(bridge_id, selected_ids, AlertStatus.IGNORED)
        msg = f"已成功忽略 {count} 条告警"
    elif op_type == "delete":
        count = batch_delete_alert_events(bridge_id, selected_ids)
        msg = f"已成功删除 {count} 条告警"

    event_list = _render_event_list(
        bridge_id,
        p_filter or ["high", "medium", "low"],
        s_filter or ["pending", "acknowledged", "ignored"]
    )
    return dbc.Alert(msg, color="success", duration=3000), event_list, [], [], True, True, True, False


@callback(
    Output("alert-mgmt-notifications", "children", allow_duplicate=True),
    Output("alert-event-list", "children", allow_duplicate=True),
    Input({"type": "event-ack-btn", "index": ALL}, "n_clicks"),
    Input({"type": "event-ignore-btn", "index": ALL}, "n_clicks"),
    Input({"type": "event-delete-btn", "index": ALL}, "n_clicks"),
    State("alert-mgmt-bridge-selector", "value"),
    State("priority-filter", "value"),
    State("status-filter", "value"),
    prevent_initial_call=True,
)
def handle_event_status(ack_clicks, ignore_clicks, delete_clicks, bridge_id, p_filter, s_filter):
    if not bridge_id:
        return no_update, no_update
    triggered = ctx.triggered[0]["prop_id"]
    try:
        obj = json.loads(triggered.split(".")[0])
        event_id = obj["index"]
        action_type = obj["type"]
    except (ValueError, KeyError, json.JSONDecodeError):
        return no_update, no_update

    msg_txt = ""
    if action_type == "event-ack-btn":
        update_alert_event_status(bridge_id, event_id, AlertStatus.ACKNOWLEDGED)
        msg_txt = "告警已确认"
    elif action_type == "event-ignore-btn":
        update_alert_event_status(bridge_id, event_id, AlertStatus.IGNORED)
        msg_txt = "告警已忽略"
    elif action_type == "event-delete-btn":
        delete_alert_event(bridge_id, event_id)
        msg_txt = "告警已删除"

    return (
        dbc.Alert(msg_txt, color="success", duration=2000),
        _render_event_list(
            bridge_id,
            p_filter or ["high", "medium", "low"],
            s_filter or ["pending", "acknowledged", "ignored"]
        )
    )


@callback(
    Output("alert-detail-modal", "is_open"),
    Output("alert-detail-title", "children"),
    Output("alert-detail-body", "children"),
    Input({"type": "event-detail-btn", "index": ALL}, "n_clicks"),
    Input("close-detail-modal", "n_clicks"),
    State("alert-mgmt-bridge-selector", "value"),
    State("alert-detail-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_detail_modal(detail_clicks, close_clicks, bridge_id, is_open):
    trigger = ctx.triggered[0]["prop_id"]
    if close_clicks or trigger.endswith("close-detail-modal.n_clicks"):
        return False, no_update, no_update

    if not any(detail_clicks):
        return no_update, no_update, no_update

    try:
        obj = json.loads(trigger.split(".")[0])
        event_id = obj["index"]
    except (ValueError, KeyError, json.JSONDecodeError):
        return no_update, no_update, no_update

    if not bridge_id:
        return True, "错误", html.P("未选择桥梁", className="text-danger")

    events = load_alert_events(bridge_id)
    event = next((e for e in events if e.id == event_id), None)
    if not event:
        return True, "错误", html.P("告警事件不存在", className="text-danger")

    metric_label = METRIC_LABELS.get(event.metric_type.value, event.metric_type.value)
    prio_color = PRIORITY_COLORS.get(event.priority.value, "secondary")
    prio_label = PRIORITY_LABELS.get(event.priority.value, event.priority.value)
    status_color = STATUS_COLORS.get(event.status.value, "secondary")
    status_label = STATUS_LABELS.get(event.status.value, event.status.value)

    title = f"告警详情 - {event.rule_name}"

    body = []
    body.append(dbc.Row([
        dbc.Col([
            html.Strong("触发时间: "),
            html.Span(event.trigger_time.strftime("%Y-%m-%d %H:%M:%S")),
        ]),
        dbc.Col([
            html.Strong("优先级: "),
            dbc.Badge(prio_label, color=prio_color, pill=True),
        ]),
        dbc.Col([
            html.Strong("状态: "),
            dbc.Badge(status_label, color=status_color),
        ]),
    ], className="mb-3"))

    if event.is_composite:
        composite_info = [
            html.Strong("复合规则触发"),
            html.Br(),
            html.Small(f"触发条件数: {len(event.triggered_condition_ids)}", className="text-muted"),
        ]
        if event.skipped_condition_ids:
            composite_info.extend([
                html.Br(),
                html.Small(f"跳过条件数(数据缺失): {len(event.skipped_condition_ids)}", className="text-warning")
            ])
        body.append(dbc.Row([
            dbc.Col(composite_info, width=12),
        ], className="mb-3"))

    body.append(dbc.Row([
        dbc.Col([
            html.Strong("触发通道: "), html.Span(f"CH{event.sensor_channel}"),
        ]),
        dbc.Col([
            html.Strong("指标类型: "), html.Span(metric_label),
        ]),
        dbc.Col([
            html.Strong("触发数值: "), html.Span(f"{event.metric_value:.6f}"),
        ]),
    ], className="mb-3"))

    waveform_fig = go.Figure()
    spectrum_fig = go.Figure()
    wave_info = ""

    if event.unarchived_file_id:
        try:
            t_axis, wave_data, sr = get_waveform_around_trigger(
                bridge_id, event.unarchived_file_id, event.sensor_channel,
                event.trigger_offset_seconds, 5.0, 5.0
            )
            if wave_data is not None and t_axis is not None:
                waveform_fig.add_trace(go.Scatter(
                    x=t_axis, y=wave_data,
                    mode="lines", name="波形",
                    line=dict(color="#0d6efd", width=1)
                ))
                waveform_fig.add_vline(x=0, line=dict(color="#dc3545", width=2, dash="dash"),
                                       annotation_text="触发点")
                waveform_fig.update_layout(
                    title="触发前后5秒波形", margin=dict(l=50, r=10, t=40, b=40),
                    xaxis_title="时间 (秒)", yaxis_title="幅值",
                    height=280
                )
                wave_info = f"采样率: {sr:.0f} Hz, 数据点数: {len(wave_data)}"
            else:
                waveform_fig.update_layout(
                    annotations=[dict(text="无法加载波形数据", showarrow=False,
                                      font=dict(size=12, color="gray"))],
                    height=280
                )

            freq_axis, freq_data = get_spectrum_around_trigger(
                bridge_id, event.unarchived_file_id, event.sensor_channel,
                event.trigger_offset_seconds, 5.0, 5.0
            )
            if freq_axis is not None and freq_data is not None:
                spectrum_fig.add_trace(go.Bar(
                    x=freq_axis, y=freq_data,
                    name="频谱", marker_color="#20c997", opacity=0.85
                ))
                spectrum_fig.update_layout(
                    title="触发前后10秒频谱分析(FFT)", margin=dict(l=50, r=10, t=40, b=40),
                    xaxis_title="频率 (Hz)", yaxis_title="幅值",
                    height=280
                )
            else:
                spectrum_fig.update_layout(
                    annotations=[dict(text="无法计算频谱", showarrow=False,
                                      font=dict(size=12, color="gray"))],
                    height=280
                )
        except Exception as ex:
            waveform_fig.update_layout(
                annotations=[dict(text=f"加载出错: {ex}", showarrow=False,
                                  font=dict(size=12, color="red"))],
                height=280
            )
    else:
        waveform_fig.update_layout(
            annotations=[dict(text="未关联数据文件", showarrow=False,
                              font=dict(size=12, color="gray"))],
            height=280
        )
        spectrum_fig.update_layout(
            annotations=[dict(text="未关联数据文件", showarrow=False,
                              font=dict(size=12, color="gray"))],
            height=280
        )

    body.append(dbc.Row([
        dbc.Col([
            dcc.Graph(figure=waveform_fig, config={"displayModeBar": True}),
            html.Small(wave_info, className="text-muted"),
        ], width=12, className="mb-3"),
    ]))

    body.append(dbc.Row([
        dbc.Col([
            dcc.Graph(figure=spectrum_fig, config={"displayModeBar": True}),
        ], width=12),
    ]))

    if event.processing_notes:
        body.append(html.Hr())
        body.append(html.Div([
            html.Strong("处理备注: "), html.Span(event.processing_notes)
        ]))

    return True, title, html.Div(body)


def _render_audit_log_list(bridge_id, op_filter, window_hours):
    if not bridge_id:
        return html.P("请先选择桥梁", className="text-muted text-center py-4")

    if window_hours and window_hours > 0:
        logs = get_audit_logs_in_window(bridge_id, window_hours)
    else:
        logs = load_alert_audit_logs(bridge_id)
        logs.sort(key=lambda l: l.operation_time, reverse=True)

    if op_filter and op_filter != "all":
        logs = [l for l in logs if l.operation_type.value == op_filter]

    if not logs:
        return html.P("暂无审计日志", className="text-muted text-center py-4")

    rows = []
    for log in logs:
        op_label = AUDIT_OPERATION_LABELS.get(log.operation_type.value, log.operation_type.value)
        op_color = AUDIT_OPERATION_COLORS.get(log.operation_type.value, "secondary")

        status_flow = ""
        if log.status_before and log.status_after:
            sb_label = STATUS_LABELS.get(log.status_before.value, log.status_before.value)
            sa_label = STATUS_LABELS.get(log.status_after.value, log.status_after.value)
            status_flow = f"{sb_label} → {sa_label}"
        elif log.status_before and not log.status_after:
            sb_label = STATUS_LABELS.get(log.status_before.value, log.status_before.value)
            status_flow = f"{sb_label} → (已删除)"

        rows.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Badge(op_label, color=op_color, pill=True, className="me-2"),
                            html.Small(
                                log.operation_time.strftime("%Y-%m-%d %H:%M:%S"),
                                className="text-muted"
                            ),
                        ], width=8),
                        dbc.Col([
                            html.Small(f"影响 {len(log.event_ids)} 条", className="text-muted float-end"),
                        ], width=4, className="text-end"),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Span(status_flow, className="text-primary small") if status_flow else "",
                        ], width=12, className="mt-1"),
                    ]),
                ])
            ], className="mb-2")
        )
    return html.Div(rows, style={"maxHeight": "500px", "overflowY": "auto"})


@callback(
    Output("audit-log-list", "children"),
    Input("audit-op-filter", "value"),
    Input("audit-window-filter", "value"),
    Input("alert-mgmt-bridge-selector", "value"),
    Input("evaluation-interval", "n_intervals"),
    prevent_initial_call=True,
)
def refresh_audit_logs(op_filter, window_hours, bridge_id, n):
    return _render_audit_log_list(bridge_id, op_filter, window_hours)


@callback(
    Output("download-rules-json", "data"),
    Input("export-rules-btn", "n_clicks"),
    State("alert-mgmt-bridge-selector", "value"),
    prevent_initial_call=True,
)
def export_all_rules(n_clicks, bridge_id):
    if not bridge_id or not n_clicks:
        return no_update

    rules = load_alert_rules(bridge_id)
    export_data = []
    for r in rules:
        rule_dict = r.to_dict()
        rule_dict.pop("last_triggered", None)
        rule_dict.pop("created_at", None)
        if rule_dict.get("conditions"):
            for cond in rule_dict["conditions"]:
                cond.pop("id", None)
        export_data.append(rule_dict)

    filename = f"rules_{bridge_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return dcc.send_bytes(json.dumps(export_data, indent=2, ensure_ascii=False).encode("utf-8"), filename)


@callback(
    Output("alert-mgmt-notifications", "children", allow_duplicate=True),
    Output("rule-card-list", "children", allow_duplicate=True),
    Output("import-result-modal", "is_open"),
    Output("import-result-title", "children"),
    Output("import-result-body", "children"),
    Input("upload-rules-json", "contents"),
    State("upload-rules-json", "filename"),
    State("alert-mgmt-bridge-selector", "value"),
    prevent_initial_call=True,
)
def handle_rules_import(contents, filename, bridge_id):
    if not contents or not bridge_id:
        return no_update, no_update, no_update, no_update, no_update

    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        imported_data = json.loads(decoded.decode("utf-8"))

        if not isinstance(imported_data, list):
            return dbc.Alert("文件格式错误：应为JSON数组", color="danger", duration=3000), no_update, True, "导入失败", html.P("JSON结构应为规则数组", className="text-danger")

        existing_rules = load_alert_rules(bridge_id)
        existing_names = {r.name for r in existing_rules}

        success_count = 0
        imported_names = []
        errors = []

        for idx, item in enumerate(imported_data):
            try:
                name = item.get("name", f"导入规则_{idx+1}")
                original_name = name
                suffix_counter = 1
                while name in existing_names or name in imported_names:
                    name = f"{original_name}(导入)"
                    if suffix_counter > 1:
                        name = f"{original_name}(导入{suffix_counter})"
                    suffix_counter += 1
                imported_names.append(name)

                is_composite = item.get("is_composite", False)
                conditions = []
                if is_composite and item.get("conditions"):
                    for cd in item["conditions"]:
                        conditions.append(RuleCondition(
                            id=generate_condition_id(),
                            sensor_channels=cd.get("sensor_channels", []),
                            metric_type=MetricType(cd.get("metric_type", "rms_amplitude")),
                            comparison=ComparisonType(cd.get("comparison", "greater_than")),
                            threshold=float(cd.get("threshold", 0.0)),
                            threshold_min=cd.get("threshold_min"),
                            threshold_max=cd.get("threshold_max"),
                            duration_seconds=int(cd.get("duration_seconds", 5))
                        ))

                new_rule = AlertRule(
                    id=generate_rule_id(),
                    bridge_id=bridge_id,
                    name=name,
                    sensor_channels=item.get("sensor_channels", []),
                    metric_type=MetricType(item.get("metric_type", "rms_amplitude")),
                    comparison=ComparisonType(item.get("comparison", "greater_than")),
                    threshold=float(item.get("threshold", 0.0)),
                    threshold_min=item.get("threshold_min"),
                    threshold_max=item.get("threshold_max"),
                    duration_seconds=int(item.get("duration_seconds", 5)),
                    priority=PriorityLevel(item.get("priority", "medium")),
                    linked_event_id=item.get("linked_event_id"),
                    enabled=item.get("enabled", True),
                    is_composite=is_composite,
                    conditions=conditions,
                    logic_operator=LogicOperator(item.get("logic_operator", "and"))
                )
                existing_rules.append(new_rule)
                success_count += 1
            except Exception as e:
                errors.append(f"第{idx+1}条规则: {str(e)}")

        save_alert_rules(bridge_id, existing_rules)
        rule_cards = _render_rule_cards(bridge_id)

        body_parts = [html.P(f"成功导入 {success_count} 条规则")]
        if errors:
            body_parts.append(html.P("以下规则导入失败:", className="text-warning mt-2"))
            for err in errors:
                body_parts.append(html.Li(err, className="text-danger small"))

        return (
            dbc.Alert(f"成功导入 {success_count} 条规则", color="success", duration=3000),
            rule_cards,
            True,
            "导入结果",
            html.Div(body_parts)
        )

    except Exception as e:
        return dbc.Alert(f"导入失败: {str(e)}", color="danger", duration=3000), no_update, True, "导入失败", html.P(str(e), className="text-danger")


@callback(
    Output("import-result-modal", "is_open", allow_duplicate=True),
    Input("close-import-result-btn", "n_clicks"),
    prevent_initial_call=True,
)
def close_import_result(n_clicks):
    return False