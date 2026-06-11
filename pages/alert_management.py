import dash
from dash import html, dcc, Input, Output, callback, State, ALL, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge import Bridge
from src.models.test_event import TestEvent
from src.models.anomaly_alert import (
    AlertRule, AnomalyAlertEvent,
    MetricType, ComparisonType, PriorityLevel, AlertStatus,
    METRIC_LABELS, COMPARISON_LABELS,
    PRIORITY_COLORS, PRIORITY_LABELS,
    STATUS_LABELS, STATUS_COLORS,
    load_alert_rules, add_alert_rule, update_alert_rule, delete_alert_rule,
    load_alert_events, update_alert_event_status, get_events_in_window,
    generate_rule_id
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

_detector = AnomalyDetector()
_rules_refresh_counter = 0


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
                        ]),
                    ], className="mb-3"),

                    html.H6("规则列表", className="mt-4 mb-2"),
                    html.Div(id="rule-card-list", children=html.P(
                        "请先选择桥梁", className="text-muted text-center py-4"
                    )),
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
                    ], className="mb-2"),

                    html.H6("告警历史列表", className="mb-2"),
                    html.Div(id="alert-event-list", children=html.P(
                        "暂无告警事件", className="text-muted text-center py-4"
                    )),

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
    Output("rule-linked-event", "value"),
    Output("save-rule-btn", "children"),
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
            1.0, -1.0, 1.0, 5, "medium", None, "保存规则"
        )

    rule_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    if not bridge_id:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    rules = load_alert_rules(bridge_id)
    target = next((r for r in rules if r.id == rule_id), None)
    if not target:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

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
        target.linked_event_id,
        "更新规则"
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
    Output("rule-linked-event", "value", allow_duplicate=True),
    Output("save-rule-btn", "children", allow_duplicate=True),
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
    State("rule-linked-event", "value"),
    prevent_initial_call=True,
)
def save_rule(
    n_clicks, bridge_id, editing_id, name, channels, metric_type,
    comparison, threshold, threshold_min, threshold_max, duration,
    priority, linked_event
):
    if not bridge_id:
        return dbc.Alert("请先选择桥梁", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    if not name or not channels or len(channels) == 0:
        return dbc.Alert("请填写规则名称并选择至少一个传感器通道", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    threshold_value = threshold
    t_min = threshold_min
    t_max = threshold_max
    if comparison == "out_of_range":
        if t_min is None or t_max is None:
            return dbc.Alert("请填写范围的最小和最大值", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        threshold_value = t_max
    else:
        if threshold is None:
            return dbc.Alert("请填写阈值", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    duration_val = max(1, int(duration or 5))

    bridge = Bridge.load(bridge_id)
    valid_channels = [s.channel for s in bridge.sensors] if bridge else []
    filtered_channels = [c for c in channels if c in valid_channels]
    if not filtered_channels:
        return dbc.Alert("所选通道在当前桥梁传感器中不存在", color="warning", duration=3000), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    if editing_id:
        rules = load_alert_rules(bridge_id)
        for r in rules:
            if r.id == editing_id:
                r.name = name
                r.sensor_channels = filtered_channels
                r.metric_type = MetricType(metric_type)
                r.comparison = ComparisonType(comparison)
                r.threshold = threshold_value
                r.threshold_min = t_min if comparison == "out_of_range" else None
                r.threshold_max = t_max if comparison == "out_of_range" else None
                r.duration_seconds = duration_val
                r.priority = PriorityLevel(priority)
                r.linked_event_id = linked_event
                from src.models.anomaly_alert import save_alert_rules
                save_alert_rules(bridge_id, rules)
                break
        msg = dbc.Alert(f"规则 '{name}' 更新成功", color="success", duration=3000)
    else:
        new_rule = AlertRule(
            id=generate_rule_id(),
            bridge_id=bridge_id,
            name=name,
            sensor_channels=filtered_channels,
            metric_type=MetricType(metric_type),
            comparison=ComparisonType(comparison),
            threshold=threshold_value,
            threshold_min=t_min if comparison == "out_of_range" else None,
            threshold_max=t_max if comparison == "out_of_range" else None,
            duration_seconds=duration_val,
            priority=PriorityLevel(priority),
            linked_event_id=linked_event
        )
        add_alert_rule(bridge_id, new_rule)
        msg = dbc.Alert(f"规则 '{name}' 创建成功", color="success", duration=3000)

    rule_cards = _render_rule_cards(bridge_id)
    return (
        msg, rule_cards,
        None, None, None, "rms_amplitude", "greater_than",
        1.0, -1.0, 1.0, 5, "medium", None, "保存规则"
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
        from src.models.anomaly_alert import save_alert_rules
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
        metric_label = METRIC_LABELS.get(r.metric_type.value, r.metric_type.value)
        comp_label = COMPARISON_LABELS.get(r.comparison.value, r.comparison.value)
        if r.comparison.value == "out_of_range":
            threshold_str = f"[{r.threshold_min} ~ {r.threshold_max}]"
        else:
            threshold_str = f"{r.threshold}"
        last_trigger_str = r.last_triggered.strftime("%Y-%m-%d %H:%M:%S") if r.last_triggered else "从未触发"

        cards.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Strong(r.name, className="fs-6"),
                            dbc.Badge(
                                PRIORITY_LABELS.get(r.priority.value, r.priority.value),
                                color=PRIORITY_COLORS.get(r.priority.value, "secondary"),
                                pill=True,
                                className="ms-2"
                            ),
                            dbc.Badge(
                                "启用" if r.enabled else "禁用",
                                color="success" if r.enabled else "secondary",
                                className="ms-1",
                                style={"fontSize": "0.7rem"}
                            ),
                        ], width=12, className="mb-2"),
                    ]),
                    html.Div([
                        html.Span(f"{metric_label} {comp_label} {threshold_str}", className="me-3"),
                        html.Span(f"通道数: {len(r.sensor_channels)}", className="me-3"),
                        html.Span(f"持续: {r.duration_seconds}s", className="me-3"),
                    ], className="small text-muted mb-1"),
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
                size="sm", color="outline-secondary"
            ))
        else:
            action_btns.append(html.Span(
                status_label, className=f"text-{status_color} small"
            ))

        rows.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Strong(e.rule_name),
                            dbc.Badge(
                                prio_label, color=prio_color, pill=True, className="ms-2"
                            ),
                            dbc.Badge(
                                f"CH{e.sensor_channel}", color="light",
                                text_color="dark", className="ms-1"
                            ),
                        ], width=7),
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
                        ], width=7),
                        dbc.Col([
                            html.Div(action_btns, className="d-flex justify-content-end"),
                        ], width=5),
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
    return html.Div(rows, style={"maxHeight": "600px", "overflowY": "auto"})


@callback(
    Output("alert-mgmt-notifications", "children", allow_duplicate=True),
    Output("alert-event-list", "children", allow_duplicate=True),
    Input({"type": "event-ack-btn", "index": ALL}, "n_clicks"),
    Input({"type": "event-ignore-btn", "index": ALL}, "n_clicks"),
    State("alert-mgmt-bridge-selector", "value"),
    State("priority-filter", "value"),
    State("status-filter", "value"),
    prevent_initial_call=True,
)
def handle_event_status(ack_clicks, ignore_clicks, bridge_id, p_filter, s_filter):
    if not bridge_id:
        return no_update, no_update
    triggered = ctx.triggered[0]["prop_id"]
    try:
        obj = json.loads(triggered.split(".")[0])
        event_id = obj["index"]
        action_type = obj["type"]
    except (ValueError, KeyError, json.JSONDecodeError):
        return no_update, no_update

    status = AlertStatus.ACKNOWLEDGED if action_type == "event-ack-btn" else AlertStatus.IGNORED
    update_alert_event_status(bridge_id, event_id, status)
    msg_txt = "告警已确认" if action_type == "event-ack-btn" else "告警已忽略"
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
