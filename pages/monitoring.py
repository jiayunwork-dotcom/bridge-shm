import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge import Bridge
from src.models.test_event import TestEvent
from src.models.modal_params import ModalParams, ModeShape
from src.models.alert import Alert, AlertLevel
from src.monitoring.trend_analysis import (
    compute_trend_data,
    create_trend_figure,
    cusum_detection,
    control_chart_limits
)

dash.register_page(__name__, path='/monitoring')

layout = dbc.Container([
    html.H2("长期趋势监控", className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("选择桥梁"),
                dbc.CardBody([
                    html.Label("选择桥梁:"),
                    dcc.Dropdown(id="monitoring-bridge-selector", placeholder="请选择桥梁"),
                    html.Hr(),
                    html.Label("监控指标:"),
                    dcc.RadioItems(
                        id="monitoring-metric",
                        options=[
                            {"label": "频率 (Hz)", "value": "frequency"},
                            {"label": "阻尼比 (%)", "value": "damping"},
                        ],
                        value="frequency",
                        inline=True
                    ),
                    html.Hr(),
                    html.Label("模态阶次:"),
                    dcc.Dropdown(id="monitoring-mode-selector", placeholder="选择模态阶次"),
                    html.Hr(),
                    dbc.Checklist(
                        options=[
                            {"label": "显示控制图", "value": "control_chart"},
                            {"label": "显示CUSUM检测", "value": "cusum"},
                        ],
                        value=["control_chart", "cusum"],
                        id="monitoring-options",
                    ),
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("趋势分析"),
                dbc.CardBody([
                    dcc.Graph(id="trend-plot"),
                ])
            ])
        ], width=9),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("统计信息"),
                dbc.CardBody(id="trend-stats")
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("预警记录"),
                dbc.CardBody([
                    dbc.Checklist(
                        options=[{"label": "只显示未处理", "value": "unack"}],
                        value=["unack"],
                        id="alert-filter",
                        switch=True
                    ),
                    html.Div(id="monitoring-alerts"),
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    dcc.Store(id="trend-data-store"),
    html.Div(id="monitoring-notifications"),
], fluid=True)


@callback(
    Output("monitoring-bridge-selector", "options"),
    Input("bridge-list-refresh", "data"),
    Input("current-bridge-store", "data"),
)
def update_bridge_selector(_, store_data):
    bridges = Bridge.list_all()
    return [{"label": f"{b.name} ({b.id})", "value": b.id} for b in bridges]


@callback(
    Output("monitoring-bridge-selector", "value"),
    Input("current-bridge-store", "data"),
)
def sync_monitoring_bridge_selector(store_data):
    if store_data and store_data.get("id"):
        return store_data["id"]
    return dash.no_update


@callback(
    Output("monitoring-mode-selector", "options"),
    Output("trend-data-store", "data"),
    Input("monitoring-bridge-selector", "value"),
)
def load_trend_data(bridge_id):
    if not bridge_id:
        return [], None
    
    events = TestEvent.list_by_bridge(bridge_id)
    modal_list = []
    valid_events = []
    
    for event in events:
        mp = ModalParams.load(event.id)
        if mp is not None and len(mp.mode_shapes) > 0:
            modal_list.append(mp)
            valid_events.append(event)
    
    if not modal_list:
        return [], None
    
    times, freq_matrix, damp_matrix = compute_trend_data(valid_events, modal_list)
    
    n_modes = freq_matrix.shape[1]
    mode_options = [{"label": f"第{i+1}阶", "value": i} for i in range(n_modes)]
    
    times_iso = [t.isoformat() for t in times]
    
    store_data = {
        "times": times_iso,
        "frequencies": freq_matrix.tolist(),
        "damping": damp_matrix.tolist(),
        "event_ids": [e.id for e in valid_events]
    }
    
    return mode_options, store_data


@callback(
    Output("trend-plot", "figure"),
    Output("trend-stats", "children"),
    Input("monitoring-mode-selector", "value"),
    Input("monitoring-metric", "value"),
    Input("monitoring-options", "value"),
    State("trend-data-store", "data"),
    prevent_initial_call=True,
)
def update_trend_plot(mode_idx, metric, options, trend_data):
    if not trend_data or mode_idx is None:
        return go.Figure(), html.P("请选择模态阶次")
    
    times = [datetime.fromisoformat(t) for t in trend_data["times"]]
    freq_matrix = np.array(trend_data["frequencies"])
    damp_matrix = np.array(trend_data["damping"])
    
    show_control = "control_chart" in options
    show_cusum = "cusum" in options
    
    if metric == "frequency":
        data = freq_matrix
        metric_name = "频率"
        unit = "Hz"
    else:
        data = damp_matrix * 100
        metric_name = "阻尼比"
        unit = "%"
    
    fig = create_trend_figure(
        times, data, mode_idx,
        metric_name=metric_name,
        unit=unit,
        show_control_limits=show_control,
        show_cusum=show_cusum
    )
    
    valid_data = data[:, mode_idx]
    valid_data = valid_data[~np.isnan(valid_data)]
    
    if len(valid_data) > 0:
        center, upper, lower = control_chart_limits(valid_data)
        _, _, anomalies = cusum_detection(valid_data)
        
        stats = html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"{np.mean(valid_data):.4f} {unit}", className="text-center"),
                            html.P("平均值", className="text-center text-muted small")
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"{np.std(valid_data):.4f} {unit}", className="text-center"),
                            html.P("标准差", className="text-center text-muted small")
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"{upper:.4f} {unit}", className="text-center"),
                            html.P("上控制限", className="text-center text-muted small")
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"{lower:.4f} {unit}", className="text-center"),
                            html.P("下控制限", className="text-center text-muted small")
                        ])
                    ])
                ]),
            ]),
            html.Hr(),
            html.P(f"数据点数: {len(valid_data)}"),
            html.P(f"CUSUM检测异常点数: {np.sum(anomalies)}"),
        ])
    else:
        stats = html.P("无有效数据")
    
    return fig, stats


@callback(
    Output("monitoring-alerts", "children"),
    Input("monitoring-bridge-selector", "value"),
    Input("alert-filter", "value"),
    prevent_initial_call=True,
)
def update_alerts(bridge_id, filter_value):
    if not bridge_id:
        return html.P("请选择桥梁")
    
    unack_only = "unack" in filter_value if filter_value else False
    alerts = Alert.load_by_bridge(bridge_id, unacknowledged_only=unack_only)
    
    if not alerts:
        return html.P("无预警记录", className="text-muted")
    
    alert_rows = []
    for alert in alerts[:20]:
        level_color = "danger" if alert.level.value == "red" else "warning" if alert.level.value == "yellow" else "info"
        level_text = "红色预警" if alert.level.value == "red" else "黄色预警" if alert.level.value == "yellow" else "信息"
        
        alert_rows.append(html.Tr([
            html.Td(alert.trigger_time.strftime("%Y-%m-%d %H:%M")),
            html.Td(dbc.Badge(level_text, color=level_color)),
            html.Td(alert.metric),
            html.Td(f"{alert.current_value:.4f}"),
            html.Td(f"{alert.threshold:.4f}"),
            html.Td(alert.suggestion),
            html.Td(
                dbc.Button(
                    "确认", 
                    id={"type": "ack-btn", "index": alert.id},
                    size="sm",
                    color="secondary",
                    disabled=alert.acknowledged
                ) if not alert.acknowledged else html.Span("已确认", className="text-success")
            ),
        ]))
    
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("时间"), html.Th("级别"), html.Th("指标"),
            html.Th("当前值"), html.Th("阈值"), html.Th("建议"), html.Th("操作")
        ])),
        html.Tbody(alert_rows)
    ], bordered=True, hover=True, size="sm")


@callback(
    Output("monitoring-notifications", "children"),
    Input({"type": "ack-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def acknowledge_alert(n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    import json
    alert_id = json.loads(trigger_id)["index"]
    
    from src.models.alert import Alert as AlertModel
    from config import TEST_EVENTS_DIR
    import json as json_lib
    
    alert = None
    
    for bridge in Bridge.list_all():
        for event in TestEvent.list_by_bridge(bridge.id):
            event_dir = os.path.join(TEST_EVENTS_DIR, event.id)
            alerts_file = os.path.join(event_dir, "alerts.json")
            if os.path.exists(alerts_file):
                with open(alerts_file, 'r', encoding='utf-8') as f:
                    alerts = json_lib.load(f)
                
                for a in alerts:
                    if a["id"] == alert_id:
                        a["acknowledged"] = True
                        with open(alerts_file, 'w', encoding='utf-8') as f:
                            json_lib.dump(alerts, f, indent=2, ensure_ascii=False)
                        return dbc.Alert("预警已确认", color="success", duration=3000)
    
    return dbc.Alert("操作失败", color="danger", duration=3000)
