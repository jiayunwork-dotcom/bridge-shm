import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge import Bridge
from src.models.test_event import TestEvent
from src.models.alert import Alert

dash.register_page(__name__, path='/')

layout = dbc.Container([
    html.H2("系统概览", className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("桥梁管理"),
                dbc.CardBody([
                    html.Div(id="bridge-stats"),
                    html.Hr(),
                    html.Label("选择桥梁:"),
                    dcc.Dropdown(id="home-bridge-selector", placeholder="请选择桥梁"),
                    html.Hr(),
                    dbc.Button("创建新桥", id="open-create-bridge-modal", color="primary", className="w-100"),
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("当前桥梁信息"),
                dbc.CardBody(id="current-bridge-info")
            ])
        ], width=8),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("最新预警"),
                dbc.CardBody(id="recent-alerts")
            ])
        ], width=12),
    ]),
    
    dbc.Modal([
        dbc.ModalHeader("创建新桥梁"),
        dbc.ModalBody([
            dbc.Form([
                dbc.Row([
                    dbc.Label("桥梁ID", width=2),
                    dbc.Col(dbc.Input(id="new-bridge-id", placeholder="bridge_001"), width=10),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("桥梁名称", width=2),
                    dbc.Col(dbc.Input(id="new-bridge-name", placeholder="示例大桥"), width=10),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Label("描述", width=2),
                    dbc.Col(dbc.Textarea(id="new-bridge-desc", rows=3), width=10),
                ], className="mb-3"),
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("取消", id="close-create-bridge-modal", color="secondary"),
            dbc.Button("创建", id="confirm-create-bridge", color="primary"),
        ])
    ], id="create-bridge-modal", is_open=False),
    
    html.Div(id="home-notifications"),
], fluid=True)


@callback(
    Output("bridge-stats", "children"),
    Output("home-bridge-selector", "options"),
    Input("home-bridge-selector", "value"),
)
def update_bridge_stats(selected_bridge_id):
    bridges = Bridge.list_all()
    n_bridges = len(bridges)
    
    options = [{"label": b.name, "value": b.id} for b in bridges]
    
    stats = html.Div([
        html.H4(f"桥梁总数: {n_bridges}"),
    ])
    
    return stats, options


@callback(
    Output("current-bridge-info", "children"),
    Output("home-bridge-selector", "value"),
    Input("home-bridge-selector", "value"),
    Input("current-bridge-store", "data"),
    State("current-bridge-store", "data"),
)
def update_current_bridge(selected_bridge_id, store_data, current_store):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    
    if triggered == "current-bridge-store.data" and store_data:
        bridge_id = store_data.get("id")
        if bridge_id:
            selected_bridge_id = bridge_id
    
    if selected_bridge_id is None:
        return html.P("请先选择一座桥梁"), None
    
    bridge = Bridge.load(selected_bridge_id)
    if bridge is None:
        return html.P("桥梁不存在"), None
    
    events = TestEvent.list_by_bridge(selected_bridge_id)
    alerts = Alert.load_by_bridge(selected_bridge_id, unacknowledged_only=True)
    
    n_sensors = len(bridge.sensors)
    n_events = len(events)
    n_alerts = len(alerts)
    
    info = html.Div([
        html.H3(bridge.name, className="text-primary"),
        html.P(f"桥梁ID: {bridge.id}"),
        html.P(f"描述: {bridge.description}"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(n_sensors, className="text-center"),
                        html.P("测点数量", className="text-center text-muted")
                    ])
                ])
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(n_events, className="text-center"),
                        html.P("测试事件", className="text-center text-muted")
                    ])
                ])
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(n_alerts, className="text-center text-danger"),
                        html.P("未处理预警", className="text-center text-muted")
                    ])
                ])
            ]),
        ]),
        html.Hr(),
        html.H5("测点列表:"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("测点ID"), html.Th("名称"), html.Th("类型"), 
                html.Th("通道"), html.Th("位置"), html.Th("方向")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(s.id), html.Td(s.name), 
                    html.Td(s.type.value), html.Td(s.channel),
                    html.Td(f"({s.location[0]:.1f}, {s.location[1]:.1f}, {s.location[2]:.1f})"),
                    html.Td(f"({s.direction[0]:.1f}, {s.direction[1]:.1f}, {s.direction[2]:.1f})")
                ]) for s in bridge.sensors
            ])
        ], bordered=True, hover=True, size="sm")
    ])
    
    return info, selected_bridge_id


@callback(
    Output("recent-alerts", "children"),
    Input("home-bridge-selector", "value"),
)
def update_recent_alerts(selected_bridge_id):
    if selected_bridge_id is None:
        return html.P("请先选择桥梁")
    
    alerts = Alert.load_by_bridge(selected_bridge_id)
    
    if not alerts:
        return html.P("暂无预警记录", className="text-muted")
    
    alert_rows = []
    for alert in alerts[:10]:
        level_color = "danger" if alert.level.value == "red" else "warning" if alert.level.value == "yellow" else "info"
        level_text = "红色预警" if alert.level.value == "red" else "黄色预警" if alert.level.value == "yellow" else "信息"
        
        alert_rows.append(html.Tr([
            html.Td(alert.trigger_time.strftime("%Y-%m-%d %H:%M")),
            html.Td(dbc.Badge(level_text, color=level_color)),
            html.Td(alert.metric),
            html.Td(f"{alert.current_value:.4f}"),
            html.Td(f"{alert.threshold:.4f}"),
            html.Td(alert.suggestion),
            html.Td("已确认" if alert.acknowledged else "待处理"),
        ]))
    
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("时间"), html.Th("级别"), html.Th("指标"),
            html.Th("当前值"), html.Th("阈值"), html.Th("建议"), html.Th("状态")
        ])),
        html.Tbody(alert_rows)
    ], bordered=True, hover=True, size="sm")


@callback(
    Output("create-bridge-modal", "is_open"),
    Input("open-create-bridge-modal", "n_clicks"),
    Input("close-create-bridge-modal", "n_clicks"),
    Input("confirm-create-bridge", "n_clicks"),
    State("create-bridge-modal", "is_open"),
    State("new-bridge-id", "value"),
    State("new-bridge-name", "value"),
    State("new-bridge-desc", "value"),
)
def toggle_create_bridge_modal(n_open, n_close, n_confirm, is_open, bridge_id, bridge_name, bridge_desc):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "open-create-bridge-modal":
        return True
    elif trigger_id in ["close-create-bridge-modal", "confirm-create-bridge"]:
        return False
    
    return is_open


@callback(
    Output("home-notifications", "children"),
    Input("confirm-create-bridge", "n_clicks"),
    State("new-bridge-id", "value"),
    State("new-bridge-name", "value"),
    State("new-bridge-desc", "value"),
    prevent_initial_call=True,
)
def create_bridge(n_clicks, bridge_id, bridge_name, bridge_desc):
    if not bridge_id or not bridge_name:
        return dbc.Alert("请填写桥梁ID和名称", color="danger")
    
    existing = Bridge.load(bridge_id)
    if existing is not None:
        return dbc.Alert("桥梁ID已存在", color="danger")
    
    bridge = Bridge(
        id=bridge_id,
        name=bridge_name,
        description=bridge_desc or "",
        sensors=[]
    )
    bridge.save()
    
    return dbc.Alert(f"桥梁 {bridge_name} 创建成功！", color="success", duration=3000)
