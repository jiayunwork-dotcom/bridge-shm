import dash
from dash import html, dcc, Input, Output, callback, State, ctx
import dash_bootstrap_components as dbc
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
    Input("bridge-list-refresh", "data"),
    Input("current-bridge-store", "data"),
)
def update_bridge_stats(_, store_data):
    bridges = Bridge.list_all()
    n_bridges = len(bridges)
    options = [{"label": b.name, "value": b.id} for b in bridges]
    stats = html.Div([html.H4(f"桥梁总数: {n_bridges}")])
    return stats, options


@callback(
    Output("home-bridge-selector", "value"),
    Input("current-bridge-store", "data"),
)
def sync_home_bridge_selector(store_data):
    if store_data and store_data.get("id"):
        return store_data["id"]
    return dash.no_update


@callback(
    Output("current-bridge-info", "children"),
    Input("current-bridge-store", "data"),
)
def update_current_bridge_info(store_data):
    if not store_data or not store_data.get("id"):
        return html.P("请先选择一座桥梁", className="text-muted")
    
    bridge = Bridge.load(store_data["id"])
    if bridge is None:
        return html.P("桥梁不存在", className="text-danger")
    
    events = TestEvent.list_by_bridge(store_data["id"])
    alerts = Alert.load_by_bridge(store_data["id"], unacknowledged_only=True)
    
    n_sensors = len(bridge.sensors)
    n_events = len(events)
    n_alerts = len(alerts)
    
    sensor_rows = []
    for s in bridge.sensors:
        try:
            loc = s.location if len(s.location) == 3 else (0, 0, 0)
            direction = s.direction if len(s.direction) == 3 else (0, 0, 1)
        except:
            loc = (0, 0, 0)
            direction = (0, 0, 1)
        sensor_rows.append(html.Tr([
            html.Td(s.id), html.Td(s.name), 
            html.Td(s.type.value if hasattr(s.type, 'value') else str(s.type)),
            html.Td(str(s.channel) if s.channel is not None else "-"),
            html.Td(f"({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f})"),
            html.Td(f"({direction[0]:.1f}, {direction[1]:.1f}, {direction[2]:.1f})")
        ]))
    
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
            html.Tbody(sensor_rows)
        ], bordered=True, hover=True, size="sm") if len(bridge.sensors) > 0 else html.P("暂无测点，请前往系统配置添加", className="text-muted")
    ])
    
    return info


@callback(
    Output("recent-alerts", "children"),
    Input("current-bridge-store", "data"),
)
def update_recent_alerts(store_data):
    if not store_data or not store_data.get("id"):
        return html.P("请先选择桥梁", className="text-muted")
    
    alerts = Alert.load_by_bridge(store_data["id"])
    
    if not alerts:
        return html.P("暂无预警记录", className="text-muted")
    
    alert_rows = []
    for alert in alerts[:10]:
        level_val = alert.level.value if hasattr(alert.level, 'value') else str(alert.level)
        level_color = "danger" if level_val == "red" else "warning" if level_val == "yellow" else "info"
        level_text = "红色预警" if level_val == "red" else "黄色预警" if level_val == "yellow" else "信息"
        
        trigger_time_str = ""
        if alert.trigger_time:
            try:
                trigger_time_str = alert.trigger_time.strftime("%Y-%m-%d %H:%M")
            except:
                trigger_time_str = str(alert.trigger_time)
        
        try:
            current_val = f"{float(alert.current_value):.4f}"
        except:
            current_val = str(alert.current_value)
        try:
            threshold_val = f"{float(alert.threshold):.4f}"
        except:
            threshold_val = str(alert.threshold)
        
        alert_rows.append(html.Tr([
            html.Td(trigger_time_str),
            html.Td(dbc.Badge(level_text, color=level_color)),
            html.Td(alert.metric or "-"),
            html.Td(current_val),
            html.Td(threshold_val),
            html.Td(alert.suggestion or "-"),
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
    prevent_initial_call=True,
)
def toggle_create_bridge_modal(n_open, n_close, n_confirm, is_open):
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
    Output("bridge-list-refresh", "data", allow_duplicate=True),
    Output("current-bridge-store", "data", allow_duplicate=True),
    Input("confirm-create-bridge", "n_clicks"),
    State("new-bridge-id", "value"),
    State("new-bridge-name", "value"),
    State("new-bridge-desc", "value"),
    State("bridge-list-refresh", "data"),
    prevent_initial_call=True,
)
def create_bridge(n_clicks, bridge_id, bridge_name, bridge_desc, refresh):
    if not bridge_id or not bridge_name:
        return dbc.Alert("请填写桥梁ID和名称", color="danger"), dash.no_update, dash.no_update
    
    existing = Bridge.load(bridge_id)
    if existing is not None:
        return dbc.Alert("桥梁ID已存在", color="danger"), dash.no_update, dash.no_update
    
    bridge = Bridge(
        id=bridge_id,
        name=bridge_name,
        description=bridge_desc or "",
        sensors=[]
    )
    bridge.save()
    
    store_data = {
        "id": bridge.id,
        "name": bridge.name,
        "baseline_event_id": bridge.baseline_event_id
    }
    
    msg = dbc.Alert(f"桥梁 {bridge_name} 创建成功！已自动切换到该桥梁", color="success", duration=4000)
    
    return msg, (refresh or 0) + 1, store_data
