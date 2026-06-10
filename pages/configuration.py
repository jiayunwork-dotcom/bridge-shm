import dash
from dash import html, dcc, Input, Output, callback, State, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import sys
import os
import uuid
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge import Bridge, Sensor, SensorType
from src.models.test_event import TestEvent
from src.monitoring.alerting import (
    AlertRule, AlertCondition, AlertLevel,
    load_alert_rules, save_alert_rules
)

dash.register_page(__name__, path='/configuration')

sensor_type_options = [
    {"label": "加速度计", "value": "acceleration"},
    {"label": "应变计", "value": "strain"},
    {"label": "位移计", "value": "displacement"},
    {"label": "温度计", "value": "temperature"},
    {"label": "风速计", "value": "wind_speed"},
]

layout = dbc.Container([
    html.H2("系统配置", className="mb-4"),
    
    dbc.Tabs([
        dbc.Tab(label="桥梁信息配置", tab_id="bridge-tab", children=[
            html.Div(className="mt-4", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("选择桥梁"),
                            dbc.CardBody([
                                dcc.Dropdown(id="config-bridge-selector", placeholder="选择桥梁"),
                            ])
                        ])
                    ], width=12),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("测点管理"),
                            dbc.CardBody([
                                html.H5("添加测点"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("测点ID"),
                                        dbc.Input(id="sensor-id", placeholder="ACC_001"),
                                    ]),
                                    dbc.Col([
                                        html.Label("测点名称"),
                                        dbc.Input(id="sensor-name", placeholder="1号墩顶竖向"),
                                    ]),
                                    dbc.Col([
                                        html.Label("类型"),
                                        dcc.Dropdown(
                                            id="sensor-type",
                                            options=sensor_type_options,
                                            value="acceleration"
                                        ),
                                    ]),
                                ], className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("通道号"),
                                        dbc.Input(id="sensor-channel", type="number", min=1, value=1),
                                    ]),
                                    dbc.Col([
                                        html.Label("X坐标 (m)"),
                                        dbc.Input(id="sensor-x", type="number", step=0.1, value=0),
                                    ]),
                                    dbc.Col([
                                        html.Label("Y坐标 (m)"),
                                        dbc.Input(id="sensor-y", type="number", step=0.1, value=0),
                                    ]),
                                    dbc.Col([
                                        html.Label("Z坐标 (m)"),
                                        dbc.Input(id="sensor-z", type="number", step=0.1, value=0),
                                    ]),
                                ], className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("方向X"),
                                        dbc.Input(id="sensor-dir-x", type="number", step=0.1, value=0),
                                    ]),
                                    dbc.Col([
                                        html.Label("方向Y"),
                                        dbc.Input(id="sensor-dir-y", type="number", step=0.1, value=0),
                                    ]),
                                    dbc.Col([
                                        html.Label("方向Z"),
                                        dbc.Input(id="sensor-dir-z", type="number", step=0.1, value=1),
                                    ]),
                                    dbc.Col([
                                        html.Label("采样率 (Hz)"),
                                        dbc.Input(id="sensor-sr", type="number", step=10, placeholder="可选"),
                                    ]),
                                ], className="mb-3"),
                                dbc.Button("添加测点", id="add-sensor-btn", color="primary", className="w-100"),
                                html.Hr(),
                                html.H5("当前测点列表"),
                                html.Div(id="sensor-list"),
                            ])
                        ])
                    ], width=12),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("基准模型设置"),
                            dbc.CardBody([
                                html.Label("选择基准测试事件:"),
                                dcc.Dropdown(id="baseline-event-selector", placeholder="请选择基准测试"),
                                html.Small("基准测试用于损伤检测的对比参考", className="text-muted"),
                                html.Hr(),
                                dbc.Button("设置为基准", id="set-baseline-btn", color="success", className="w-100"),
                            ])
                        ])
                    ], width=12),
                ]),
            ])
        ]),
        
        dbc.Tab(label="预警规则配置", tab_id="alert-tab", children=[
            html.Div(className="mt-4", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("添加预警规则"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("规则名称"),
                                        dbc.Input(id="alert-rule-name", placeholder="基频下降预警"),
                                    ]),
                                    dbc.Col([
                                        html.Label("预警条件"),
                                        dcc.Dropdown(
                                            id="alert-condition",
                                            options=[
                                                {"label": "频率下降", "value": "frequency_drop"},
                                                {"label": "阻尼增加", "value": "damping_increase"},
                                                {"label": "损伤指标超限", "value": "damage_index_exceed"},
                                            ],
                                            value="frequency_drop"
                                        ),
                                    ]),
                                ], className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("模态阶次"),
                                        dbc.Input(id="alert-mode-index", type="number", min=0, value=0),
                                    ]),
                                    dbc.Col([
                                        html.Label("阈值"),
                                        dbc.Input(id="alert-threshold", type="number", step=0.1, value=-2.0),
                                    ]),
                                    dbc.Col([
                                        html.Label("预警级别"),
                                        dcc.Dropdown(
                                            id="alert-level",
                                            options=[
                                                {"label": "黄色预警", "value": "yellow"},
                                                {"label": "红色预警", "value": "red"},
                                            ],
                                            value="yellow"
                                        ),
                                    ]),
                                ], className="mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("建议操作"),
                                        dbc.Input(id="alert-suggestion", placeholder="建议检查桥梁结构"),
                                    ]),
                                ], className="mb-3"),
                                dbc.Button("添加规则", id="add-alert-rule-btn", color="primary", className="w-100"),
                            ])
                        ])
                    ], width=12),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("当前预警规则"),
                            dbc.CardBody(id="alert-rule-list"),
                        ])
                    ], width=12),
                ]),
            ])
        ]),
    ]),
    
    html.Div(id="config-notifications"),
], fluid=True)


@callback(
    Output("config-bridge-selector", "options"),
    Input("config-bridge-selector", "value"),
)
def update_bridge_selector(_):
    bridges = Bridge.list_all()
    return [{"label": b.name, "value": b.id} for b in bridges]


@callback(
    Output("baseline-event-selector", "options"),
    Output("sensor-list", "children"),
    Output("alert-rule-list", "children"),
    Input("config-bridge-selector", "value"),
    prevent_initial_call=True,
)
def update_bridge_config(bridge_id):
    if not bridge_id:
        return [], html.P("请选择桥梁"), html.P("请选择桥梁")
    
    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return [], html.P("桥梁不存在"), html.P("桥梁不存在")
    
    events = TestEvent.list_by_bridge(bridge_id)
    event_options = [{"label": f"{e.name} ({e.metadata.collection_time.strftime('%Y-%m-%d')})", "value": e.id} for e in events]
    
    if bridge.sensors:
        sensor_rows = []
        for s in bridge.sensors:
            sensor_rows.append(html.Tr([
                html.Td(s.id),
                html.Td(s.name),
                html.Td(s.type.value),
                html.Td(s.channel),
                html.Td(f"({s.location[0]:.1f}, {s.location[1]:.1f}, {s.location[2]:.1f})"),
                html.Td(
                    dbc.Button(
                        "删除", 
                        id={"type": "delete-sensor", "index": s.id},
                        color="danger", size="sm"
                    )
                ),
            ]))
        
        sensor_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("ID"), html.Th("名称"), html.Th("类型"), 
                html.Th("通道"), html.Th("位置"), html.Th("操作")
            ])),
            html.Tbody(sensor_rows)
        ], bordered=True, hover=True, size="sm")
    else:
        sensor_table = html.P("暂无测点，请添加", className="text-muted")
    
    alert_rules = load_alert_rules(bridge_id)
    
    if alert_rules:
        alert_rows = []
        for rule in alert_rules:
            level_color = "danger" if rule.level.value == "red" else "warning"
            level_text = "红色预警" if rule.level.value == "red" else "黄色预警"
            cond_text = {
                "frequency_drop": "频率下降",
                "damping_increase": "阻尼增加",
                "damage_index_exceed": "损伤指标超限"
            }.get(rule.condition.value, rule.condition.value)
            
            alert_rows.append(html.Tr([
                html.Td(rule.name),
                html.Td(cond_text),
                html.Td(f"第{rule.mode_index+1}阶"),
                html.Td(f"{rule.threshold}"),
                html.Td(dbc.Badge(level_text, color=level_color)),
                html.Td(rule.suggestion or "-"),
                html.Td(
                    dbc.Switch(
                        id={"type": "toggle-rule", "index": rule.id},
                        value=rule.enabled,
                        label="启用" if rule.enabled else "禁用"
                    )
                ),
                html.Td(
                    dbc.Button(
                        "删除", 
                        id={"type": "delete-rule", "index": rule.id},
                        color="danger", size="sm"
                    )
                ),
            ]))
        
        alert_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("名称"), html.Th("条件"), html.Th("模态"), 
                html.Th("阈值"), html.Th("级别"), html.Th("建议"),
                html.Th("状态"), html.Th("操作")
            ])),
            html.Tbody(alert_rows)
        ], bordered=True, hover=True, size="sm")
    else:
        alert_table = html.P("暂无预警规则，请添加", className="text-muted")
    
    return event_options, sensor_table, alert_table


@callback(
    Output("config-notifications", "children"),
    Input("add-sensor-btn", "n_clicks"),
    State("config-bridge-selector", "value"),
    State("sensor-id", "value"),
    State("sensor-name", "value"),
    State("sensor-type", "value"),
    State("sensor-channel", "value"),
    State("sensor-x", "value"),
    State("sensor-y", "value"),
    State("sensor-z", "value"),
    State("sensor-dir-x", "value"),
    State("sensor-dir-y", "value"),
    State("sensor-dir-z", "value"),
    State("sensor-sr", "value"),
    prevent_initial_call=True,
)
def add_sensor(n_clicks, bridge_id, sensor_id, sensor_name, sensor_type, channel,
              x, y, z, dx, dy, dz, sr):
    if not bridge_id or not sensor_id or not sensor_name:
        return dbc.Alert("请填写完整的测点信息", color="warning")
    
    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return dbc.Alert("桥梁不存在", color="danger")
    
    for s in bridge.sensors:
        if s.id == sensor_id or s.channel == channel:
            return dbc.Alert("测点ID或通道号已存在", color="danger")
    
    sensor = Sensor(
        id=sensor_id,
        name=sensor_name,
        type=SensorType(sensor_type),
        channel=channel,
        location=(float(x or 0), float(y or 0), float(z or 0)),
        direction=(float(dx or 0), float(dy or 0), float(dz or 0)),
        sampling_rate=sr
    )
    
    bridge.sensors.append(sensor)
    bridge.save()
    
    return dbc.Alert(f"测点 {sensor_name} 添加成功", color="success", duration=3000)


@callback(
    Output("config-notifications", "children", allow_duplicate=True),
    Input({"type": "delete-sensor", "index": dash.ALL}, "n_clicks"),
    State("config-bridge-selector", "value"),
    prevent_initial_call=True,
)
def delete_sensor(n_clicks, bridge_id):
    ctx = dash.callback_context
    if not ctx.triggered or not bridge_id:
        return None
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    import json
    sensor_id = json.loads(trigger_id)["index"]
    
    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return dbc.Alert("桥梁不存在", color="danger")
    
    bridge.sensors = [s for s in bridge.sensors if s.id != sensor_id]
    bridge.save()
    
    return dbc.Alert(f"测点 {sensor_id} 已删除", color="success", duration=3000)


@callback(
    Output("config-notifications", "children", allow_duplicate=True),
    Input("set-baseline-btn", "n_clicks"),
    State("config-bridge-selector", "value"),
    State("baseline-event-selector", "value"),
    prevent_initial_call=True,
)
def set_baseline(n_clicks, bridge_id, event_id):
    if not bridge_id or not event_id:
        return dbc.Alert("请选择桥梁和基准事件", color="warning")
    
    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return dbc.Alert("桥梁不存在", color="danger")
    
    bridge.baseline_event_id = event_id
    bridge.save()
    
    return dbc.Alert("基准测试设置成功", color="success", duration=3000)


@callback(
    Output("config-notifications", "children", allow_duplicate=True),
    Input("add-alert-rule-btn", "n_clicks"),
    State("config-bridge-selector", "value"),
    State("alert-rule-name", "value"),
    State("alert-condition", "value"),
    State("alert-mode-index", "value"),
    State("alert-threshold", "value"),
    State("alert-level", "value"),
    State("alert-suggestion", "value"),
    prevent_initial_call=True,
)
def add_alert_rule(n_clicks, bridge_id, name, condition, mode_idx, threshold, level, suggestion):
    if not bridge_id or not name:
        return dbc.Alert("请填写完整信息", color="warning")
    
    rules = load_alert_rules(bridge_id)
    
    rule = AlertRule(
        id=str(uuid.uuid4())[:8],
        name=name,
        bridge_id=bridge_id,
        condition=AlertCondition(condition),
        mode_index=mode_idx or 0,
        threshold=threshold,
        level=AlertLevel(level),
        suggestion=suggestion or ""
    )
    
    rules.append(rule)
    save_alert_rules(bridge_id, rules)
    
    return dbc.Alert(f"预警规则 '{name}' 添加成功", color="success", duration=3000)


@callback(
    Output("config-notifications", "children", allow_duplicate=True),
    Input({"type": "delete-rule", "index": dash.ALL}, "n_clicks"),
    State("config-bridge-selector", "value"),
    prevent_initial_call=True,
)
def delete_alert_rule(n_clicks, bridge_id):
    ctx = dash.callback_context
    if not ctx.triggered or not bridge_id:
        return None
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    import json
    rule_id = json.loads(trigger_id)["index"]
    
    rules = load_alert_rules(bridge_id)
    rules = [r for r in rules if r.id != rule_id]
    save_alert_rules(bridge_id, rules)
    
    return dbc.Alert("预警规则已删除", color="success", duration=3000)
