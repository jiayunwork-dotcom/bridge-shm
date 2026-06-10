import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge import Bridge, SensorType
from src.models.test_event import TestEvent
from src.models.modal_params import ModalParams, ModeShape
from src.models.damage_index import DamageType, DamageIndex
from src.damage_detection.indices import compute_all_damage_indices
from src.temperature_compensation.compensation import (
    build_temperature_model,
    collect_temperature_frequency_pairs,
    compensate_temperature_effect,
    TemperatureModel
)
from src.monitoring.alerting import evaluate_alert_rules

dash.register_page(__name__, path='/damage-detection')

layout = dbc.Container([
    html.H2("损伤检测", className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("选择测试事件"),
                dbc.CardBody([
                    html.Label("选择桥梁:"),
                    dcc.Dropdown(id="damage-bridge-selector", placeholder="请选择桥梁"),
                    html.Hr(),
                    html.Label("选择当前测试事件:"),
                    dcc.Dropdown(id="damage-event-selector", placeholder="请选择测试事件"),
                    html.Hr(),
                    html.Label("选择基准测试事件:"),
                    dcc.Dropdown(id="damage-baseline-selector", placeholder="请选择基准"),
                    html.Hr(),
                    dbc.Checklist(
                        options=[{"label": "启用温度补偿", "value": "temp_comp"}],
                        value=[],
                        id="enable-temp-comp",
                        switch=True
                    ),
                    html.Div(id="temp-model-info", className="mt-2"),
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("损伤指标计算"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[
                                    {"label": "频率变化率", "value": "freq"},
                                    {"label": "柔度矩阵变化", "value": "flex"},
                                    {"label": "曲率模态", "value": "curv"},
                                    {"label": "模态应变能", "value": "mse"},
                                ],
                                value=["freq", "flex", "curv", "mse"],
                                id="damage-indices-options",
                            ),
                        ]),
                        dbc.Col([
                            html.Label("使用模态阶数:"),
                            dbc.Input(id="flexibility-modes", type="number", min=1, max=10, value=5),
                        ]),
                    ]),
                    dbc.Button("计算损伤指标", id="run-damage-btn", color="primary", className="mt-3 w-100"),
                ])
            ])
        ], width=9),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("频率变化率"),
                dbc.CardBody([
                    dcc.Graph(id="freq-change-plot"),
                    html.Div(id="freq-change-table"),
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("柔度矩阵变化"),
                dbc.CardBody([
                    dcc.Graph(id="flex-change-plot"),
                    html.Div(id="flex-change-table"),
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("曲率模态变化"),
                dbc.CardBody([
                    dcc.Graph(id="curvature-plot"),
                    html.Div(id="curvature-table"),
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("模态应变能变化"),
                dbc.CardBody([
                    dcc.Graph(id="mse-plot"),
                    html.Div(id="mse-table"),
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("触发的预警"),
                dbc.CardBody(id="damage-alerts"),
            ])
        ], width=12),
    ], className="mb-4"),
    
    dcc.Store(id="current-modal-store"),
    dcc.Store(id="damage-baseline-modal-store"),
    dcc.Store(id="damage-indices-store"),
    html.Div(id="damage-notifications"),
], fluid=True)


@callback(
    Output("damage-bridge-selector", "options"),
    Input("bridge-list-refresh", "data"),
    Input("current-bridge-store", "data"),
)
def update_bridge_selector(_, store_data):
    bridges = Bridge.list_all()
    return [{"label": b.name, "value": b.id} for b in bridges]


@callback(
    Output("damage-bridge-selector", "value"),
    Input("current-bridge-store", "data"),
)
def sync_damage_bridge_selector(store_data):
    if store_data and store_data.get("id"):
        return store_data["id"]
    return dash.no_update


@callback(
    Output("damage-event-selector", "options"),
    Output("damage-baseline-selector", "options"),
    Input("damage-bridge-selector", "value"),
)
def update_event_selectors(bridge_id):
    if not bridge_id:
        return [], []
    
    events = TestEvent.list_by_bridge(bridge_id)
    options = []
    for e in events:
        try:
            time_str = e.metadata.collection_time.strftime('%Y-%m-%d %H:%M') if e.metadata and hasattr(e.metadata, 'collection_time') else e.id
        except:
            time_str = e.id
        options.append({"label": f"{e.name} ({time_str})", "value": e.id})
    
    return options, options


@callback(
    Output("current-modal-store", "data"),
    Input("damage-event-selector", "value"),
)
def load_current_modal(event_id):
    if not event_id:
        return None
    
    modal_params = ModalParams.load(event_id)
    if modal_params is None:
        return None
    
    return {
        "event_id": event_id,
        "frequencies": modal_params.frequencies.tolist(),
        "singular_values": modal_params.singular_values.tolist(),
        "mode_shapes": [ms.to_dict() for ms in modal_params.mode_shapes]
    }


@callback(
    Output("damage-baseline-modal-store", "data"),
    Output("damage-baseline-selector", "value"),
    Input("damage-bridge-selector", "value"),
    Input("damage-baseline-selector", "value"),
)
def load_baseline_modal(bridge_id, selected_baseline_id):
    if not bridge_id:
        return None, None
    
    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return None, None
    
    baseline_id = selected_baseline_id or bridge.baseline_event_id
    
    if not baseline_id:
        return None, selected_baseline_id
    
    modal_params = ModalParams.load(baseline_id)
    if modal_params is None:
        return None, selected_baseline_id
    
    return {
        "event_id": baseline_id,
        "frequencies": modal_params.frequencies.tolist(),
        "singular_values": modal_params.singular_values.tolist(),
        "mode_shapes": [ms.to_dict() for ms in modal_params.mode_shapes]
    }, baseline_id


@callback(
    Output("temp-model-info", "children"),
    Input("enable-temp-comp", "value"),
    Input("damage-bridge-selector", "value"),
    prevent_initial_call=True,
)
def update_temp_model_info(enabled, bridge_id):
    if not enabled or "temp_comp" not in enabled:
        return html.P("温度补偿已禁用", className="text-muted")
    
    if not bridge_id:
        return html.P("请先选择桥梁", className="text-warning")
    
    events = TestEvent.list_by_bridge(bridge_id)
    
    if len(events) < 20:
        return html.Div([
            html.P(f"数据不足，暂无法建立温度模型（当前{len(events)}组，需要至少20组）", 
                   className="text-warning"),
            html.Small("将使用PCA方法进行环境因子去除", className="text-muted")
        ])
    
    return html.Div([
        html.P(f"温度数据充足（{len(events)}组测试）", className="text-success"),
        html.Small("将使用线性回归模型进行温度补偿", className="text-muted")
    ])


@callback(
    Output("damage-indices-store", "data"),
    Output("damage-notifications", "children"),
    Output("freq-change-plot", "figure"),
    Output("flex-change-plot", "figure"),
    Output("curvature-plot", "figure"),
    Output("mse-plot", "figure"),
    Output("freq-change-table", "children"),
    Output("flex-change-table", "children"),
    Output("curvature-table", "children"),
    Output("mse-table", "children"),
    Output("damage-alerts", "children"),
    Input("run-damage-btn", "n_clicks"),
    State("damage-bridge-selector", "value"),
    State("damage-event-selector", "value"),
    State("damage-baseline-selector", "value"),
    State("current-modal-store", "data"),
    State("damage-baseline-modal-store", "data"),
    State("enable-temp-comp", "value"),
    State("flexibility-modes", "value"),
    prevent_initial_call=True,
)
def run_damage_detection(n_clicks, bridge_id, event_id, baseline_id,
                         current_data, baseline_data, temp_comp_enabled, n_modes):
    if not bridge_id or not event_id or not baseline_id:
        return None, dbc.Alert("请选择桥梁、当前事件和基准事件", color="warning"), \
               go.Figure(), go.Figure(), go.Figure(), go.Figure(), \
               None, None, None, None, None
    
    if not current_data or not baseline_data:
        return None, dbc.Alert("模态数据未加载，请先运行模态分析", color="warning"), \
               go.Figure(), go.Figure(), go.Figure(), go.Figure(), \
               None, None, None, None, None
    
    try:
        bridge = Bridge.load(bridge_id)
        event = TestEvent.load(event_id)
        if bridge is None or event is None:
            return None, dbc.Alert("数据加载失败", color="danger"), \
                   go.Figure(), go.Figure(), go.Figure(), go.Figure(), \
                   None, None, None, None, None
        
        current_params = ModalParams(
            event_id=event_id,
            frequencies=np.array(current_data["frequencies"]),
            singular_values=np.array(current_data["singular_values"]),
            mode_shapes=[ModeShape.from_dict(ms) for ms in current_data["mode_shapes"]]
        )
        
        baseline_params = ModalParams(
            event_id=baseline_id,
            frequencies=np.array(baseline_data["frequencies"]),
            singular_values=np.array(baseline_data["singular_values"]),
            mode_shapes=[ModeShape.from_dict(ms) for ms in baseline_data["mode_shapes"]]
        )
        
        temp_compensated = False
        temp_models = {}
        
        if temp_comp_enabled and "temp_comp" in temp_comp_enabled:
            events = TestEvent.list_by_bridge(bridge_id)
            modal_list = []
            for e in events:
                mp = ModalParams.load(e.id)
                if mp is not None:
                    modal_list.append(mp)
            
            if len(modal_list) >= 20:
                for mode_idx in range(min(5, len(current_params.mode_shapes))):
                    temps, freqs, _ = collect_temperature_frequency_pairs(events, modal_list, mode_idx)
                    if len(temps) >= 20:
                        model, msg = build_temperature_model(temps, freqs, mode_idx, 'linear')
                        if model is not None:
                            temp_models[mode_idx] = model
                
                if event.metadata.temperature is not None:
                    current_params = compensate_temperature_effect(
                        current_params, event.metadata.temperature, temp_models
                    )
                    temp_compensated = True
                    msg = f"温度补偿已启用，使用{len(temp_models)}个模态的回归模型"
                else:
                    msg = "温度数据缺失，温度补偿未执行"
            else:
                msg = "温度数据不足(<20组)，建议补充数据后重新分析"
        else:
            msg = "温度补偿已禁用"
        
        damage_indices = compute_all_damage_indices(
            current_params, baseline_params, bridge,
            temperature_compensated=temp_compensated,
            n_modes=n_modes or 5
        )
        
        for dt, di in damage_indices.items():
            di.save()
        
        alerts = evaluate_alert_rules(bridge, current_params, baseline_params, damage_indices)
        
        fig_freq = go.Figure()
        fig_flex = go.Figure()
        fig_curv = go.Figure()
        fig_mse = go.Figure()
        
        table_freq = None
        table_flex = None
        table_curv = None
        table_mse = None
        
        if DamageType.FREQUENCY_CHANGE in damage_indices:
            di = damage_indices[DamageType.FREQUENCY_CHANGE]
            if len(di.values) > 0:
                x_labels = di.locations or [f"模态{i+1}" for i in range(len(di.values))]
                fig_freq = go.Figure(go.Bar(
                    x=x_labels, y=di.values,
                    marker_color=['red' if abs(v) > abs(di.threshold) else 'blue' for v in di.values]
                ))
                fig_freq.add_hline(y=di.threshold, line_dash="dash", line_color="red", annotation_text="阈值")
                fig_freq.update_layout(title="频率变化率 (%)", yaxis_title="变化率 (%)", height=350)
                
                table_freq = _create_damage_table(di, "频率变化率")
        
        if DamageType.FLEXIBILITY_CHANGE in damage_indices:
            di = damage_indices[DamageType.FLEXIBILITY_CHANGE]
            if len(di.values) > 0:
                x_labels = di.locations or [f"测点{i+1}" for i in range(len(di.values))]
                fig_flex = go.Figure(go.Bar(
                    x=x_labels, y=di.values,
                    marker_color=['red' if abs(v) > abs(di.threshold) else 'blue' for v in di.values]
                ))
                fig_flex.add_hline(y=di.threshold, line_dash="dash", line_color="red", annotation_text="阈值")
                fig_flex.update_layout(title="柔度矩阵变化 (%)", yaxis_title="变化率 (%)", height=350)
                
                table_flex = _create_damage_table(di, "柔度变化")
        
        if DamageType.CURVATURE_MODE in damage_indices:
            di = damage_indices[DamageType.CURVATURE_MODE]
            if len(di.values) > 0:
                x_labels = di.locations or [f"测点{i+1}" for i in range(len(di.values))]
                fig_curv = go.Figure(go.Bar(
                    x=x_labels, y=di.values,
                    marker_color=['red' if di.threshold and abs(v) > abs(di.threshold) else 'blue' for v in di.values]
                ))
                if di.threshold:
                    fig_curv.add_hline(y=di.threshold, line_dash="dash", line_color="red", annotation_text="阈值")
                fig_curv.update_layout(title="曲率模态变化", yaxis_title="曲率变化", height=350)
                
                table_curv = _create_damage_table(di, "曲率模态")
        
        if DamageType.MODAL_STRAIN_ENERGY in damage_indices:
            di = damage_indices[DamageType.MODAL_STRAIN_ENERGY]
            if len(di.values) > 0:
                x_labels = di.locations or [f"测点{i+1}" for i in range(len(di.values))]
                fig_mse = go.Figure(go.Bar(
                    x=x_labels, y=di.values,
                    marker_color=['red' if abs(v) > abs(di.threshold) else 'blue' for v in di.values]
                ))
                fig_mse.add_hline(y=di.threshold, line_dash="dash", line_color="red", annotation_text="阈值")
                fig_mse.update_layout(title="模态应变能变化 (%)", yaxis_title="变化率 (%)", height=350)
                
                table_mse = _create_damage_table(di, "模态应变能")
        
        alert_div = _create_alerts_table(alerts)
        
        store_data = {dt.value: di.to_dict() for dt, di in damage_indices.items()}
        
        notification = dbc.Alert(msg, color="info" if temp_compensated else "warning", duration=5000)
        
        return store_data, notification, fig_freq, fig_flex, fig_curv, fig_mse, \
               table_freq, table_flex, table_curv, table_mse, alert_div
        
    except Exception as e:
        return None, dbc.Alert(f"计算失败: {str(e)}", color="danger"), \
               go.Figure(), go.Figure(), go.Figure(), go.Figure(), \
               None, None, None, None, None


def _create_damage_table(di, title):
    if len(di.values) == 0:
        return html.P("无有效数据")
    
    table_data = [["位置", "指标值", "阈值", "状态"]]
    for i, (loc, val) in enumerate(zip(di.locations or [], di.values)):
        status = "超限" if di.threshold and abs(val) > abs(di.threshold) else "正常"
        status_color = "danger" if status == "超限" else "success"
        table_data.append([
            loc or f"测点{i+1}",
            f"{val:.4f}",
            f"{di.threshold:.4f}" if di.threshold else "-",
            html.Span(status, className=f"badge bg-{status_color}")
        ])
    
    return dbc.Table(table_data, bordered=True, hover=True, size="sm")


def _create_alerts_table(alerts):
    if not alerts:
        return html.P("无预警触发", className="text-muted")
    
    alert_rows = []
    for alert in alerts:
        level_color = "danger" if alert.level.value == "red" else "warning" if alert.level.value == "yellow" else "info"
        level_text = "红色预警" if alert.level.value == "red" else "黄色预警" if alert.level.value == "yellow" else "信息"
        alert_rows.append(html.Tr([
            html.Td(alert.trigger_time.strftime("%Y-%m-%d %H:%M")),
            html.Td(dbc.Badge(level_text, color=level_color)),
            html.Td(alert.metric),
            html.Td(f"{alert.current_value:.4f}"),
            html.Td(f"{alert.threshold:.4f}"),
            html.Td(alert.suggestion),
        ]))
    
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("时间"), html.Th("级别"), html.Th("指标"),
            html.Th("当前值"), html.Th("阈值"), html.Th("建议操作")
        ])),
        html.Tbody(alert_rows)
    ], bordered=True, hover=True, size="sm")
