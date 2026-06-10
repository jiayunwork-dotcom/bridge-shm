import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import sys
import os
from config import WINDOW_FUNCTIONS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge import Bridge
from src.models.test_event import TestEvent
from src.models.modal_params import ModalParams
from src.modal_analysis.fdd import fdd_analysis, get_default_fft_params
from src.visualization.mode_shape import (
    create_mode_shape_animation_2d,
    create_mode_shape_animation_3d,
    create_multi_mode_comparison,
    create_mac_heatmap
)

dash.register_page(__name__, path='/modal-analysis')

layout = dbc.Container([
    html.H2("模态参数识别", className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("选择测试事件"),
                dbc.CardBody([
                    html.Label("选择桥梁:"),
                    dcc.Dropdown(id="modal-bridge-selector", placeholder="请选择桥梁"),
                    html.Hr(),
                    html.Label("选择测试事件:"),
                    dcc.Dropdown(id="modal-event-selector", placeholder="请选择测试事件"),
                    html.Hr(),
                    html.Label("基准测试事件 (用于对比):"),
                    dcc.Dropdown(id="modal-baseline-selector", placeholder="可选"),
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FDD/EFDD参数设置"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("FFT窗长:"),
                            dbc.Input(id="fft-window-length", type="number", placeholder="自动计算"),
                        ]),
                        dbc.Col([
                            html.Label("重叠率:"),
                            dbc.Input(id="fft-overlap", type="number", min=0, max=1, step=0.1, value=0.5),
                        ]),
                        dbc.Col([
                            html.Label("窗函数:"),
                            dcc.Dropdown(
                                id="fft-window-type",
                                options=[{"label": w, "value": w} for w in WINDOW_FUNCTIONS],
                                value="hann"
                            ),
                        ]),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("峰值高度比:"),
                            dbc.Input(id="peak-height-ratio", type="number", min=0.01, max=1, step=0.01, value=0.1),
                        ]),
                        dbc.Col([
                            html.Label("峰间距比:"),
                            dbc.Input(id="peak-distance-ratio", type="number", min=1, max=10, step=0.5, value=3.0),
                        ]),
                        dbc.Col([
                            html.Label("MAC阈值:"),
                            dbc.Input(id="mac-threshold", type="number", min=0, max=1, step=0.05, value=0.8),
                        ]),
                    ]),
                    dbc.Button("运行模态分析", id="run-modal-btn", color="primary", className="mt-3 w-100"),
                ])
            ])
        ], width=9),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("奇异值谱 & 峰值拾取"),
                dbc.CardBody([
                    dcc.Graph(id="singular-values-plot"),
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("识别的模态参数"),
                dbc.CardBody(id="modal-params-table")
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("振型可视化"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("选择模态阶次:"),
                            dcc.Dropdown(id="mode-selector", placeholder="请选择模态"),
                        ]),
                        dbc.Col([
                            html.Label("显示模式:"),
                            dcc.RadioItems(
                                id="view-mode",
                                options=[{"label": "2D", "value": "2d"}, {"label": "3D", "value": "3d"}],
                                value="2d",
                                inline=True
                            ),
                        ]),
                        dbc.Col([
                            html.Label("振型振幅:"),
                            dbc.Input(id="mode-amplitude", type="number", min=0.1, max=10, step=0.1, value=1.0),
                        ]),
                    ]),
                    dcc.Graph(id="mode-shape-plot"),
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("多阶模态对比"),
                dbc.CardBody([
                    dcc.Graph(id="multi-mode-plot"),
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("MAC矩阵 (与基准对比)"),
                dbc.CardBody([
                    dcc.Graph(id="mac-heatmap"),
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    dcc.Store(id="modal-params-store"),
    dcc.Store(id="baseline-modal-store"),
    html.Div(id="modal-notifications"),
], fluid=True)


@callback(
    Output("modal-bridge-selector", "options"),
    Input("bridge-selector-trigger", "data"),
    Input("home-refresh-trigger", "data"),
    Input("config-refresh-trigger", "data"),
)
def update_bridge_selector(_1, _2, _3):
    bridges = Bridge.list_all()
    return [{"label": b.name, "value": b.id} for b in bridges]


@callback(
    Output("modal-bridge-selector", "value"),
    Input("current-bridge-store", "data"),
    State("modal-bridge-selector", "value"),
)
def sync_modal_bridge_selector(store_data, current_value):
    if store_data and store_data.get("id"):
        if store_data["id"] != current_value:
            return store_data["id"]
    return dash.no_update


@callback(
    Output("current-bridge-store", "data", allow_duplicate=True),
    Output("bridge-selector-trigger", "data", allow_duplicate=True),
    Output("home-refresh-trigger", "data", allow_duplicate=True),
    Input("modal-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("bridge-selector-trigger", "data"),
    State("home-refresh-trigger", "data"),
    prevent_initial_call=True,
)
def on_modal_bridge_change(bridge_id, current_store, bridge_trigger, home_trigger):
    if not bridge_id:
        return current_store, bridge_trigger, home_trigger
    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return current_store, bridge_trigger, home_trigger
    store_data = {
        "id": bridge.id,
        "name": bridge.name,
        "baseline_event_id": bridge.baseline_event_id
    }
    return store_data, (bridge_trigger or 0) + 1, (home_trigger or 0) + 1


@callback(
    Output("modal-event-selector", "options"),
    Output("modal-baseline-selector", "options"),
    Input("modal-bridge-selector", "value"),
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
    Output("fft-window-length", "placeholder"),
    Input("modal-event-selector", "value"),
)
def update_fft_placeholder(event_id):
    if not event_id:
        return "自动计算"
    
    event = TestEvent.load(event_id)
    if event is None:
        return "自动计算"
    
    params = get_default_fft_params(len(event.data))
    return f"默认: {params['window_length']}"


@callback(
    Output("modal-params-store", "data"),
    Output("modal-notifications", "children"),
    Output("singular-values-plot", "figure"),
    Output("modal-params-table", "children"),
    Output("mode-selector", "options"),
    Input("run-modal-btn", "n_clicks"),
    State("modal-event-selector", "value"),
    State("fft-window-length", "value"),
    State("fft-overlap", "value"),
    State("fft-window-type", "value"),
    State("peak-height-ratio", "value"),
    State("peak-distance-ratio", "value"),
    State("mac-threshold", "value"),
    prevent_initial_call=True,
)
def run_modal_analysis(n_clicks, event_id, window_length, overlap, window_type,
                       height_ratio, distance_ratio, mac_threshold):
    if not event_id:
        return None, dbc.Alert("请选择测试事件", color="warning"), go.Figure(), None, []
    
    try:
        event = TestEvent.load(event_id)
        if event is None:
            return None, dbc.Alert("测试事件不存在", color="danger"), go.Figure(), None, []
        
        data = event.data.values
        
        fft_params = {
            'window_length': window_length,
            'overlap': overlap or 0.5,
            'window_type': window_type or 'hann',
        }
        if window_length is not None:
            fft_params['nfft'] = window_length
        
        peak_params = {
            'height_ratio': height_ratio or 0.1,
            'distance_ratio': distance_ratio or 3.0
        }
        
        modal_params = fdd_analysis(
            data, event.sampling_rate,
            fft_params=fft_params,
            peak_params=peak_params,
            mac_threshold=mac_threshold or 0.8
        )
        modal_params.event_id = event_id
        modal_params.save()
        
        freq_res = modal_params.frequencies[1] - modal_params.frequencies[0]
        
        fig = go.Figure()
        for i in range(min(3, modal_params.singular_values.shape[1])):
            fig.add_trace(go.Scatter(
                x=modal_params.frequencies,
                y=modal_params.singular_values[:, i],
                name=f'奇异值 {i+1}',
                mode='lines'
            ))
        
        for mode in modal_params.mode_shapes:
            fig.add_vline(
                x=mode.frequency,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{mode.frequency:.2f} Hz",
                annotation_position="top right"
            )
        
        fig.update_layout(
            title=f"奇异值谱 (频率分辨率: {freq_res:.4f} Hz)",
            xaxis_title="频率 (Hz)",
            yaxis_title="奇异值",
            height=400
        )
        
        if modal_params.mode_shapes:
            table_data = [["模态阶次", "频率 (Hz)", "阻尼比 (%)", "阻尼质量", "MAC值"]]
            for i, mode in enumerate(modal_params.mode_shapes):
                quality = "良好" if mode.damping_quality == "good" else "较差"
                quality_color = "success" if mode.damping_quality == "good" else "warning"
                mac = f"{mode.mac_value:.3f}" if mode.mac_value else "-"
                table_data.append([
                    f"第{i+1}阶",
                    f"{mode.frequency:.4f}",
                    f"{mode.damping_ratio * 100:.4f}",
                    html.Span(quality, className=f"badge bg-{quality_color}"),
                    mac
                ])
            
            table = dbc.Table(table_data, bordered=True, hover=True)
        else:
            table = html.P("未识别到有效模态")
        
        mode_options = [{"label": f"第{i+1}阶 ({m.frequency:.3f} Hz)", "value": i} 
                        for i, m in enumerate(modal_params.mode_shapes)]
        
        store_data = {
            "event_id": event_id,
            "frequencies": modal_params.frequencies.tolist(),
            "singular_values": modal_params.singular_values.tolist(),
            "mode_shapes": [ms.to_dict() for ms in modal_params.mode_shapes],
            "params": modal_params.params
        }
        
        return store_data, dbc.Alert(f"成功识别 {len(modal_params.mode_shapes)} 阶模态", color="success"), fig, table, mode_options
        
    except Exception as e:
        return None, dbc.Alert(f"分析失败: {str(e)}", color="danger"), go.Figure(), None, []


@callback(
    Output("mode-shape-plot", "figure"),
    Input("mode-selector", "value"),
    Input("view-mode", "value"),
    Input("mode-amplitude", "value"),
    Input("modal-bridge-selector", "value"),
    State("modal-params-store", "data"),
    prevent_initial_call=True,
)
def update_mode_shape(mode_idx, view_mode, amplitude, bridge_id, modal_data):
    if not modal_data or mode_idx is None or not bridge_id:
        return go.Figure()
    
    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return go.Figure()
    
    from src.models.modal_params import ModeShape, ModalParams
    mode_shapes = [ModeShape.from_dict(ms) for ms in modal_data["mode_shapes"]]
    
    if mode_idx >= len(mode_shapes):
        return go.Figure()
    
    mode_shape = mode_shapes[mode_idx]
    
    if view_mode == "2d":
        fig = create_mode_shape_animation_2d(bridge, mode_shape, amplitude=amplitude or 1.0)
    else:
        fig = create_mode_shape_animation_3d(bridge, mode_shape, amplitude=amplitude or 1.0)
    
    return fig


@callback(
    Output("multi-mode-plot", "figure"),
    Input("modal-params-store", "data"),
    Input("modal-bridge-selector", "value"),
    prevent_initial_call=True,
)
def update_multi_mode(modal_data, bridge_id):
    if not modal_data or not bridge_id:
        return go.Figure()
    
    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return go.Figure()
    
    from src.models.modal_params import ModalParams, ModeShape
    modal_params = ModalParams(
        event_id=modal_data["event_id"],
        frequencies=np.array(modal_data["frequencies"]),
        singular_values=np.array(modal_data["singular_values"]),
        mode_shapes=[ModeShape.from_dict(ms) for ms in modal_data["mode_shapes"]]
    )
    
    fig = create_multi_mode_comparison(bridge, modal_params)
    return fig


@callback(
    Output("baseline-modal-store", "data"),
    Input("modal-baseline-selector", "value"),
    prevent_initial_call=True,
)
def load_baseline_modal(baseline_event_id):
    if not baseline_event_id:
        return None
    
    modal_params = ModalParams.load(baseline_event_id)
    if modal_params is None:
        return None
    
    return {
        "event_id": baseline_event_id,
        "frequencies": modal_params.frequencies.tolist(),
        "singular_values": modal_params.singular_values.tolist(),
        "mode_shapes": [ms.to_dict() for ms in modal_params.mode_shapes],
    }


@callback(
    Output("mac-heatmap", "figure"),
    Input("modal-params-store", "data"),
    Input("baseline-modal-store", "data"),
    State("modal-event-selector", "value"),
    State("modal-baseline-selector", "value"),
    prevent_initial_call=True,
)
def update_mac_heatmap(current_data, baseline_data, current_event_id, baseline_event_id):
    if not current_data:
        return go.Figure()
    
    from src.models.modal_params import ModalParams, ModeShape
    
    current_params = ModalParams(
        event_id=current_data["event_id"],
        frequencies=np.array(current_data["frequencies"]),
        singular_values=np.array(current_data["singular_values"]),
        mode_shapes=[ModeShape.from_dict(ms) for ms in current_data["mode_shapes"]]
    )
    
    if baseline_data:
        baseline_params = ModalParams(
            event_id=baseline_data["event_id"],
            frequencies=np.array(baseline_data["frequencies"]),
            singular_values=np.array(baseline_data["singular_values"]),
            mode_shapes=[ModeShape.from_dict(ms) for ms in baseline_data["mode_shapes"]]
        )
        
        fig = create_mac_heatmap(
            baseline_params, current_params,
            f"基准测试", f"本次测试"
        )
    else:
        fig = create_mac_heatmap(current_params)
    
    return fig
