import dash
from dash import html, dcc, Input, Output, callback, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge import Bridge, Sensor, SensorType
from src.models.test_event import TestEvent, TestEventMetadata
from src.data_processing.data_import import import_csv, check_file_size, detect_channels
from src.data_processing.preprocessing import preprocess_pipeline

dash.register_page(__name__, path='/data-import')

layout = dbc.Container([
    html.H2("数据导入与预处理", className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("文件导入"),
                dbc.CardBody([
                    html.Label("选择桥梁:"),
                    dcc.Dropdown(id="import-bridge-selector", placeholder="请选择桥梁"),
                    html.Hr(),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            '拖拽CSV文件到此处或 ',
                            html.A('点击选择')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px 0'
                        },
                        multiple=False
                    ),
                    html.Div(id="file-info"),
                    html.Hr(),
                    html.Label("采样频率 (Hz):"),
                    dbc.Input(id="import-sampling-rate", type="number", min=100, max=1000, step=10, placeholder="100-1000"),
                    html.Small("如果CSV包含时间列可自动检测", className="text-muted"),
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("测试事件信息"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Label("测试名称", width=3),
                        dbc.Col(dbc.Input(id="event-name", placeholder="输入测试名称"), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Label("采集时间", width=3),
                        dbc.Col([
                            dbc.Input(
                                id="event-time",
                                type="datetime-local",
                                value=datetime.now().strftime("%Y-%m-%dT%H:%M")
                            ),
                            html.Small("格式: YYYY-MM-DD HH:MM", className="text-muted")
                        ], width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Label("天气", width=3),
                        dbc.Col(dcc.Dropdown(
                            id="event-weather",
                            options=[
                                {"label": "晴", "value": "sunny"},
                                {"label": "多云", "value": "cloudy"},
                                {"label": "阴", "value": "overcast"},
                                {"label": "雨", "value": "rain"},
                                {"label": "雪", "value": "snow"},
                                {"label": "未知", "value": "unknown"},
                            ],
                            value="unknown"
                        ), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Label("温度 (°C)", width=3),
                        dbc.Col(dbc.Input(id="event-temperature", type="number", step=0.1), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Label("风速 (m/s)", width=3),
                        dbc.Col(dbc.Input(id="event-wind", type="number", step=0.1), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Label("交通状态", width=3),
                        dbc.Col(dcc.Dropdown(
                            id="event-traffic",
                            options=[
                                {"label": "正常", "value": "normal"},
                                {"label": "繁忙", "value": "busy"},
                                {"label": "封路", "value": "closed"},
                            ],
                            value="normal"
                        ), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Label("操作人员", width=3),
                        dbc.Col(dbc.Input(id="event-operator"), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Label("备注", width=3),
                        dbc.Col(dbc.Textarea(id="event-notes", rows=2), width=9),
                    ], className="mb-2"),
                ])
            ])
        ], width=8),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("数据裁剪"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("开始时间 (s):"),
                            dbc.Input(id="clip-start", type="number", min=0, step=0.1, value=0),
                        ]),
                        dbc.Col([
                            html.Label("结束时间 (s):"),
                            dbc.Input(id="clip-end", type="number", min=0, step=0.1),
                        ]),
                    ]),
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("预处理选项"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "去趋势", "value": "detrend"}],
                                value=["detrend"],
                                id="preprocess-detrend",
                                switch=True
                            ),
                        ]),
                        dbc.Col([
                            dcc.Dropdown(
                                id="detrend-method",
                                options=[{"label": "线性", "value": "linear"}, {"label": "常数", "value": "constant"}],
                                value="linear",
                                placeholder="去趋势方法"
                            ),
                        ]),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "抗混叠滤波", "value": "filter"}],
                                value=["filter"],
                                id="preprocess-filter",
                                switch=True
                            ),
                        ]),
                        dbc.Col([
                            dbc.Input(
                                id="cutoff-freq", type="number", min=1, step=1,
                                placeholder="截止频率(Hz)"
                            ),
                        ]),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "重采样", "value": "resample"}],
                                value=[],
                                id="preprocess-resample",
                                switch=True
                            ),
                        ]),
                        dbc.Col([
                            dbc.Input(
                                id="target-sr", type="number", min=100, max=1000, step=10,
                                placeholder="目标采样率(Hz)"
                            ),
                        ]),
                    ]),
                ])
            ])
        ], width=8),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Button("预览数据", id="preview-data-btn", color="info", className="me-2"),
            dbc.Button("保存测试事件", id="save-event-btn", color="success"),
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("数据预览"),
                dbc.CardBody([
                    dcc.Graph(id="preview-plot"),
                    html.Hr(),
                    html.Div(id="preview-stats"),
                ])
            ])
        ], width=12),
    ]),
    
    dcc.Store(id="imported-data-store"),
    html.Div(id="import-notifications"),
], fluid=True)


@callback(
    Output("import-bridge-selector", "options"),
    Input("bridge-list-refresh", "data"),
    Input("current-bridge-store", "data"),
)
def update_bridge_selector(_, store_data):
    bridges = Bridge.list_all()
    return [{"label": b.name, "value": b.id} for b in bridges]


@callback(
    Output("import-bridge-selector", "value"),
    Input("current-bridge-store", "data"),
)
def sync_import_bridge_selector(store_data):
    if store_data and store_data.get("id"):
        return store_data["id"]
    return dash.no_update


@callback(
    Output("file-info", "children"),
    Output("imported-data-store", "data"),
    Output("import-sampling-rate", "value"),
    Input("upload-data", "filename"),
    Input("upload-data", "contents"),
    State("import-sampling-rate", "value"),
    prevent_initial_call=True,
)
def handle_file_upload(filename, contents, custom_sr):
    if filename is None:
        return None, None, None
    
    import base64
    import io
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    tmp_path = f"/tmp/{filename}"
    with open(tmp_path, 'wb') as f:
        f.write(decoded)
    
    try:
        valid, msg = check_file_size(tmp_path)
        if not valid:
            return dbc.Alert(msg, color="danger"), None, None
        
        df, detected_sr, channel_names = import_csv(
            tmp_path,
            has_time_column=True,
            custom_sampling_rate=custom_sr
        )
        
        file_size = os.path.getsize(tmp_path) / 1024 / 1024
        n_samples = len(df)
        n_channels = detect_channels(df)
        duration = n_samples / detected_sr
        
        data_dict = {
            "filename": filename,
            "sampling_rate": detected_sr,
            "channel_names": channel_names,
            "data": df.to_dict(orient='records'),
            "duration": duration
        }
        
        info = html.Div([
            html.P(f"文件名: {filename}"),
            html.P(f"文件大小: {file_size:.2f} MB"),
            html.P(f"采样率: {detected_sr:.0f} Hz"),
            html.P(f"通道数: {n_channels}"),
            html.P(f"数据点数: {n_samples}"),
            html.P(f"时长: {duration:.1f} s"),
        ])
        
        return info, data_dict, detected_sr
        
    except Exception as e:
        return dbc.Alert(f"导入失败: {str(e)}", color="danger"), None, custom_sr
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@callback(
    Output("clip-end", "value"),
    Output("clip-end", "max"),
    Input("imported-data-store", "data"),
)
def update_clip_range(data_store):
    if data_store is None:
        return None, 100
    
    duration = data_store.get("duration", 100)
    return duration, duration


@callback(
    Output("preview-plot", "figure"),
    Output("preview-stats", "children"),
    Input("preview-data-btn", "n_clicks"),
    State("imported-data-store", "data"),
    State("clip-start", "value"),
    State("clip-end", "value"),
    State("preprocess-detrend", "value"),
    State("detrend-method", "value"),
    State("preprocess-filter", "value"),
    State("cutoff-freq", "value"),
    State("preprocess-resample", "value"),
    State("target-sr", "value"),
    prevent_initial_call=True,
)
def preview_data(n_clicks, data_store, clip_start, clip_end, detrend, detrend_method,
                 do_filter, cutoff_freq, do_resample, target_sr):
    if data_store is None:
        return go.Figure(), html.P("请先导入数据")
    
    df = pd.DataFrame(data_store["data"])
    sampling_rate = data_store["sampling_rate"]
    
    if clip_start is not None and clip_end is not None and clip_end > clip_start:
        n_start = int(clip_start * sampling_rate)
        n_end = int(clip_end * sampling_rate)
        df = df.iloc[n_start:n_end].reset_index(drop=True)
    
    do_detrend = "detrend" in detrend if detrend else False
    do_filter_flag = "filter" in do_filter if do_filter else False
    do_resample_flag = "resample" in do_resample if do_resample else False
    
    try:
        processed_df, final_sr, params = preprocess_pipeline(
            df, sampling_rate,
            detrend=do_detrend,
            detrend_method=detrend_method or 'linear',
            filter=do_filter_flag,
            cutoff_freq=cutoff_freq,
            resample=do_resample_flag,
            target_sr=target_sr
        )
        
        fig = go.Figure()
        time_vec = np.arange(len(processed_df)) / final_sr
        
        for i, col in enumerate(processed_df.columns[:5]):
            fig.add_trace(go.Scatter(
                x=time_vec, y=processed_df[col].values,
                name=col, mode='lines'
            ))
        
        fig.update_layout(
            title="预处理后数据波形 (前5通道)",
            xaxis_title="时间 (s)",
            yaxis_title="幅值",
            height=400
        )
        
        stats = html.Div([
            html.H6("预处理参数:"),
            html.Ul([
                html.Li(f"去趋势: {'是 (' + detrend_method + ')' if do_detrend else '否'}"),
                html.Li(f"滤波: {'是 (截止=' + str(cutoff_freq) + 'Hz)' if do_filter_flag else '否'}"),
                html.Li(f"重采样: {'是 (目标=' + str(target_sr) + 'Hz)' if do_resample_flag else '否'}"),
                html.Li(f"最终采样率: {final_sr:.0f} Hz"),
                html.Li(f"数据点数: {len(processed_df)}"),
            ])
        ])
        
        return fig, stats
        
    except Exception as e:
        return go.Figure(), dbc.Alert(f"预览失败: {str(e)}", color="danger")


@callback(
    Output("import-notifications", "children"),
    Input("save-event-btn", "n_clicks"),
    State("import-bridge-selector", "value"),
    State("imported-data-store", "data"),
    State("event-name", "value"),
    State("event-time", "value"),
    State("event-weather", "value"),
    State("event-temperature", "value"),
    State("event-wind", "value"),
    State("event-traffic", "value"),
    State("event-operator", "value"),
    State("event-notes", "value"),
    State("clip-start", "value"),
    State("clip-end", "value"),
    State("preprocess-detrend", "value"),
    State("detrend-method", "value"),
    State("preprocess-filter", "value"),
    State("cutoff-freq", "value"),
    State("preprocess-resample", "value"),
    State("target-sr", "value"),
    prevent_initial_call=True,
)
def save_test_event(n_clicks, bridge_id, data_store, event_name, event_time, weather,
                    temperature, wind_speed, traffic, operator, notes,
                    clip_start, clip_end, detrend, detrend_method,
                    do_filter, cutoff_freq, do_resample, target_sr):
    if not bridge_id or not data_store or not event_name:
        return dbc.Alert("请填写完整信息", color="danger")
    
    try:
        df = pd.DataFrame(data_store["data"])
        sampling_rate = data_store["sampling_rate"]
        channel_names = data_store["channel_names"]
        
        if clip_start is not None and clip_end is not None and clip_end > clip_start:
            n_start = int(clip_start * sampling_rate)
            n_end = int(clip_end * sampling_rate)
            df = df.iloc[n_start:n_end].reset_index(drop=True)
        
        do_detrend = "detrend" in detrend if detrend else False
        do_filter_flag = "filter" in do_filter if do_filter else False
        do_resample_flag = "resample" in do_resample if do_resample else False
        
        processed_df, final_sr, preprocess_params = preprocess_pipeline(
            df, sampling_rate,
            detrend=do_detrend,
            detrend_method=detrend_method or 'linear',
            filter=do_filter_flag,
            cutoff_freq=cutoff_freq,
            resample=do_resample_flag,
            target_sr=target_sr
        )
        
        if isinstance(event_time, str):
            if 'T' in event_time:
                event_time = datetime.fromisoformat(event_time)
            else:
                event_time = datetime.fromisoformat(event_time.replace(' ', 'T'))
        
        metadata = TestEventMetadata(
            collection_time=event_time,
            weather=weather or "unknown",
            temperature=temperature,
            wind_speed=wind_speed,
            traffic_status=traffic or "normal",
            operator=operator or "",
            notes=notes or ""
        )
        
        event_id = str(uuid.uuid4())[:8]
        event = TestEvent(
            id=event_id,
            bridge_id=bridge_id,
            name=event_name,
            metadata=metadata,
            sampling_rate=final_sr,
            data=processed_df,
            channel_names=channel_names,
            preprocessing_params=preprocess_params
        )
        
        event.save()
        
        return dbc.Alert(f"测试事件 '{event_name}' 保存成功！ID: {event_id}", color="success", duration=5000)
        
    except Exception as e:
        return dbc.Alert(f"保存失败: {str(e)}", color="danger")
