import dash
from dash import html, dcc, Input, Output, callback, State, ctx, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os
import uuid
import base64
import io
import re
from datetime import datetime
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge import Bridge, Sensor, SensorType
from src.models.test_event import TestEvent, TestEventMetadata
from src.models.preprocess_preset import PreprocessPreset
from src.models.unarchived_file import UnarchivedFile
from src.data_processing.data_import import import_csv, check_file_size, detect_channels
from src.data_processing.preprocessing import preprocess_pipeline
from config import TEST_EVENTS_DIR

dash.register_page(__name__, path='/data-import')


def extract_timestamp_from_filename(filename):
    patterns = [
        r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})',
        r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})',
        r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 6:
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]),
                                    int(groups[3]), int(groups[4]), int(groups[5]))
                elif len(groups) == 5:
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]),
                                    int(groups[3]), int(groups[4]))
                elif len(groups) == 3:
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
            except ValueError:
                continue
    return None


def sort_files_by_timestamp(filenames):
    def sort_key(filename):
        ts = extract_timestamp_from_filename(filename)
        return ts or datetime.min
    return sorted(filenames, key=sort_key)


WEATHER_OPTIONS = [
    {"label": "晴", "value": "sunny"},
    {"label": "多云", "value": "cloudy"},
    {"label": "阴", "value": "overcast"},
    {"label": "雨", "value": "rain"},
    {"label": "雪", "value": "snow"},
    {"label": "未知", "value": "unknown"},
]

TRAFFIC_OPTIONS = [
    {"label": "正常", "value": "normal"},
    {"label": "繁忙", "value": "busy"},
    {"label": "封路", "value": "closed"},
]

DETREND_OPTIONS = [
    {"label": "无", "value": "none"},
    {"label": "线性", "value": "linear"},
    {"label": "多项式", "value": "polynomial"},
]

FILTER_OPTIONS = [
    {"label": "无", "value": "none"},
    {"label": "低通", "value": "low"},
    {"label": "高通", "value": "high"},
    {"label": "带通", "value": "band"},
]

RESAMPLE_OPTIONS = [
    {"label": "保持原始", "value": "original"},
    {"label": "100 Hz", "value": 100},
    {"label": "200 Hz", "value": 200},
    {"label": "500 Hz", "value": 500},
    {"label": "1000 Hz", "value": 1000},
]


layout = dbc.Container([
    html.H2("数据导入与预处理", className="mb-4"),
    
    dcc.Store(id="batch-queue-store", data={"files": [], "index": 0, "results": [], "bridge_id": None, "custom_sr": None}),
    dcc.Store(id="batch-processing-flag", data=False),
    dcc.Store(id="preview-data-store", data=None),
    dcc.Store(id="preview-clip-store", data=None),
    dcc.Store(id="show-all-channels-store", data=False),
    dcc.Store(id="event-list-refresh", data=0),
    dcc.Store(id="unarchived-list-refresh", data=0),
    dcc.Store(id="expanded-event-store", data=None),
    
    dcc.Interval(id="batch-process-interval", interval=300, disabled=True, n_intervals=0),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("批量文件导入"),
                dbc.CardBody([
                    html.Label("选择桥梁:"),
                    dcc.Dropdown(id="import-bridge-selector", placeholder="请选择桥梁"),
                    html.Hr(),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("采样频率 (Hz):"),
                            dbc.Input(id="import-sampling-rate", type="number", min=100, max=1000, step=10, placeholder="100-1000"),
                            html.Small("如CSV包含时间列可自动检测", className="text-muted"),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dcc.Upload(
                        id='upload-multiple-data',
                        children=html.Div([
                            '拖拽多个CSV文件到此处或 ',
                            html.A('点击选择')
                        ]),
                        style={
                            'width': '100%',
                            'height': '80px',
                            'lineHeight': '80px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px 0'
                        },
                        multiple=True
                    ),
                    
                    html.Div(id="batch-progress-container", style={"display": "none"}, children=[
                        html.Label("批量处理进度:"),
                        dbc.Progress(id="batch-progress-bar", value=0, max=100, color="primary", className="mb-2"),
                        html.Div(id="batch-progress-text", className="text-center text-muted mb-2"),
                    ]),
                    
                    html.Hr(),
                    html.Label("处理结果列表:"),
                    html.Div(id="batch-results-list", className="mt-2", style={
                        "maxHeight": "200px",
                        "overflowY": "auto",
                        "border": "1px solid #dee2e6",
                        "borderRadius": "5px",
                        "padding": "10px"
                    }),
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader([
                    "预处理参数配置",
                    dbc.Badge("预设", color="info", className="ms-2", id="preset-badge")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("预设方案:"),
                            dcc.Dropdown(id="preset-selector", placeholder="选择预设方案..."),
                        ], width=8),
                        dbc.Col([
                            dbc.Button("保存", id="save-preset-btn", color="primary", className="mt-4 w-100", size="sm"),
                        ], width=4),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("去趋势方式:"),
                            dcc.Dropdown(id="detrend-method-select", options=DETREND_OPTIONS, value="linear"),
                        ], width=6),
                        dbc.Col([
                            html.Label("多项式阶数:"),
                            dbc.Input(id="poly-order-input", type="number", min=1, max=10, step=1, value=3, disabled=True),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("滤波类型:"),
                            dcc.Dropdown(id="filter-type-select", options=FILTER_OPTIONS, value="none"),
                        ], width=6),
                        dbc.Col([
                            html.Label("截止频率1 (Hz):"),
                            dbc.Input(id="cutoff-freq-1", type="number", min=0.1, step=0.1, disabled=True),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("截止频率2 (Hz):", id="cutoff-freq-2-label", style={"visibility": "hidden"}),
                            dbc.Input(id="cutoff-freq-2", type="number", min=0.1, step=0.1, disabled=True),
                        ], width=6),
                        dbc.Col([
                            html.Label("目标重采样:"),
                            dcc.Dropdown(id="resample-select", options=RESAMPLE_OPTIONS, value="original"),
                        ], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("应用到预览", id="apply-preprocess-btn", color="info", size="sm", className="me-2"),
                            dbc.Button("删除预设", id="delete-preset-btn", color="danger", size="sm"),
                        ], width=12),
                    ]),
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("数据裁剪"),
                dbc.CardBody([
                    html.Div(id="clip-info", className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("确认裁剪", id="confirm-clip-btn", color="warning", disabled=True, className="w-100"),
                        ], width=6),
                        dbc.Col([
                            dbc.Button("取消裁剪", id="cancel-clip-btn", color="secondary", disabled=True, className="w-100"),
                        ], width=6),
                    ]),
                ])
            ]),
            
        ], width=5),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "实时数据预览",
                    dbc.Button("展开全部通道", id="toggle-channels-btn", color="link", size="sm", className="ms-auto", style={"display": "none"})
                ]),
                dbc.CardBody([
                    dcc.Graph(id="realtime-preview-plot", config={
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['select2d'],
                        'modeBarButtonsToRemove': ['lasso2d', 'autoScale2d']
                    }),
                    html.Div(id="preview-file-info", className="mt-2 text-muted"),
                ])
            ]),
        ], width=7),
    ], className="mb-4"),
    
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "事件管理",
                    dbc.Button("+ 创建新事件", id="create-event-btn", color="primary", size="sm", className="ms-auto")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("事件列表:"),
                            html.Div(id="events-list-container", style={
                                "maxHeight": "400px",
                                "overflowY": "auto",
                                "border": "1px solid #dee2e6",
                                "borderRadius": "5px",
                                "padding": "10px"
                            }),
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Tabs([
                                dbc.Tab(label="未归档文件", tab_id="unassigned-tab"),
                                dbc.Tab(label="事件详情", tab_id="event-detail-tab"),
                            ], id="event-tabs", active_tab="unassigned-tab"),
                            
                            html.Div(id="event-tabs-content", className="mt-3"),
                        ], width=6),
                    ]),
                ])
            ])
        ], width=12),
    ]),
    
    html.Div(id="import-notifications"),
    
    dbc.Modal([
        dbc.ModalHeader("创建新测试事件"),
        dbc.ModalBody([
            dbc.Row([
                dbc.Label("事件名称", width=4),
                dbc.Col(dbc.Input(id="new-event-name", placeholder="输入事件名称"), width=8),
            ], className="mb-3"),
            dbc.Row([
                dbc.Label("采集时间", width=4),
                dbc.Col([
                    dbc.Input(
                        id="new-event-time",
                        type="datetime-local",
                        value=datetime.now().strftime("%Y-%m-%dT%H:%M")
                    ),
                ], width=8),
            ], className="mb-3"),
            dbc.Row([
                dbc.Label("天气条件", width=4),
                dbc.Col(dcc.Dropdown(id="new-event-weather", options=WEATHER_OPTIONS, value="unknown"), width=8),
            ], className="mb-3"),
            dbc.Row([
                dbc.Label("交通状态", width=4),
                dbc.Col(dcc.Dropdown(id="new-event-traffic", options=TRAFFIC_OPTIONS, value="normal"), width=8),
            ], className="mb-3"),
            dbc.Row([
                dbc.Label("温度 (°C)", width=4),
                dbc.Col(dbc.Input(id="new-event-temperature", type="number", step=0.1), width=8),
            ], className="mb-3"),
            dbc.Row([
                dbc.Label("风速 (m/s)", width=4),
                dbc.Col(dbc.Input(id="new-event-wind", type="number", step=0.1), width=8),
            ], className="mb-3"),
            dbc.Row([
                dbc.Label("操作人员", width=4),
                dbc.Col(dbc.Input(id="new-event-operator"), width=8),
            ], className="mb-3"),
            dbc.Row([
                dbc.Label("备注", width=4),
                dbc.Col(dbc.Textarea(id="new-event-notes", rows=2), width=8),
            ], className="mb-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button("取消", id="cancel-create-event-btn", color="secondary"),
            dbc.Button("创建", id="confirm-create-event-btn", color="primary"),
        ]),
    ], id="create-event-modal", is_open=False),
    
    dbc.Modal([
        dbc.ModalHeader("保存预设方案"),
        dbc.ModalBody([
            dbc.Row([
                dbc.Label("预设名称", width=4),
                dbc.Col(dbc.Input(id="new-preset-name", placeholder="输入预设方案名称"), width=8),
            ], className="mb-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button("取消", id="cancel-save-preset-btn", color="secondary"),
            dbc.Button("保存", id="confirm-save-preset-btn", color="primary"),
        ]),
    ], id="save-preset-modal", is_open=False),
    
], fluid=True)


@callback(
    Output("import-bridge-selector", "options"),
    Output("import-bridge-selector", "value"),
    Input("url", "pathname"),
    Input("bridge-list-refresh", "data"),
    State("current-bridge-store", "data"),
)
def update_import_bridge_selector(pathname, refresh, store_data):
    bridges = Bridge.list_all()
    options = [{"label": f"{b.name} ({b.id})", "value": b.id} for b in bridges]
    
    bridge_id = None
    if store_data and store_data.get("id"):
        bridge_id = store_data["id"]
        valid_ids = [o["value"] for o in options]
        if bridge_id not in valid_ids:
            bridge_id = None
    
    return options, bridge_id


@callback(
    Output("poly-order-input", "disabled"),
    Input("detrend-method-select", "value"),
)
def toggle_poly_order(detrend_method):
    return detrend_method != "polynomial"


@callback(
    Output("cutoff-freq-1", "disabled"),
    Output("cutoff-freq-2", "disabled"),
    Output("cutoff-freq-2-label", "style"),
    Input("filter-type-select", "value"),
)
def toggle_filter_inputs(filter_type):
    if filter_type == "none":
        return True, True, {"visibility": "hidden"}
    elif filter_type == "band":
        return False, False, {"visibility": "visible"}
    else:
        return False, True, {"visibility": "hidden"}


@callback(
    Output("preset-selector", "options"),
    Input("import-bridge-selector", "value"),
    Input("current-bridge-store", "data"),
)
def update_preset_selector(bridge_id, store_data):
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return []
    presets = PreprocessPreset.list_by_bridge(bid)
    return [{"label": p.name, "value": p.id} for p in presets]


@callback(
    Output("detrend-method-select", "value"),
    Output("poly-order-input", "value"),
    Output("filter-type-select", "value"),
    Output("cutoff-freq-1", "value"),
    Output("cutoff-freq-2", "value"),
    Output("resample-select", "value"),
    Input("preset-selector", "value"),
    State("import-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    prevent_initial_call=True,
)
def load_preset(preset_id, bridge_id, store_data):
    if not preset_id:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    preset = PreprocessPreset.load(bid, preset_id)
    if preset:
        resample_val = preset.target_sr if preset.target_sr else "original"
        return (
            preset.detrend_method,
            preset.poly_order,
            preset.filter_type,
            preset.cutoff_freq,
            preset.cutoff_freq2,
            resample_val
        )
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("save-preset-modal", "is_open"),
    Input("save-preset-btn", "n_clicks"),
    Input("cancel-save-preset-btn", "n_clicks"),
    Input("confirm-save-preset-btn", "n_clicks"),
    State("save-preset-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_save_preset_modal(n1, n2, n3, is_open):
    return not is_open


@callback(
    Output("import-notifications", "children", allow_duplicate=True),
    Output("preset-selector", "options", allow_duplicate=True),
    Output("new-preset-name", "value"),
    Input("confirm-save-preset-btn", "n_clicks"),
    State("new-preset-name", "value"),
    State("import-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("detrend-method-select", "value"),
    State("poly-order-input", "value"),
    State("filter-type-select", "value"),
    State("cutoff-freq-1", "value"),
    State("cutoff-freq-2", "value"),
    State("resample-select", "value"),
    prevent_initial_call=True,
)
def save_preset(n_clicks, name, bridge_id, store_data, detrend_method, poly_order,
                filter_type, cutoff1, cutoff2, resample_val):
    if not name:
        return dbc.Alert("请输入预设名称", color="warning", duration=3000), dash.no_update, dash.no_update
    
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return dbc.Alert("请先选择桥梁", color="danger", duration=3000), dash.no_update, dash.no_update
    
    target_sr = None if resample_val == "original" else resample_val
    
    preset = PreprocessPreset(
        id=str(uuid.uuid4())[:8],
        bridge_id=bid,
        name=name,
        detrend_method=detrend_method or "none",
        poly_order=poly_order or 3,
        filter_type=filter_type or "none",
        cutoff_freq=cutoff1,
        cutoff_freq2=cutoff2,
        target_sr=target_sr
    )
    preset.save()
    
    presets = PreprocessPreset.list_by_bridge(bid)
    options = [{"label": p.name, "value": p.id} for p in presets]
    
    return dbc.Alert(f"预设方案 '{name}' 保存成功", color="success", duration=3000), options, ""


@callback(
    Output("import-notifications", "children", allow_duplicate=True),
    Output("preset-selector", "options", allow_duplicate=True),
    Output("preset-selector", "value"),
    Input("delete-preset-btn", "n_clicks"),
    State("preset-selector", "value"),
    State("import-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    prevent_initial_call=True,
)
def delete_preset(n_clicks, preset_id, bridge_id, store_data):
    if not preset_id:
        return dbc.Alert("请先选择要删除的预设", color="warning", duration=3000), dash.no_update, dash.no_update
    
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return dbc.Alert("请先选择桥梁", color="danger", duration=3000), dash.no_update, dash.no_update
    
    PreprocessPreset.delete(bid, preset_id)
    
    presets = PreprocessPreset.list_by_bridge(bid)
    options = [{"label": p.name, "value": p.id} for p in presets]
    
    return dbc.Alert("预设方案已删除", color="info", duration=3000), options, None


@callback(
    Output("batch-queue-store", "data"),
    Output("batch-process-interval", "disabled"),
    Output("batch-progress-container", "style"),
    Output("batch-results-list", "children"),
    Input("upload-multiple-data", "filename"),
    Input("upload-multiple-data", "contents"),
    State("import-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("import-sampling-rate", "value"),
    State("batch-queue-store", "data"),
    prevent_initial_call=True,
)
def start_batch_upload(filenames, contents, bridge_id, store_data, custom_sr, current_queue):
    if not filenames or not contents:
        return dash.no_update, True, {"display": "none"}, []
    
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return current_queue, True, {"display": "none"}, [
            html.Div([
                html.Span("请先选择桥梁再上传文件", className="text-danger fw-bold")
            ])
        ]
    
    sorted_filenames = sort_files_by_timestamp(filenames)
    sorted_contents = [contents[filenames.index(f)] for f in sorted_filenames]
    
    file_data = []
    for fname, content in zip(sorted_filenames, sorted_contents):
        file_data.append({
            "filename": fname,
            "content": content,
            "status": "pending",
            "error": None,
            "channels": None,
            "sampling_rate": None,
            "n_samples": None,
            "file_id": None
        })
    
    queue_data = {
        "files": file_data,
        "index": 0,
        "results": [],
        "bridge_id": bid,
        "custom_sr": custom_sr
    }
    
    results_rows = []
    for fd in file_data:
        results_rows.append(html.Div([
            html.Span(fd["filename"], className="me-2"),
            html.Span("等待处理...", className="text-muted"),
        ], className="mb-1"))
    
    return queue_data, False, {"display": "block"}, results_rows


@callback(
    Output("batch-queue-store", "data", allow_duplicate=True),
    Output("batch-results-list", "children", allow_duplicate=True),
    Output("batch-progress-bar", "value"),
    Output("batch-progress-text", "children"),
    Output("batch-process-interval", "disabled", allow_duplicate=True),
    Output("preview-data-store", "data"),
    Output("import-notifications", "children", allow_duplicate=True),
    Input("batch-process-interval", "n_intervals"),
    State("batch-queue-store", "data"),
    State("preview-data-store", "data"),
    prevent_initial_call=True,
)
def process_next_file(n_intervals, queue_data, current_preview):
    if not queue_data or not queue_data.get("files"):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update
    
    files = queue_data["files"]
    idx = queue_data["index"]
    total = len(files)
    bid = queue_data["bridge_id"]
    cust_sr = queue_data.get("custom_sr")
    
    if idx >= total:
        results_rows = []
        for fd in files:
            if fd["status"] == "error":
                color = "danger"
                status = f"失败: {fd.get('error', '')}"
            else:
                color = "success"
                status = "成功"
            results_rows.append(html.Div([
                html.Span(fd["filename"], className="me-2"),
                html.Span(f"通道: {fd.get('channels', '-')} | "
                          f"采样率: {fd.get('sampling_rate', '-')} Hz | "
                          f"点数: {fd.get('n_samples', '-')} | "),
                html.Span(status, className=f"text-{color} fw-bold"),
            ], className="mb-1"))
        
        success_count = sum(1 for f in files if f["status"] == "success")
        msg = dbc.Alert(f"批量处理完成，成功 {success_count}/{total}", color="success", duration=5000)
        
        return queue_data, results_rows, 100, f"处理完成: {total} / {total}", True, current_preview, msg
    
    current_file = files[idx]
    filename = current_file["filename"]
    content = current_file["content"]
    
    preview_data = current_preview
    
    try:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        tmp_path = f"/tmp/{filename}"
        with open(tmp_path, 'wb') as f:
            f.write(decoded)
        
        valid, msg = check_file_size(tmp_path)
        if not valid:
            raise ValueError(msg)
        
        df, detected_sr, channel_names = import_csv(
            tmp_path,
            has_time_column=True,
            custom_sampling_rate=cust_sr
        )
        
        n_samples = len(df)
        n_channels = detect_channels(df)
        duration = n_samples / detected_sr
        
        file_id = str(uuid.uuid4())[:8]
        uf = UnarchivedFile(
            id=file_id,
            bridge_id=bid,
            filename=filename,
            sampling_rate=detected_sr,
            channel_names=channel_names,
            n_samples=n_samples,
            duration=duration,
            upload_time=datetime.now()
        )
        uf.save_metadata()
        uf.save_data(df)
        
        files[idx]["status"] = "success"
        files[idx]["channels"] = n_channels
        files[idx]["sampling_rate"] = int(detected_sr)
        files[idx]["n_samples"] = n_samples
        files[idx]["file_id"] = file_id
        
        preview_data = {
            "file_id": file_id,
            "filename": filename,
            "bridge_id": bid,
            "sampling_rate": detected_sr,
            "channel_names": channel_names,
            "data": df.to_dict(orient='records'),
            "duration": duration,
            "clip_range": None
        }
        
    except Exception as e:
        files[idx]["status"] = "error"
        files[idx]["error"] = str(e)[:80]
    
    finally:
        if 'tmp_path' in dir() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    queue_data["index"] = idx + 1
    queue_data["files"] = files
    
    results_rows = []
    for i, fd in enumerate(files):
        if fd["status"] == "success":
            color = "success"
            status = "成功"
            info = f"通道: {fd.get('channels', '-')} | 采样率: {fd.get('sampling_rate', '-')} Hz | 点数: {fd.get('n_samples', '-')} | "
        elif fd["status"] == "error":
            color = "danger"
            status = f"失败: {fd.get('error', '')}"
            info = ""
        elif i == idx:
            color = "primary"
            status = "处理中..."
            info = ""
        else:
            color = "muted"
            status = "等待处理..."
            info = ""
        
        results_rows.append(html.Div([
            html.Span(fd["filename"], className="me-2"),
            html.Span(info) if info else None,
            html.Span(status, className=f"text-{color} fw-bold"),
        ], className="mb-1"))
    
    progress = (idx / total) * 100
    
    return queue_data, results_rows, progress, f"处理中: {idx + 1} / {total}", False, preview_data, None


@callback(
    Output("realtime-preview-plot", "figure"),
    Output("preview-file-info", "children"),
    Output("toggle-channels-btn", "style"),
    Output("toggle-channels-btn", "children"),
    Output("show-all-channels-store", "data", allow_duplicate=True),
    Input("preview-data-store", "data"),
    Input("apply-preprocess-btn", "n_clicks"),
    Input("preview-clip-store", "data"),
    Input("toggle-channels-btn", "n_clicks"),
    State("detrend-method-select", "value"),
    State("poly-order-input", "value"),
    State("filter-type-select", "value"),
    State("cutoff-freq-1", "value"),
    State("cutoff-freq-2", "value"),
    State("resample-select", "value"),
    State("show-all-channels-store", "data"),
    prevent_initial_call=True,
)
def update_realtime_preview(preview_data, apply_n_clicks, clip_range, toggle_n,
                            detrend_method, poly_order, filter_type, cutoff1, cutoff2, resample_val,
                            current_show_all):
    if not preview_data:
        fig = go.Figure()
        fig.update_layout(
            title="请先导入数据文件",
            xaxis_title="时间 (s)",
            yaxis_title="幅值"
        )
        return fig, html.P("无数据", className="text-muted"), {"display": "none"}, "展开全部通道", current_show_all
    
    df = pd.DataFrame(preview_data["data"])
    sampling_rate = preview_data["sampling_rate"]
    channel_names = preview_data["channel_names"]
    filename = preview_data["filename"]
    show_all_val = current_show_all
    
    triggered = ctx.triggered_id
    
    if triggered == "toggle-channels-btn":
        show_all_val = not current_show_all
    
    if clip_range and clip_range.get("start") is not None and clip_range.get("end") is not None:
        start_idx = int(clip_range["start"] * sampling_rate)
        end_idx = int(clip_range["end"] * sampling_rate)
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    if triggered == "apply-preprocess-btn":
        target_sr = None if resample_val == "original" else resample_val
        try:
            df, final_sr, _ = preprocess_pipeline(
                df, sampling_rate,
                detrend_method=detrend_method or "none",
                poly_order=poly_order or 3,
                filter_type=filter_type or "none",
                cutoff_freq=cutoff1,
                cutoff_freq2=cutoff2,
                target_sr=target_sr
            )
            sampling_rate = final_sr
        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=f"预处理错误: {str(e)}")
            return fig, html.P(str(e), className="text-danger"), {"display": "none"}, "展开全部通道", current_show_all
    
    n_channels = len(channel_names)
    display_channels = channel_names if show_all_val or n_channels <= 8 else channel_names[:8]
    n_display = len(display_channels)
    
    fig = make_subplots(rows=n_display, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    time_vec = np.arange(len(df)) / sampling_rate
    
    for i, ch in enumerate(display_channels):
        fig.add_trace(
            go.Scatter(
                x=time_vec,
                y=df[ch].values,
                name=ch,
                mode='lines',
                hovertemplate=f"通道: {ch}<br>时间: %{{x:.3f}} s<br>幅值: %{{y:.4f}}<extra></extra>"
            ),
            row=i + 1, col=1
        )
        fig.update_yaxes(title_text="幅值", row=i + 1, col=1)
    
    fig.update_xaxes(title_text="时间 (s)", row=n_display, col=1)
    
    title = f"数据预览: {filename} ({n_display}/{n_channels} 通道)"
    if clip_range and clip_range.get("start") is not None:
        title += f" [已裁剪: {clip_range['start']:.1f}s - {clip_range['end']:.1f}s]"
    
    fig.update_layout(
        title=title,
        height=100 + n_display * 80,
        showlegend=False,
        hovermode='x unified',
        dragmode='select'
    )
    
    if clip_range and clip_range.get("start") is not None:
        for i in range(n_display):
            fig.add_vrect(
                x0=clip_range["start"],
                x1=clip_range["end"],
                fillcolor="rgba(255, 200, 0, 0.2)",
                layer="below",
                line_width=0,
                row=i + 1, col=1
            )
    
    info = html.Div([
        html.Span(f"文件: {filename} | "),
        html.Span(f"通道数: {n_channels} | "),
        html.Span(f"采样率: {sampling_rate:.0f} Hz | "),
        html.Span(f"数据点数: {len(df)} | "),
        html.Span(f"时长: {len(df) / sampling_rate:.1f} s"),
    ])
    
    btn_style = {"display": "inline-block"} if n_channels > 8 else {"display": "none"}
    btn_text = "收起通道" if show_all_val else "展开全部通道"
    
    return fig, info, btn_style, btn_text, show_all_val


@callback(
    Output("preview-clip-store", "data"),
    Output("clip-info", "children"),
    Output("confirm-clip-btn", "disabled"),
    Output("cancel-clip-btn", "disabled"),
    Input("realtime-preview-plot", "selectedData"),
    Input("confirm-clip-btn", "n_clicks"),
    Input("cancel-clip-btn", "n_clicks"),
    State("preview-data-store", "data"),
    State("preview-clip-store", "data"),
    prevent_initial_call=True,
)
def handle_clip_selection(selected_data, confirm_n, cancel_n, preview_data, current_clip):
    triggered = ctx.triggered_id
    
    if triggered == "cancel-clip-btn":
        return None, html.P("未选择裁剪区域", className="text-muted"), True, True
    
    if triggered == "confirm-clip-btn":
        return None, html.P("裁剪已应用", className="text-success"), True, True
    
    if triggered == "realtime-preview-plot" and selected_data and preview_data:
        x_range = selected_data.get("range", {}).get("x")
        if x_range and len(x_range) == 2:
            start, end = x_range
            sr = preview_data["sampling_rate"]
            duration = preview_data["duration"]
            start = max(0, min(start, end))
            end = min(duration, max(start, end))
            
            info = html.Div([
                html.Span("选中区域: ", className="fw-bold"),
                html.Span(f"{start:.2f}s - {end:.2f}s", className="text-warning"),
                html.Span(f" (时长: {end - start:.2f}s)"),
            ])
            
            return {"start": start, "end": end}, info, False, False
    
    return current_clip, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("preview-data-store", "data", allow_duplicate=True),
    Input("confirm-clip-btn", "n_clicks"),
    State("preview-data-store", "data"),
    State("preview-clip-store", "data"),
    prevent_initial_call=True,
)
def apply_clip_to_preview(n_clicks, preview_data, clip_range):
    if not preview_data or not clip_range:
        return dash.no_update
    
    df = pd.DataFrame(preview_data["data"])
    sr = preview_data["sampling_rate"]
    start_idx = int(clip_range["start"] * sr)
    end_idx = int(clip_range["end"] * sr)
    clipped_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    file_id = preview_data["file_id"]
    bridge_id = preview_data["bridge_id"]
    uf = UnarchivedFile.load(bridge_id, file_id)
    if uf:
        uf.save_data(clipped_df)
    
    return {
        **preview_data,
        "data": clipped_df.to_dict(orient='records'),
        "duration": len(clipped_df) / sr,
        "clip_range": clip_range
    }


@callback(
    Output("create-event-modal", "is_open"),
    Input("create-event-btn", "n_clicks"),
    Input("cancel-create-event-btn", "n_clicks"),
    Input("confirm-create-event-btn", "n_clicks"),
    State("create-event-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_create_event_modal(n1, n2, n3, is_open):
    return not is_open


@callback(
    Output("import-notifications", "children", allow_duplicate=True),
    Output("event-list-refresh", "data"),
    Output("new-event-name", "value"),
    Output("new-event-time", "value"),
    Output("new-event-weather", "value"),
    Output("new-event-traffic", "value"),
    Output("new-event-temperature", "value"),
    Output("new-event-wind", "value"),
    Output("new-event-operator", "value"),
    Output("new-event-notes", "value"),
    Input("confirm-create-event-btn", "n_clicks"),
    State("import-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("new-event-name", "value"),
    State("new-event-time", "value"),
    State("new-event-weather", "value"),
    State("new-event-traffic", "value"),
    State("new-event-temperature", "value"),
    State("new-event-wind", "value"),
    State("new-event-operator", "value"),
    State("new-event-notes", "value"),
    State("event-list-refresh", "data"),
    prevent_initial_call=True,
)
def create_new_event(n_clicks, bridge_id, store_data, name, event_time, weather, traffic,
                     temp, wind, operator, notes, refresh):
    if not name:
        return dbc.Alert("请输入事件名称", color="warning", duration=3000), \
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return dbc.Alert("请先选择桥梁", color="danger", duration=3000), \
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        if isinstance(event_time, str):
            if 'T' in event_time:
                event_time_dt = datetime.fromisoformat(event_time)
            else:
                event_time_dt = datetime.fromisoformat(event_time.replace(' ', 'T'))
        else:
            event_time_dt = datetime.now()
        
        metadata = TestEventMetadata(
            collection_time=event_time_dt,
            weather=weather or "unknown",
            temperature=temp,
            wind_speed=wind,
            traffic_status=traffic or "normal",
            operator=operator or "",
            notes=notes or ""
        )
        
        event_id = str(uuid.uuid4())[:8]
        event = TestEvent(
            id=event_id,
            bridge_id=bid,
            name=name,
            metadata=metadata,
            sampling_rate=0,
            data=pd.DataFrame(),
            channel_names=[],
            preprocessing_params={}
        )
        event.save()
        
        return dbc.Alert(f"事件 '{name}' 创建成功！ID: {event_id}", color="success", duration=3000), \
               (refresh or 0) + 1, "", datetime.now().strftime("%Y-%m-%dT%H:%M"), \
               "unknown", "normal", None, None, "", ""
        
    except Exception as e:
        return dbc.Alert(f"创建失败: {str(e)}", color="danger", duration=5000), \
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@callback(
    Output("events-list-container", "children"),
    Input("event-list-refresh", "data"),
    Input("import-bridge-selector", "value"),
    Input("current-bridge-store", "data"),
    Input("unarchived-list-refresh", "data"),
)
def update_events_list(_, bridge_id, store_data, _2):
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return html.P("请先选择桥梁", className="text-muted text-center")
    
    events = TestEvent.list_by_bridge(bid)
    if not events:
        return html.P("暂无测试事件", className="text-muted text-center")
    
    event_rows = []
    for event in events:
        assigned_files = UnarchivedFile.list_by_bridge(bid, only_unassigned=False)
        event_files = [f for f in assigned_files if f.event_id == event.id]
        
        weather_label = dict(WEATHER_OPTIONS).get(event.metadata.weather, event.metadata.weather)
        traffic_label = dict(TRAFFIC_OPTIONS).get(event.metadata.traffic_status, event.metadata.traffic_status)
        
        row = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6(event.name, className="mb-1"),
                        html.Small(f"采集时间: {event.metadata.collection_time.strftime('%Y-%m-%d %H:%M')}", className="text-muted"),
                    ], width=4),
                    dbc.Col(html.Div(f"天气: {weather_label}"), width=2),
                    dbc.Col(html.Div(f"交通: {traffic_label}"), width=2),
                    dbc.Col(html.Div(f"文件: {len(event_files)}"), width=2),
                    dbc.Col([
                        dbc.Button("查看", id={"type": "view-event-btn", "index": event.id},
                                   size="sm", color="info", className="me-1"),
                        dbc.Button("删除", id={"type": "delete-event-btn", "index": event.id},
                                   size="sm", color="danger"),
                    ], width=2, className="text-end"),
                ]),
            ])
        ], className="mb-2")
        event_rows.append(row)
    
    return event_rows


@callback(
    Output("expanded-event-store", "data"),
    Output("event-tabs", "active_tab"),
    Input({"type": "view-event-btn", "index": ALL}, "n_clicks"),
    State("expanded-event-store", "data"),
    prevent_initial_call=True,
)
def view_event_detail(n_clicks_list, current_event):
    if not any(n_clicks_list):
        return dash.no_update, dash.no_update
    
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "view-event-btn":
        event_id = triggered["index"]
        return event_id, "event-detail-tab"
    
    return dash.no_update, dash.no_update


@callback(
    Output("event-tabs-content", "children"),
    Input("event-tabs", "active_tab"),
    Input("expanded-event-store", "data"),
    Input("event-list-refresh", "data"),
    Input("unarchived-list-refresh", "data"),
    State("import-bridge-selector", "value"),
    State("current-bridge-store", "data"),
)
def render_event_tabs(active_tab, expanded_event_id, _1, _2, bridge_id, store_data):
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return html.P("请先选择桥梁", className="text-muted")
    
    if active_tab == "unassigned-tab":
        unassigned = UnarchivedFile.list_by_bridge(bid, only_unassigned=True)
        
        if not unassigned:
            return html.Div([
                html.P("暂无未归档文件", className="text-muted"),
                html.Hr(),
                html.Label("选择事件以分配文件:"),
                dcc.Dropdown(id="assign-event-selector", placeholder="选择目标事件..."),
                html.Div(id="assign-status", className="mt-2"),
            ])
        
        checklist_options = []
        for uf in unassigned:
            label = (f"{uf.filename} | 通道: {len(uf.channel_names)} | "
                     f"{uf.sampling_rate:.0f}Hz | {uf.n_samples}点 | "
                     f"{uf.duration:.1f}s")
            checklist_options.append({"label": label, "value": uf.id})
        
        events = TestEvent.list_by_bridge(bid)
        event_options = [{"label": e.name, "value": e.id} for e in events]
        
        return html.Div([
            html.Label(f"未归档文件 ({len(unassigned)}):"),
            dcc.Checklist(
                id="unassigned-files-checklist",
                options=checklist_options,
                value=[],
                labelStyle={"display": "block", "margin": "5px 0"},
                style={"maxHeight": "250px", "overflowY": "auto"}
            ),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Label("分配到事件:"),
                    dcc.Dropdown(id="assign-event-selector", options=event_options,
                                 placeholder="选择目标事件..."),
                ], width=8),
                dbc.Col([
                    dbc.Button("分配选中文件", id="assign-files-btn", color="primary",
                               className="mt-4 w-100", disabled=len(event_options) == 0),
                ], width=4),
            ]),
            html.Div(id="assign-status", className="mt-2"),
        ])
    
    elif active_tab == "event-detail-tab":
        if not expanded_event_id:
            return html.P("请从左侧列表点击'查看'按钮查看事件详情", className="text-muted")
        
        event = TestEvent.load(expanded_event_id)
        if not event:
            return html.P("事件不存在", className="text-danger")
        
        assigned_files = UnarchivedFile.list_by_bridge(bid, only_unassigned=False)
        event_files = [f for f in assigned_files if f.event_id == event.id]
        
        weather_label = dict(WEATHER_OPTIONS).get(event.metadata.weather, event.metadata.weather)
        traffic_label = dict(TRAFFIC_OPTIONS).get(event.metadata.traffic_status, event.metadata.traffic_status)
        
        file_rows = []
        for f in event_files:
            file_rows.append(html.Div([
                html.Span(f.filename, className="me-2"),
                html.Span(f"通道: {len(f.channel_names)} | "
                          f"{f.sampling_rate:.0f}Hz | {f.n_samples}点 | "
                          f"{f.duration:.1f}s", className="text-muted me-2"),
                dbc.Button("移出", id={"type": "unassign-file-btn", "index": f.id},
                           size="sm", color="warning", outline=True),
            ], className="mb-1"))
        
        return html.Div([
            html.H5(event.name, className="mb-3"),
            html.Div([
                html.P(f"采集时间: {event.metadata.collection_time.strftime('%Y-%m-%d %H:%M')}"),
                html.P(f"天气: {weather_label}"),
                html.P(f"交通状态: {traffic_label}"),
                html.P(f"温度: {event.metadata.temperature or '-'} °C"),
                html.P(f"风速: {event.metadata.wind_speed or '-'} m/s"),
                html.P(f"操作人员: {event.metadata.operator or '-'}"),
                html.P(f"备注: {event.metadata.notes or '-'}"),
            ], className="mb-3"),
            html.Hr(),
            html.Label(f"关联文件 ({len(event_files)}):"),
            html.Div(file_rows if file_rows else html.P("暂无关联文件", className="text-muted"),
                     style={"maxHeight": "200px", "overflowY": "auto"}),
        ])
    
    return html.P("请选择标签页", className="text-muted")


@callback(
    Output("assign-status", "children"),
    Output("unarchived-list-refresh", "data"),
    Output("unassigned-files-checklist", "value"),
    Input("assign-files-btn", "n_clicks"),
    State("unassigned-files-checklist", "value"),
    State("assign-event-selector", "value"),
    State("import-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("unarchived-list-refresh", "data"),
    prevent_initial_call=True,
)
def assign_files_to_event(n_clicks, selected_files, event_id, bridge_id, store_data, refresh):
    if not selected_files:
        return dbc.Alert("请先选择要分配的文件", color="warning", duration=3000), \
               dash.no_update, dash.no_update
    if not event_id:
        return dbc.Alert("请选择目标事件", color="warning", duration=3000), \
               dash.no_update, dash.no_update
    
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return dbc.Alert("请先选择桥梁", color="danger", duration=3000), \
               dash.no_update, dash.no_update
    
    for file_id in selected_files:
        uf = UnarchivedFile.load(bid, file_id)
        if uf:
            uf.assign_to_event(event_id)
    
    event = TestEvent.load(event_id)
    return dbc.Alert(f"已将 {len(selected_files)} 个文件分配到事件 '{event.name}'",
                     color="success", duration=3000), (refresh or 0) + 1, []


@callback(
    Output("import-notifications", "children", allow_duplicate=True),
    Output("unarchived-list-refresh", "data", allow_duplicate=True),
    Input({"type": "unassign-file-btn", "index": ALL}, "n_clicks"),
    State("import-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("unarchived-list-refresh", "data"),
    prevent_initial_call=True,
)
def unassign_file_from_event(n_clicks_list, bridge_id, store_data, refresh):
    if not any(n_clicks_list):
        return dash.no_update, dash.no_update
    
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "unassign-file-btn":
        return dash.no_update, dash.no_update
    
    file_id = triggered["index"]
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return dash.no_update, dash.no_update
    
    uf = UnarchivedFile.load(bid, file_id)
    if uf:
        uf.unassign_from_event()
        return dbc.Alert(f"文件 '{uf.filename}' 已移出事件", color="info", duration=3000), \
               (refresh or 0) + 1
    
    return dash.no_update, dash.no_update


@callback(
    Output("import-notifications", "children", allow_duplicate=True),
    Output("event-list-refresh", "data", allow_duplicate=True),
    Output("unarchived-list-refresh", "data", allow_duplicate=True),
    Input({"type": "delete-event-btn", "index": ALL}, "n_clicks"),
    State("import-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("event-list-refresh", "data"),
    State("unarchived-list-refresh", "data"),
    prevent_initial_call=True,
)
def delete_event(n_clicks_list, bridge_id, store_data, event_refresh, unarchived_refresh):
    if not any(n_clicks_list):
        return dash.no_update, dash.no_update, dash.no_update
    
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "delete-event-btn":
        return dash.no_update, dash.no_update, dash.no_update
    
    event_id = triggered["index"]
    bid = bridge_id or (store_data.get("id") if store_data else None)
    if not bid:
        return dash.no_update, dash.no_update, dash.no_update
    
    event = TestEvent.load(event_id)
    if not event:
        return dash.no_update, dash.no_update, dash.no_update
    
    event_dir = os.path.join(TEST_EVENTS_DIR, event_id)
    if os.path.exists(event_dir):
        shutil.rmtree(event_dir)
    
    assigned_files = UnarchivedFile.list_by_bridge(bid, only_unassigned=False)
    for uf in assigned_files:
        if uf.event_id == event_id:
            UnarchivedFile.delete(bid, uf.id)
    
    return dbc.Alert(f"事件 '{event.name}' 已删除", color="info", duration=3000), \
           (event_refresh or 0) + 1, (unarchived_refresh or 0) + 1
