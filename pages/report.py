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
from src.models.damage_index import DamageIndex, DamageType
from src.models.alert import Alert
from src.report.pdf_report import generate_health_report
from src.monitoring.trend_analysis import compute_trend_data, create_trend_figure

dash.register_page(__name__, path='/report')

layout = dbc.Container([
    html.H2("报告生成", className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("报告内容选择"),
                dbc.CardBody([
                    html.Label("选择桥梁:"),
                    dcc.Dropdown(id="report-bridge-selector", placeholder="请选择桥梁"),
                    html.Hr(),
                    html.Label("选择测试事件:"),
                    dcc.Dropdown(id="report-event-selector", placeholder="请选择测试事件"),
                    html.Hr(),
                    html.Label("选择基准测试事件:"),
                    dcc.Dropdown(id="report-baseline-selector", placeholder="可选"),
                    html.Hr(),
                    html.Label("报告内容:"),
                    dbc.Checklist(
                        options=[
                            {"label": "模态参数识别结果", "value": "modal"},
                            {"label": "损伤指标分析", "value": "damage"},
                            {"label": "温度补偿说明", "value": "temp"},
                            {"label": "长期趋势分析", "value": "trend"},
                            {"label": "预警记录", "value": "alerts"},
                        ],
                        value=["modal", "damage", "trend", "alerts"],
                        id="report-content-options",
                    ),
                    html.Hr(),
                    html.Label("输出路径 (可选):"),
                    dbc.Input(id="report-output-path", placeholder="默认为当前目录"),
                    html.Hr(),
                    dbc.Button("生成PDF报告", id="generate-report-btn", color="primary", className="w-100"),
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("报告预览"),
                dbc.CardBody(id="report-preview")
            ])
        ], width=8),
    ], className="mb-4"),
    
    dcc.Download(id="download-report"),
    html.Div(id="report-notifications"),
], fluid=True)


@callback(
    Output("report-bridge-selector", "options"),
    Input("report-bridge-selector", "value"),
)
def update_bridge_selector(_):
    bridges = Bridge.list_all()
    return [{"label": b.name, "value": b.id} for b in bridges]


@callback(
    Output("report-event-selector", "options"),
    Output("report-baseline-selector", "options"),
    Input("report-bridge-selector", "value"),
)
def update_event_selectors(bridge_id):
    if not bridge_id:
        return [], []
    
    events = TestEvent.list_by_bridge(bridge_id)
    options = [{"label": f"{e.name} ({e.metadata.collection_time.strftime('%Y-%m-%d %H:%M')})", "value": e.id} for e in events]
    
    return options, options


@callback(
    Output("report-preview", "children"),
    Input("report-event-selector", "value"),
    Input("report-baseline-selector", "value"),
    State("report-bridge-selector", "value"),
    State("report-content-options", "value"),
    prevent_initial_call=True,
)
def update_report_preview(event_id, baseline_id, bridge_id, content_options):
    if not bridge_id or not event_id:
        return html.P("请选择桥梁和测试事件")
    
    bridge = Bridge.load(bridge_id)
    event = TestEvent.load(event_id)
    
    if bridge is None or event is None:
        return html.P("数据加载失败")
    
    preview = []
    preview.append(html.H4(f"报告预览 - {event.name}"))
    preview.append(html.Hr())
    
    modal_params = ModalParams.load(event_id)
    baseline_params = ModalParams.load(baseline_id) if baseline_id else None
    
    if "modal" in content_options and modal_params is not None:
        preview.append(html.H5("1. 模态参数识别结果"))
        if modal_params.mode_shapes:
            table_data = [["模态阶次", "频率 (Hz)", "阻尼比 (%)", "质量"]]
            for i, mode in enumerate(modal_params.mode_shapes):
                quality = "良好" if mode.damping_quality == "good" else "较差"
                table_data.append([
                    f"第{i+1}阶",
                    f"{mode.frequency:.4f}",
                    f"{mode.damping_ratio * 100:.4f}",
                    quality
                ])
            preview.append(dbc.Table(table_data, bordered=True, hover=True, size="sm"))
        preview.append(html.Hr())
    
    if "damage" in content_options:
        preview.append(html.H5("2. 损伤指标分析"))
        damage_indices = DamageIndex.load_all(event_id)
        
        if damage_indices:
            for dt, di in damage_indices.items():
                preview.append(html.H6(f"2.1 {dt.value}"))
                if len(di.values) > 0:
                    preview.append(html.P(f"温度补偿: {'是' if di.temperature_compensated else '否'}"))
                    max_val = np.max(np.abs(di.values))
                    preview.append(html.P(f"最大值: {max_val:.4f}, 阈值: {di.threshold}"))
                    
                    anomalous = di.get_anomalous_indices()
                    if len(anomalous) > 0:
                        preview.append(html.P(f"超限位置: {', '.join([str(i+1) for i in anomalous])}", className="text-danger"))
        else:
            preview.append(html.P("请先运行损伤检测", className="text-muted"))
        preview.append(html.Hr())
    
    if "alerts" in content_options:
        preview.append(html.H5("3. 预警记录"))
        alerts = Alert.load_by_event(event_id)
        if alerts:
            alert_rows = []
            for alert in alerts:
                level_text = "红色" if alert.level.value == "red" else "黄色" if alert.level.value == "yellow" else "信息"
                alert_rows.append(html.Tr([
                    html.Td(alert.trigger_time.strftime("%Y-%m-%d %H:%M")),
                    html.Td(level_text),
                    html.Td(alert.metric),
                    html.Td(f"{alert.current_value:.4f}"),
                    html.Td(alert.suggestion),
                ]))
            preview.append(dbc.Table([
                html.Thead(html.Tr([html.Th("时间"), html.Th("级别"), html.Th("指标"), html.Th("值"), html.Th("建议")])),
                html.Tbody(alert_rows)
            ], bordered=True, hover=True, size="sm"))
        else:
            preview.append(html.P("无预警记录", className="text-muted"))
        preview.append(html.Hr())
    
    return html.Div(preview)


@callback(
    Output("report-notifications", "children"),
    Output("download-report", "data"),
    Input("generate-report-btn", "n_clicks"),
    State("report-bridge-selector", "value"),
    State("report-event-selector", "value"),
    State("report-baseline-selector", "value"),
    State("report-content-options", "value"),
    State("report-output-path", "value"),
    prevent_initial_call=True,
)
def generate_report(n_clicks, bridge_id, event_id, baseline_id, content_options, output_path):
    if not bridge_id or not event_id:
        return dbc.Alert("请选择桥梁和测试事件", color="warning"), None
    
    try:
        bridge = Bridge.load(bridge_id)
        event = TestEvent.load(event_id)
        
        if bridge is None or event is None:
            return dbc.Alert("数据加载失败", color="danger"), None
        
        modal_params = ModalParams.load(event_id)
        if modal_params is None:
            return dbc.Alert("请先运行模态分析", color="warning"), None
        
        baseline_params = ModalParams.load(baseline_id) if baseline_id else None
        damage_indices = DamageIndex.load_all(event_id) if "damage" in content_options else None
        alerts = Alert.load_by_event(event_id) if "alerts" in content_options else None
        
        trend_figures = []
        if "trend" in content_options:
            events = TestEvent.list_by_bridge(bridge_id)
            modal_list = []
            valid_events = []
            for e in events:
                mp = ModalParams.load(e.id)
                if mp is not None and len(mp.mode_shapes) > 0:
                    modal_list.append(mp)
                    valid_events.append(e)
            
            if len(modal_list) > 1:
                times, freq_matrix, damp_matrix = compute_trend_data(valid_events, modal_list)
                for i in range(min(3, freq_matrix.shape[1])):
                    fig = create_trend_figure(
                        times, freq_matrix, i,
                        metric_name="频率",
                        unit="Hz",
                        show_cusum=False
                    )
                    trend_figures.append(fig)
        
        temp_models = None
        if "temp" in content_options:
            temp_models = {}
        
        if not output_path:
            output_path = os.path.join(
                os.path.expanduser("~"),
                f"健康评估报告_{bridge.name}_{event.metadata.collection_time.strftime('%Y%m%d_%H%M%S')}.pdf"
            )
        
        report_path = generate_health_report(
            bridge=bridge,
            test_event=event,
            modal_params=modal_params,
            baseline_params=baseline_params,
            damage_indices=damage_indices,
            temperature_models=temp_models,
            trend_figures=trend_figures,
            alerts=alerts,
            output_path=output_path
        )
        
        return dbc.Alert(f"报告生成成功！保存至: {report_path}", color="success", duration=5000), \
               dcc.send_file(report_path)
        
    except Exception as e:
        return dbc.Alert(f"报告生成失败: {str(e)}", color="danger"), None
