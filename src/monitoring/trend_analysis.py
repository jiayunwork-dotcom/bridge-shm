import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime
from src.models.test_event import TestEvent
from src.models.modal_params import ModalParams
from config import (
    DEFAULT_CUSUM_K,
    DEFAULT_CUSUM_H,
    DEFAULT_CONTROL_CHART_SIGMA
)


def compute_trend_data(
    events: List[TestEvent],
    modal_params_list: List[ModalParams]
) -> Tuple[List[datetime], np.ndarray, np.ndarray]:
    times = []
    frequencies = []
    damping_ratios = []
    
    for event, params in zip(events, modal_params_list):
        times.append(event.metadata.collection_time)
        frequencies.append(params.get_mode_frequencies())
        damping_ratios.append(params.get_mode_damping_ratios())
    
    n_modes = max([len(f) for f in frequencies]) if frequencies else 0
    
    freq_matrix = np.full((len(times), n_modes), np.nan)
    damp_matrix = np.full((len(times), n_modes), np.nan)
    
    for i, (freq, damp) in enumerate(zip(frequencies, damping_ratios)):
        n = len(freq)
        freq_matrix[i, :n] = freq
        n_d = len(damp)
        damp_matrix[i, :n_d] = damp
    
    return times, freq_matrix, damp_matrix


def cusum_detection(
    data: np.ndarray,
    k: float = DEFAULT_CUSUM_K,
    h: float = DEFAULT_CUSUM_H,
    reset_period: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data)
    
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    
    mean = np.nanmean(data)
    std = np.nanstd(data)
    
    if std == 0:
        std = 1.0
    
    k_sigma = k * std
    h_sigma = h * std
    
    cumsum_pos = np.zeros(n)
    cumsum_neg = np.zeros(n)
    anomalies = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if np.isnan(data[i]):
            cumsum_pos[i] = cumsum_pos[i-1]
            cumsum_neg[i] = cumsum_neg[i-1]
            continue
        
        deviation = data[i] - mean
        
        cumsum_pos[i] = max(0, cumsum_pos[i-1] + deviation - k_sigma)
        cumsum_neg[i] = max(0, cumsum_neg[i-1] - deviation - k_sigma)
        
        if cumsum_pos[i] > h_sigma or cumsum_neg[i] > h_sigma:
            anomalies[i] = True
            if reset_period is not None and i % reset_period == 0:
                cumsum_pos[i] = 0
                cumsum_neg[i] = 0
    
    return cumsum_pos, cumsum_neg, anomalies


def control_chart_limits(
    data: np.ndarray,
    sigma_level: float = DEFAULT_CONTROL_CHART_SIGMA
) -> Tuple[float, float, float]:
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return 0.0, 0.0, 0.0
    
    center_line = np.mean(valid_data)
    std = np.std(valid_data)
    
    upper_limit = center_line + sigma_level * std
    lower_limit = center_line - sigma_level * std
    
    return center_line, upper_limit, lower_limit


def detect_control_chart_anomalies(
    data: np.ndarray,
    sigma_level: float = DEFAULT_CONTROL_CHART_SIGMA
) -> Tuple[np.ndarray, float, float, float]:
    center_line, upper_limit, lower_limit = control_chart_limits(data, sigma_level)
    
    anomalies = np.zeros_like(data, dtype=bool)
    for i, val in enumerate(data):
        if not np.isnan(val):
            if val > upper_limit or val < lower_limit:
                anomalies[i] = True
    
    return anomalies, center_line, upper_limit, lower_limit


def create_trend_figure(
    times: List[datetime],
    data: np.ndarray,
    mode_index: int,
    metric_name: str = "频率",
    unit: str = "Hz",
    show_control_limits: bool = True,
    show_cusum: bool = True
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    valid_data = data[:, mode_index]
    center_line, upper_limit, lower_limit = control_chart_limits(valid_data)
    anomalies, _, _, _ = detect_control_chart_anomalies(valid_data)
    
    rows = 2 if show_cusum else 1
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            f"第{mode_index+1}阶{metric_name}趋势图",
            "CUSUM突变检测"
        ][:rows]
    )
    
    fig.add_trace(
        go.Scatter(
            x=times, y=valid_data,
            mode='lines+markers',
            name=f'{metric_name}',
            line=dict(width=2)
        ),
        row=1, col=1
    )
    
    anomaly_times = [t for t, a in zip(times, anomalies) if a]
    anomaly_values = [v for v, a in zip(valid_data, anomalies) if a]
    if anomaly_times:
        fig.add_trace(
            go.Scatter(
                x=anomaly_times, y=anomaly_values,
                mode='markers',
                name='异常点',
                marker=dict(color='red', size=10, symbol='circle')
            ),
            row=1, col=1
        )
    
    if show_control_limits:
        fig.add_hline(
            y=center_line, line_dash="dash", line_color="green",
            annotation_text=f"中心线: {center_line:.4f} {unit}",
            row=1, col=1
        )
        fig.add_hline(
            y=upper_limit, line_dash="dash", line_color="orange",
            annotation_text=f"上控制限: {upper_limit:.4f} {unit}",
            row=1, col=1
        )
        fig.add_hline(
            y=lower_limit, line_dash="dash", line_color="orange",
            annotation_text=f"下控制限: {lower_limit:.4f} {unit}",
            row=1, col=1
        )
    
    if show_cusum:
        cumsum_pos, cumsum_neg, cusum_anomalies = cusum_detection(valid_data)
        
        fig.add_trace(
            go.Scatter(
                x=times, y=cumsum_pos,
                mode='lines',
                name='CUSUM+',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=times, y=cumsum_neg,
                mode='lines',
                name='CUSUM-',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        h_sigma = DEFAULT_CUSUM_H * np.nanstd(valid_data)
        fig.add_hline(
            y=h_sigma, line_dash="dash", line_color="red",
            annotation_text=f"决策限: {h_sigma:.4f}",
            row=2, col=1
        )
        fig.add_hline(
            y=-h_sigma, line_dash="dash", line_color="red",
            row=2, col=1
        )
    
    fig.update_layout(
        height=400 + (300 if show_cusum else 0),
        showlegend=True
    )
    
    fig.update_yaxes(title_text=f"{metric_name} ({unit})", row=1, col=1)
    if show_cusum:
        fig.update_yaxes(title_text="累积和", row=2, col=1)
    
    return fig
