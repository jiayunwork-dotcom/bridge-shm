import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional
from src.models.bridge import Bridge, Sensor, SensorType
from src.models.modal_params import ModalParams, ModeShape
from src.modal_analysis.efdd import compute_mac_matrix


def generate_mode_animation_frames(
    mode_vector: np.ndarray,
    locations: np.ndarray,
    n_frames: int = 30,
    amplitude: float = 1.0
) -> List[np.ndarray]:
    frames = []
    for i in range(n_frames):
        phase = 2 * np.pi * i / n_frames
        displacement = amplitude * mode_vector * np.sin(phase)
        frame = locations.copy()
        if locations.shape[1] >= 3:
            frame[:, 2] += displacement
        else:
            frame[:, 1] += displacement
        frames.append(frame)
    return frames


def create_mode_shape_animation_2d(
    bridge: Bridge,
    mode_shape: ModeShape,
    amplitude: float = 1.0,
    n_frames: int = 30
) -> go.Figure:
    accel_sensors = bridge.get_sensors_by_type(SensorType.ACCELERATION)
    accel_sensors.sort(key=lambda s: s.location[0])
    
    locations = np.array([s.location[:2] for s in accel_sensors])
    mode_vec = mode_shape.mode_vector
    
    if len(mode_vec) != len(accel_sensors):
        if len(mode_vec) > len(accel_sensors):
            mode_vec = mode_vec[:len(accel_sensors)]
        else:
            mode_vec = np.pad(mode_vec, (0, len(accel_sensors) - len(mode_vec)))
    
    frames = generate_mode_animation_frames(mode_vec, locations, n_frames, amplitude)
    
    fig = go.Figure()
    
    x_static = locations[:, 0]
    y_static = locations[:, 1]
    
    fig.add_trace(go.Scatter(
        x=x_static, y=y_static,
        mode='lines+markers',
        name='桥梁结构',
        line=dict(color='blue', width=2),
        marker=dict(size=8, color='blue')
    ))
    
    for i, frame in enumerate(frames):
        fig.add_trace(go.Scatter(
            x=frame[:, 0], y=frame[:, 1],
            mode='lines+markers',
            name=f'振型 (相位{i})',
            line=dict(color='red', width=2),
            marker=dict(size=8, color='red'),
            visible=(i == 0)
        ))
    
    steps = []
    for i in range(n_frames):
        step = dict(
            method="update",
            args=[{"visible": [True] + [j == i for j in range(n_frames)]}],
            label=f"帧 {i+1}"
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "相位: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        title=f"第1阶振型 (频率: {mode_shape.frequency:.3f} Hz, 阻尼: {mode_shape.damping_ratio*100:.4f}%)",
        xaxis_title="X坐标 (m)",
        yaxis_title="Y坐标 (m)",
        sliders=sliders,
        showlegend=True,
        height=500
    )
    
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )
    
    return fig


def create_mode_shape_animation_3d(
    bridge: Bridge,
    mode_shape: ModeShape,
    amplitude: float = 1.0,
    n_frames: int = 30
) -> go.Figure:
    accel_sensors = bridge.get_sensors_by_type(SensorType.ACCELERATION)
    accel_sensors.sort(key=lambda s: s.location[0])
    
    locations = np.array([s.location for s in accel_sensors])
    mode_vec = mode_shape.mode_vector
    
    if len(mode_vec) != len(accel_sensors):
        if len(mode_vec) > len(accel_sensors):
            mode_vec = mode_vec[:len(accel_sensors)]
        else:
            mode_vec = np.pad(mode_vec, (0, len(accel_sensors) - len(mode_vec)))
    
    frames = generate_mode_animation_frames(mode_vec, locations, n_frames, amplitude)
    
    fig = go.Figure()
    
    x_static = locations[:, 0]
    y_static = locations[:, 1]
    z_static = locations[:, 2]
    
    fig.add_trace(go.Scatter3d(
        x=x_static, y=y_static, z=z_static,
        mode='lines+markers',
        name='桥梁结构',
        line=dict(color='blue', width=3),
        marker=dict(size=6, color='blue')
    ))
    
    for i, frame in enumerate(frames):
        fig.add_trace(go.Scatter3d(
            x=frame[:, 0], y=frame[:, 1], z=frame[:, 2],
            mode='lines+markers',
            name=f'振型变形',
            line=dict(color='red', width=3),
            marker=dict(size=6, color='red'),
            visible=(i == 0)
        ))
    
    steps = []
    for i in range(n_frames):
        step = dict(
            method="update",
            args=[{"visible": [True] + [j == i for j in range(n_frames)]}],
            label=f"帧 {i+1}"
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "相位: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        title=f"第1阶振型 3D展示 (频率: {mode_shape.frequency:.3f} Hz, 阻尼: {mode_shape.damping_ratio*100:.4f}%)",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        sliders=sliders,
        showlegend=True,
        height=600
    )
    
    return fig


def create_multi_mode_comparison(
    bridge: Bridge,
    modal_params: ModalParams,
    mode_indices: Optional[List[int]] = None
) -> go.Figure:
    if mode_indices is None:
        mode_indices = list(range(min(3, len(modal_params.mode_shapes))))
    
    n_modes = len(mode_indices)
    if n_modes == 0:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=n_modes,
        subplot_titles=[f"第{i+1}阶模态" for i in mode_indices],
        specs=[[{'type': 'scatter'}] * n_modes]
    )
    
    accel_sensors = bridge.get_sensors_by_type(SensorType.ACCELERATION)
    accel_sensors.sort(key=lambda s: s.location[0])
    x_coords = np.array([s.location[0] for s in accel_sensors])
    
    for plot_idx, mode_idx in enumerate(mode_indices):
        if mode_idx >= len(modal_params.mode_shapes):
            continue
        
        mode_shape = modal_params.mode_shapes[mode_idx]
        mode_vec = mode_shape.mode_vector
        
        if len(mode_vec) != len(accel_sensors):
            if len(mode_vec) > len(accel_sensors):
                mode_vec = mode_vec[:len(accel_sensors)]
            else:
                mode_vec = np.pad(mode_vec, (0, len(accel_sensors) - len(mode_vec)))
        
        fig.add_trace(
            go.Scatter(
                x=x_coords, y=mode_vec,
                mode='lines+markers',
                name=f'频率: {mode_shape.frequency:.3f} Hz<br>阻尼: {mode_shape.damping_ratio*100:.4f}%',
                line=dict(width=2),
                marker=dict(size=8)
            ),
            row=1, col=plot_idx + 1
        )
    
    fig.update_layout(
        title="多阶模态参数对比",
        height=400,
        showlegend=True
    )
    
    for i in range(n_modes):
        fig.update_xaxes(title_text="X坐标 (m)", row=1, col=i+1)
        fig.update_yaxes(title_text="振型幅值", row=1, col=i+1)
    
    return fig


def create_mac_heatmap(
    modal_params1: ModalParams,
    modal_params2: Optional[ModalParams] = None,
    event1_name: str = "测试1",
    event2_name: str = "测试2"
) -> go.Figure:
    if modal_params2 is None:
        modes1 = modal_params1.get_mode_matrix()
        modes2 = modes1
        title = "MAC矩阵 (自相关)"
    else:
        modes1 = modal_params1.get_mode_matrix()
        modes2 = modal_params2.get_mode_matrix()
        title = f"MAC矩阵 ({event1_name} vs {event2_name})"
    
    if modes1.size == 0 or modes2.size == 0:
        fig = go.Figure()
        fig.update_layout(title="MAC矩阵: 无有效数据")
        return fig
    
    mac_matrix = compute_mac_matrix(modes1, modes2)
    
    n_modes1 = mac_matrix.shape[0]
    n_modes2 = mac_matrix.shape[1]
    
    x_labels = [f"模态{i+1}" for i in range(n_modes2)]
    y_labels = [f"模态{i+1}" for i in range(n_modes1)]
    
    fig = go.Figure(data=go.Heatmap(
        z=mac_matrix,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlBu_r',
        zmin=0, zmax=1,
        text=[[f"{val:.3f}" for val in row] for row in mac_matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="%{y}<br>%{x}<br>MAC: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=event2_name + " 模态",
        yaxis_title=event1_name + " 模态",
        height=400 + 30 * max(n_modes1, n_modes2)
    )
    
    return fig
