import numpy as np
from typing import Tuple, Dict, Optional
from src.models.modal_params import ModalParams, ModeShape
from src.models.damage_index import DamageIndex, DamageType
from src.models.bridge import Bridge, SensorType
from config import DEFAULT_FLEXIBILITY_MODES


def compute_frequency_change_rate(
    current_params: ModalParams,
    baseline_params: ModalParams,
    temperature_compensated: bool = False
) -> DamageIndex:
    current_freqs = current_params.get_mode_frequencies()
    baseline_freqs = baseline_params.get_mode_frequencies()
    
    n_modes = min(len(current_freqs), len(baseline_freqs))
    current_freqs = current_freqs[:n_modes]
    baseline_freqs = baseline_freqs[:n_modes]
    
    change_rates = np.where(
        baseline_freqs != 0,
        (current_freqs - baseline_freqs) / baseline_freqs * 100,
        0.0
    )
    
    threshold = -2.0
    
    return DamageIndex(
        event_id=current_params.event_id,
        baseline_event_id=baseline_params.event_id,
        damage_type=DamageType.FREQUENCY_CHANGE,
        values=change_rates,
        threshold=threshold,
        locations=[f"第{i+1}阶模态" for i in range(n_modes)],
        temperature_compensated=temperature_compensated
    )


def compute_flexibility_matrix(
    modal_params: ModalParams,
    n_modes: int = DEFAULT_FLEXIBILITY_MODES
) -> np.ndarray:
    mode_shapes = modal_params.mode_shapes
    n_modes_available = len(mode_shapes)
    n_modes_used = min(n_modes, n_modes_available)
    
    if n_modes_used == 0:
        return np.array([])
    
    n_channels = len(mode_shapes[0].mode_vector)
    flexibility = np.zeros((n_channels, n_channels))
    
    for i in range(n_modes_used):
        mode = mode_shapes[i]
        omega = 2 * np.pi * mode.frequency
        phi = mode.mode_vector.reshape(-1, 1)
        flexibility += (phi @ phi.T) / (omega ** 2)
    
    return flexibility


def compute_flexibility_change(
    current_params: ModalParams,
    baseline_params: ModalParams,
    n_modes: int = DEFAULT_FLEXIBILITY_MODES,
    temperature_compensated: bool = False
) -> DamageIndex:
    current_flex = compute_flexibility_matrix(current_params, n_modes)
    baseline_flex = compute_flexibility_matrix(baseline_params, n_modes)
    
    if current_flex.size == 0 or baseline_flex.size == 0:
        return DamageIndex(
            event_id=current_params.event_id,
            baseline_event_id=baseline_params.event_id,
            damage_type=DamageType.FLEXIBILITY_CHANGE,
            values=np.array([]),
            temperature_compensated=temperature_compensated
        )
    
    n_channels = current_flex.shape[0]
    flexibility_change = np.diag(current_flex - baseline_flex)
    
    normalized_change = np.where(
        np.abs(np.diag(baseline_flex)) > 1e-10,
        flexibility_change / np.abs(np.diag(baseline_flex)) * 100,
        0.0
    )
    
    threshold = 10.0
    
    return DamageIndex(
        event_id=current_params.event_id,
        baseline_event_id=baseline_params.event_id,
        damage_type=DamageType.FLEXIBILITY_CHANGE,
        values=normalized_change,
        threshold=threshold,
        locations=[f"测点{i+1}" for i in range(n_channels)],
        temperature_compensated=temperature_compensated
    )


def compute_curvature_mode(
    mode_shape: ModeShape,
    sensor_spacing: float = 1.0
) -> np.ndarray:
    phi = mode_shape.mode_vector
    h = sensor_spacing
    
    curvature = np.zeros_like(phi)
    for i in range(1, len(phi) - 1):
        curvature[i] = (phi[i-1] - 2 * phi[i] + phi[i+1]) / (h ** 2)
    
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    
    return curvature


def compute_curvature_mode_change(
    current_params: ModalParams,
    baseline_params: ModalParams,
    bridge: Bridge,
    n_modes: int = DEFAULT_FLEXIBILITY_MODES,
    temperature_compensated: bool = False
) -> DamageIndex:
    accel_sensors = bridge.get_sensors_by_type(SensorType.ACCELERATION)
    accel_sensors.sort(key=lambda s: s.location[0])
    
    if len(accel_sensors) < 3:
        return DamageIndex(
            event_id=current_params.event_id,
            baseline_event_id=baseline_params.event_id,
            damage_type=DamageType.CURVATURE_MODE,
            values=np.array([]),
            temperature_compensated=temperature_compensated
        )
    
    x_coords = np.array([s.location[0] for s in accel_sensors])
    sensor_spacing = np.median(np.diff(x_coords))
    if sensor_spacing <= 0:
        sensor_spacing = 1.0
    
    n_modes_current = len(current_params.mode_shapes)
    n_modes_baseline = len(baseline_params.mode_shapes)
    n_modes_used = min(n_modes, n_modes_current, n_modes_baseline)
    
    if n_modes_used == 0:
        return DamageIndex(
            event_id=current_params.event_id,
            baseline_event_id=baseline_params.event_id,
            damage_type=DamageType.CURVATURE_MODE,
            values=np.array([]),
            temperature_compensated=temperature_compensated
        )
    
    n_channels = len(accel_sensors)
    curvature_change_sum = np.zeros(n_channels)
    
    for i in range(n_modes_used):
        current_mode = current_params.mode_shapes[i]
        baseline_mode = baseline_params.mode_shapes[i]
        
        current_vec = current_mode.mode_vector[:n_channels]
        baseline_vec = baseline_mode.mode_vector[:n_channels]
        
        if len(current_vec) < n_channels:
            current_vec = np.pad(current_vec, (0, n_channels - len(current_vec)))
        if len(baseline_vec) < n_channels:
            baseline_vec = np.pad(baseline_vec, (0, n_channels - len(baseline_vec)))
        
        current_curvature = compute_curvature_mode(
            ModeShape(frequency=current_mode.frequency, damping_ratio=0, mode_vector=current_vec),
            sensor_spacing
        )
        baseline_curvature = compute_curvature_mode(
            ModeShape(frequency=baseline_mode.frequency, damping_ratio=0, mode_vector=baseline_vec),
            sensor_spacing
        )
        
        curvature_change_sum += np.abs(current_curvature - baseline_curvature)
    
    curvature_change = curvature_change_sum / n_modes_used
    
    threshold = np.mean(curvature_change) + 2 * np.std(curvature_change) if len(curvature_change) > 0 else None
    
    return DamageIndex(
        event_id=current_params.event_id,
        baseline_event_id=baseline_params.event_id,
        damage_type=DamageType.CURVATURE_MODE,
        values=curvature_change,
        threshold=threshold,
        locations=[f"测点{i+1}" for i in range(n_channels)],
        temperature_compensated=temperature_compensated
    )


def compute_modal_strain_energy(
    current_params: ModalParams,
    baseline_params: ModalParams,
    bridge: Bridge,
    n_modes: int = DEFAULT_FLEXIBILITY_MODES,
    temperature_compensated: bool = False
) -> DamageIndex:
    from src.models.bridge import SensorType
    
    accel_sensors = bridge.get_sensors_by_type(SensorType.ACCELERATION)
    accel_sensors.sort(key=lambda s: s.location[0])
    
    if len(accel_sensors) < 2:
        return DamageIndex(
            event_id=current_params.event_id,
            baseline_event_id=baseline_params.event_id,
            damage_type=DamageType.MODAL_STRAIN_ENERGY,
            values=np.array([]),
            temperature_compensated=temperature_compensated
        )
    
    n_channels = len(accel_sensors)
    x_coords = np.array([s.location[0] for s in accel_sensors])
    
    n_modes_current = len(current_params.mode_shapes)
    n_modes_baseline = len(baseline_params.mode_shapes)
    n_modes_used = min(n_modes, n_modes_current, n_modes_baseline)
    
    if n_modes_used == 0:
        return DamageIndex(
            event_id=current_params.event_id,
            baseline_event_id=baseline_params.event_id,
            damage_type=DamageType.MODAL_STRAIN_ENERGY,
            values=np.array([]),
            temperature_compensated=temperature_compensated
        )
    
    mse_change = np.zeros(n_channels)
    
    for i in range(n_modes_used):
        current_mode = current_params.mode_shapes[i]
        baseline_mode = baseline_params.mode_shapes[i]
        
        current_vec = current_mode.mode_vector[:n_channels]
        baseline_vec = baseline_mode.mode_vector[:n_channels]
        
        if len(current_vec) < n_channels:
            current_vec = np.pad(current_vec, (0, n_channels - len(current_vec)))
        if len(baseline_vec) < n_channels:
            baseline_vec = np.pad(baseline_vec, (0, n_channels - len(baseline_vec)))
        
        dphi_current = np.gradient(current_vec, x_coords)
        dphi_baseline = np.gradient(baseline_vec, x_coords)
        
        current_mse = dphi_current ** 2
        baseline_mse = dphi_baseline ** 2
        
        mode_mse_change = np.where(
            baseline_mse > 1e-10,
            (current_mse - baseline_mse) / baseline_mse * 100,
            0.0
        )
        
        mse_change += mode_mse_change
    
    mse_change = mse_change / n_modes_used
    
    threshold = 20.0
    
    return DamageIndex(
        event_id=current_params.event_id,
        baseline_event_id=baseline_params.event_id,
        damage_type=DamageType.MODAL_STRAIN_ENERGY,
        values=mse_change,
        threshold=threshold,
        locations=[f"测点{i+1}" for i in range(n_channels)],
        temperature_compensated=temperature_compensated
    )


def compute_all_damage_indices(
    current_params: ModalParams,
    baseline_params: ModalParams,
    bridge: Bridge,
    temperature_compensated: bool = False,
    n_modes: int = DEFAULT_FLEXIBILITY_MODES
) -> Dict[DamageType, DamageIndex]:
    indices = {}
    
    indices[DamageType.FREQUENCY_CHANGE] = compute_frequency_change_rate(
        current_params, baseline_params, temperature_compensated
    )
    
    indices[DamageType.FLEXIBILITY_CHANGE] = compute_flexibility_change(
        current_params, baseline_params, n_modes, temperature_compensated
    )
    
    indices[DamageType.CURVATURE_MODE] = compute_curvature_mode_change(
        current_params, baseline_params, bridge, n_modes, temperature_compensated
    )
    
    indices[DamageType.MODAL_STRAIN_ENERGY] = compute_modal_strain_energy(
        current_params, baseline_params, bridge, n_modes, temperature_compensated
    )
    
    return indices
