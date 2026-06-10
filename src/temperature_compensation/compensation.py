import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from src.models.test_event import TestEvent
from src.models.modal_params import ModalParams
from config import DEFAULT_TEMP_MODEL_MIN_POINTS


@dataclass
class TemperatureModel:
    mode_index: int
    model_type: str
    coefficients: np.ndarray
    intercept: float
    r_squared: float
    temperature_data: np.ndarray
    frequency_data: np.ndarray
    
    def predict(self, temperature: float) -> float:
        if self.model_type == 'linear':
            return self.coefficients[0] * temperature + self.intercept
        elif self.model_type == 'quadratic':
            return (self.coefficients[0] * temperature ** 2 + 
                    self.coefficients[1] * temperature + 
                    self.intercept)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def compensate(self, temperature: float, measured_freq: float) -> float:
        predicted_freq = self.predict(temperature)
        residual = measured_freq - predicted_freq
        mean_freq = np.mean(self.frequency_data)
        return mean_freq + residual


def collect_temperature_frequency_pairs(
    events: List[TestEvent],
    modal_params_list: List[ModalParams],
    mode_index: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    temperatures = []
    frequencies = []
    valid_event_ids = []
    
    for event, params in zip(events, modal_params_list):
        if event.metadata.temperature is None:
            continue
        if mode_index >= len(params.mode_shapes):
            continue
        
        temperatures.append(event.metadata.temperature)
        frequencies.append(params.mode_shapes[mode_index].frequency)
        valid_event_ids.append(event.id)
    
    return np.array(temperatures), np.array(frequencies), valid_event_ids


def build_temperature_model(
    temperatures: np.ndarray,
    frequencies: np.ndarray,
    mode_index: int,
    model_type: str = 'linear',
    degree: int = 2
) -> Tuple[Optional[TemperatureModel], str]:
    n_points = len(temperatures)
    if n_points < DEFAULT_TEMP_MODEL_MIN_POINTS:
        return None, f"数据不足，暂无法建立温度模型（当前{n_points}组，需要至少{DEFAULT_TEMP_MODEL_MIN_POINTS}组）"
    
    X = temperatures.reshape(-1, 1)
    y = frequencies
    
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        return TemperatureModel(
            mode_index=mode_index,
            model_type='linear',
            coefficients=np.array([model.coef_[0]]),
            intercept=model.intercept_,
            r_squared=r2,
            temperature_data=temperatures,
            frequency_data=frequencies
        ), "建模成功"
    
    elif model_type == 'quadratic':
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        r2 = model.score(X_poly, y)
        return TemperatureModel(
            mode_index=mode_index,
            model_type='quadratic',
            coefficients=model.coef_,
            intercept=model.intercept_,
            r_squared=r2,
            temperature_data=temperatures,
            frequency_data=frequencies
        ), "建模成功"
    
    else:
        return None, f"不支持的模型类型: {model_type}"


def compensate_temperature_effect(
    modal_params: ModalParams,
    temperature: float,
    temperature_models: Dict[int, TemperatureModel]
) -> ModalParams:
    from copy import deepcopy
    
    compensated_params = deepcopy(modal_params)
    compensated_params.mode_shapes = []
    
    for mode_idx, mode_shape in enumerate(modal_params.mode_shapes):
        compensated_mode = deepcopy(mode_shape)
        
        if mode_idx in temperature_models:
            model = temperature_models[mode_idx]
            compensated_mode.frequency = model.compensate(temperature, mode_shape.frequency)
        
        compensated_params.mode_shapes.append(compensated_mode)
    
    return compensated_params


def compensate_temperature_pca(
    all_frequencies: np.ndarray,
    n_components: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    n_modes = all_frequencies.shape[1]
    
    mean_freq = np.mean(all_frequencies, axis=0)
    std_freq = np.std(all_frequencies, axis=0)
    std_freq = np.where(std_freq == 0, 1, std_freq)
    
    normalized = (all_frequencies - mean_freq) / std_freq
    
    pca = PCA(n_components=n_components)
    pca.fit(normalized)
    
    environmental_factor = pca.transform(normalized)
    
    reconstructed = pca.inverse_transform(environmental_factor)
    
    residuals = normalized - reconstructed
    
    compensated = residuals * std_freq + mean_freq
    
    return compensated, environmental_factor
