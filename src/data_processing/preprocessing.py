import numpy as np
from scipy import signal
from typing import Tuple, Optional
import pandas as pd
from config import DEFAULT_FFT_OVERLAP


def get_window_function(window_type: str, n_points: int) -> np.ndarray:
    window_map = {
        'hann': signal.windows.hann,
        'hamming': signal.windows.hamming,
        'blackman': signal.windows.blackman,
        'rectangular': signal.windows.boxcar
    }
    if window_type not in window_map:
        raise ValueError(f"不支持的窗函数类型: {window_type}")
    return window_map[window_type](n_points)


def detrend_data(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    if method == 'linear':
        return signal.detrend(data, axis=0, type='linear')
    elif method == 'constant':
        return signal.detrend(data, axis=0, type='constant')
    else:
        raise ValueError(f"不支持的去趋势方法: {method}")


def anti_aliasing_filter(
    data: np.ndarray,
    sampling_rate: float,
    cutoff_freq: Optional[float] = None,
    order: int = 4
) -> np.ndarray:
    if cutoff_freq is None:
        cutoff_freq = sampling_rate * 0.4
    
    nyquist = sampling_rate / 2.0
    if cutoff_freq >= nyquist:
        raise ValueError(f"截止频率({cutoff_freq})必须小于奈奎斯特频率({nyquist})")
    
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    
    filtered_data = np.zeros_like(data)
    if data.ndim == 1:
        filtered_data = signal.filtfilt(b, a, data)
    else:
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
    
    return filtered_data


def resample_data(
    data: np.ndarray,
    original_sr: float,
    target_sr: float
) -> Tuple[np.ndarray, float]:
    if original_sr == target_sr:
        return data, target_sr
    
    ratio = target_sr / original_sr
    n_samples = len(data)
    n_target = int(n_samples * ratio)
    
    resampled = np.zeros((n_target, data.shape[1])) if data.ndim > 1 else np.zeros(n_target)
    
    if data.ndim == 1:
        resampled = signal.resample(data, n_target)
    else:
        for i in range(data.shape[1]):
            resampled[:, i] = signal.resample(data[:, i], n_target)
    
    return resampled, target_sr


def preprocess_pipeline(
    df: pd.DataFrame,
    sampling_rate: float,
    detrend: bool = True,
    detrend_method: str = 'linear',
    filter: bool = True,
    cutoff_freq: Optional[float] = None,
    resample: bool = False,
    target_sr: Optional[float] = None
) -> Tuple[pd.DataFrame, float, dict]:
    data = df.values
    current_sr = sampling_rate
    params = {}
    
    if detrend:
        data = detrend_data(data, method=detrend_method)
        params['detrend'] = {'method': detrend_method}
    
    if filter:
        data = anti_aliasing_filter(data, current_sr, cutoff_freq)
        params['filter'] = {'cutoff_freq': cutoff_freq or current_sr * 0.4, 'order': 4}
    
    if resample and target_sr is not None and target_sr != current_sr:
        data, current_sr = resample_data(data, sampling_rate, target_sr)
        params['resample'] = {'original_sr': sampling_rate, 'target_sr': target_sr}
    
    processed_df = pd.DataFrame(data, columns=df.columns)
    params['final_sampling_rate'] = current_sr
    
    return processed_df, current_sr, params
