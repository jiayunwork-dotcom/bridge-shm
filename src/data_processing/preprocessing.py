import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
from typing import Tuple, Optional, List
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


def detrend_data(data: np.ndarray, method: str = 'linear', poly_order: int = 3) -> np.ndarray:
    if method == 'linear':
        return signal.detrend(data, axis=0, type='linear')
    elif method == 'constant':
        return signal.detrend(data, axis=0, type='constant')
    elif method == 'polynomial':
        if data.ndim == 1:
            n = len(data)
            x = np.arange(n)
            coeffs = np.polyfit(x, data, poly_order)
            trend = np.polyval(coeffs, x)
            return data - trend
        else:
            n, m = data.shape
            x = np.arange(n)
            detrended = np.zeros_like(data)
            for i in range(m):
                coeffs = np.polyfit(x, data[:, i], poly_order)
                trend = np.polyval(coeffs, x)
                detrended[:, i] = data[:, i] - trend
            return detrended
    elif method == 'none':
        return data
    else:
        raise ValueError(f"不支持的去趋势方法: {method}")


def filter_data(
    data: np.ndarray,
    sampling_rate: float,
    filter_type: str = 'low',
    cutoff_freq: Optional[float] = None,
    cutoff_freq2: Optional[float] = None,
    order: int = 4
) -> np.ndarray:
    if filter_type == 'none':
        return data
    
    nyquist = sampling_rate / 2.0
    
    if filter_type in ['low', 'high']:
        if cutoff_freq is None:
            cutoff_freq = sampling_rate * 0.4
        if cutoff_freq >= nyquist:
            raise ValueError(f"截止频率({cutoff_freq})必须小于奈奎斯特频率({nyquist})")
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype=filter_type, analog=False)
    elif filter_type == 'band':
        if cutoff_freq is None or cutoff_freq2 is None:
            raise ValueError("带通滤波需要指定两个截止频率")
        if cutoff_freq >= cutoff_freq2:
            raise ValueError("带通滤波的下限截止频率必须小于上限截止频率")
        if cutoff_freq2 >= nyquist:
            raise ValueError(f"上限截止频率({cutoff_freq2})必须小于奈奎斯特频率({nyquist})")
        low = cutoff_freq / nyquist
        high = cutoff_freq2 / nyquist
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
    else:
        raise ValueError(f"不支持的滤波类型: {filter_type}")
    
    filtered_data = np.zeros_like(data)
    if data.ndim == 1:
        filtered_data = signal.filtfilt(b, a, data)
    else:
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
    
    return filtered_data


def anti_aliasing_filter(data, sampling_rate, cutoff_freq=None, order=4):
    return filter_data(data, sampling_rate, filter_type='low', cutoff_freq=cutoff_freq, order=order)


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
    detrend_method: str = 'linear',
    poly_order: int = 3,
    filter_type: str = 'none',
    cutoff_freq: Optional[float] = None,
    cutoff_freq2: Optional[float] = None,
    target_sr: Optional[float] = None
) -> Tuple[pd.DataFrame, float, dict]:
    data = df.values
    current_sr = sampling_rate
    params = {}
    
    if detrend_method != 'none':
        data = detrend_data(data, method=detrend_method, poly_order=poly_order)
        params['detrend'] = {'method': detrend_method, 'poly_order': poly_order}
    
    if filter_type != 'none':
        data = filter_data(data, current_sr, filter_type=filter_type, 
                           cutoff_freq=cutoff_freq, cutoff_freq2=cutoff_freq2)
        params['filter'] = {
            'type': filter_type,
            'cutoff_freq': cutoff_freq,
            'cutoff_freq2': cutoff_freq2,
            'order': 4
        }
    
    if target_sr is not None and target_sr != current_sr:
        data, current_sr = resample_data(data, sampling_rate, target_sr)
        params['resample'] = {'original_sr': sampling_rate, 'target_sr': target_sr}
    
    processed_df = pd.DataFrame(data, columns=df.columns)
    params['final_sampling_rate'] = current_sr
    
    return processed_df, current_sr, params
