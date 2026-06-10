import numpy as np
from scipy.fft import ifft
from typing import Tuple, Optional
from sklearn.linear_model import LinearRegression
from src.models.modal_params import ModalParams, ModeShape
from config import (
    DEFAULT_MAC_THRESHOLD,
    DEFAULT_EFDD_MIN_POINTS,
    DEFAULT_DAMPING_R2_THRESHOLD
)


def compute_mac(phi1: np.ndarray, phi2: np.ndarray) -> float:
    phi1 = np.asarray(phi1, dtype=np.complex128)
    phi2 = np.asarray(phi2, dtype=np.complex128)
    
    numerator = np.abs(np.vdot(phi1, phi2)) ** 2
    denominator = np.abs(np.vdot(phi1, phi1)) * np.abs(np.vdot(phi2, phi2))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_mac_matrix(modes1: np.ndarray, modes2: np.ndarray) -> np.ndarray:
    n_modes1 = modes1.shape[1] if modes1.ndim > 1 else 1
    n_modes2 = modes2.shape[1] if modes2.ndim > 1 else 1
    
    if modes1.ndim == 1:
        modes1 = modes1.reshape(-1, 1)
    if modes2.ndim == 1:
        modes2 = modes2.reshape(-1, 1)
    
    mac_matrix = np.zeros((n_modes1, n_modes2))
    for i in range(n_modes1):
        for j in range(n_modes2):
            mac_matrix[i, j] = compute_mac(modes1[:, i], modes2[:, j])
    
    return mac_matrix


def extract_free_decay(
    frequencies: np.ndarray,
    singular_values: np.ndarray,
    mode_vectors: np.ndarray,
    peak_idx: int,
    peak_vec: np.ndarray,
    mac_threshold: float = DEFAULT_MAC_THRESHOLD,
    min_points: int = DEFAULT_EFDD_MIN_POINTS
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    n_freqs = len(frequencies)
    
    left_idx = peak_idx
    while left_idx > 0:
        mac = compute_mac(mode_vectors[left_idx - 1, :, 0], peak_vec)
        if mac < mac_threshold:
            break
        left_idx -= 1
    
    right_idx = peak_idx
    while right_idx < n_freqs - 1:
        mac = compute_mac(mode_vectors[right_idx + 1, :, 0], peak_vec)
        if mac < mac_threshold:
            break
        right_idx += 1
    
    if (right_idx - left_idx + 1) < min_points:
        half = min_points // 2
        left_idx = max(0, peak_idx - half)
        right_idx = min(n_freqs - 1, peak_idx + half)
    
    freq_segment = frequencies[left_idx:right_idx + 1]
    sv_segment = singular_values[left_idx:right_idx + 1, 0]
    
    return freq_segment, sv_segment, left_idx, right_idx


def compute_damping_ratio(
    frequencies: np.ndarray,
    singular_values: np.ndarray,
    sampling_rate: float,
    nfft: int
) -> Tuple[float, float, str]:
    n_points = len(frequencies)
    
    ifft_input = np.zeros(nfft, dtype=np.complex128)
    start_idx = np.argmin(np.abs(frequencies - frequencies[0]))
    
    for i, sv in enumerate(singular_values):
        if start_idx + i < nfft // 2 + 1:
            ifft_input[start_idx + i] = sv
    
    ifft_input[nfft // 2 + 1:] = np.conj(ifft_input[nfft // 2 - 1:0:-1])
    
    free_decay = np.real(ifft(ifft_input))
    free_decay = free_decay[:len(free_decay) // 2]
    
    envelope = np.abs(free_decay)
    
    peak_idx = np.where((envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] > envelope[2:]))[0] + 1
    
    if len(peak_idx) < 5:
        return 0.0, 0.0, "poor"
    
    log_env = np.log(envelope[peak_idx])
    times = peak_idx / sampling_rate
    
    X = times.reshape(-1, 1)
    y = log_env.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    r2 = model.score(X, y)
    slope = model.coef_[0][0]
    
    quality = "good" if r2 >= DEFAULT_DAMPING_R2_THRESHOLD else "poor"
    
    peak_freq = np.median(frequencies)
    damping_ratio = -slope / (2 * np.pi * peak_freq)
    damping_ratio = max(0, damping_ratio)
    
    return damping_ratio, r2, quality


def efdd_analysis(
    modal_params: ModalParams,
    data: np.ndarray,
    sampling_rate: float,
    mac_threshold: float = DEFAULT_MAC_THRESHOLD
) -> ModalParams:
    from .fdd import compute_cpsd_matrix, compute_svd
    
    n_samples, n_channels = data.shape
    
    fft_params = modal_params.params
    nfft = fft_params.get('nfft', 1024)
    window_length = fft_params.get('window_length', nfft)
    overlap = fft_params.get('overlap', 0.5)
    window_type = fft_params.get('window_type', 'hann')
    
    frequencies, cpsd_matrix = compute_cpsd_matrix(
        data, sampling_rate,
        window_length=window_length,
        overlap=overlap,
        window_type=window_type,
        nfft=nfft
    )
    
    singular_values_full, mode_vectors_full = compute_svd(cpsd_matrix)
    
    updated_mode_shapes = []
    for mode_shape in modal_params.mode_shapes:
        target_freq = mode_shape.frequency
        peak_idx = np.argmin(np.abs(frequencies - target_freq))
        peak_vec = mode_vectors_full[peak_idx, :, 0]
        
        freq_segment, sv_segment, left_idx, right_idx = extract_free_decay(
            frequencies, singular_values_full, mode_vectors_full,
            peak_idx, peak_vec, mac_threshold=mac_threshold
        )
        
        damping_ratio, r2, quality = compute_damping_ratio(
            freq_segment, sv_segment, sampling_rate, nfft
        )
        
        updated_mode_shape = ModeShape(
            frequency=target_freq,
            damping_ratio=damping_ratio,
            mode_vector=mode_shape.mode_vector,
            mac_value=compute_mac(peak_vec, mode_shape.mode_vector),
            damping_quality=quality
        )
        updated_mode_shapes.append(updated_mode_shape)
    
    modal_params.mode_shapes = updated_mode_shapes
    
    return modal_params
