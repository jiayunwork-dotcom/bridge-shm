import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict
from scipy.signal import find_peaks as scipy_find_peaks
from src.models.modal_params import ModalParams, ModeShape
from src.data_processing.preprocessing import get_window_function
from config import (
    DEFAULT_FFT_OVERLAP,
    DEFAULT_SINGULAR_VALUES_COUNT,
    DEFAULT_PEAK_HEIGHT_RATIO,
    DEFAULT_PEAK_DISTANCE_RATIO,
    DEFAULT_MAC_THRESHOLD
)


def get_default_fft_params(n_samples: int) -> Dict:
    window_length = int(np.ceil(n_samples / 8))
    window_length = int(2 ** np.ceil(np.log2(window_length)))
    return {
        'window_length': window_length,
        'overlap': DEFAULT_FFT_OVERLAP,
        'window_type': 'hann',
        'nfft': window_length
    }


def compute_cpsd_matrix(
    data: np.ndarray,
    sampling_rate: float,
    window_length: Optional[int] = None,
    overlap: float = DEFAULT_FFT_OVERLAP,
    window_type: str = 'hann',
    nfft: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_channels = data.shape
    
    if window_length is None:
        params = get_default_fft_params(n_samples)
        window_length = params['window_length']
    
    if nfft is None:
        nfft = window_length
    
    window = get_window_function(window_type, window_length)
    noverlap = int(window_length * overlap)
    
    n_freqs = nfft // 2 + 1
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sampling_rate)
    cpsd_matrix = np.zeros((n_channels, n_channels, n_freqs), dtype=np.complex128)
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            f, Pxy = signal.csd(
                data[:, i], data[:, j],
                fs=sampling_rate,
                window=window,
                nperseg=window_length,
                noverlap=noverlap,
                nfft=nfft,
                scaling='density',
                return_onesided=True
            )
            cpsd_matrix[i, j, :] = Pxy
            if i != j:
                cpsd_matrix[j, i, :] = np.conj(Pxy)
    
    return frequencies, cpsd_matrix


def compute_svd(
    cpsd_matrix: np.ndarray,
    n_singular_values: int = DEFAULT_SINGULAR_VALUES_COUNT
) -> Tuple[np.ndarray, np.ndarray]:
    n_channels, _, n_freqs = cpsd_matrix.shape
    singular_values = np.zeros((n_freqs, n_singular_values))
    mode_vectors = np.zeros((n_freqs, n_channels, n_singular_values), dtype=np.complex128)
    
    for k in range(n_freqs):
        G = cpsd_matrix[:, :, k]
        U, S, Vh = np.linalg.svd(G, full_matrices=False)
        singular_values[k, :] = S[:n_singular_values]
        mode_vectors[k, :, :] = U[:, :n_singular_values]
    
    return singular_values, mode_vectors


def find_peaks(
    frequencies: np.ndarray,
    singular_values: np.ndarray,
    height_ratio: float = DEFAULT_PEAK_HEIGHT_RATIO,
    distance_ratio: float = DEFAULT_PEAK_DISTANCE_RATIO
) -> Tuple[np.ndarray, np.ndarray]:
    freq_res = frequencies[1] - frequencies[0]
    max_sv = np.max(singular_values[:, 0])
    
    height = max_sv * height_ratio
    distance = int(freq_res * distance_ratio / freq_res)
    
    first_singular = singular_values[:, 0]
    peaks, properties = scipy_find_peaks(
        first_singular,
        height=height,
        distance=distance
    )
    
    return frequencies[peaks], peaks


def fdd_analysis(
    data: np.ndarray,
    sampling_rate: float,
    fft_params: Optional[Dict] = None,
    peak_params: Optional[Dict] = None,
    mac_threshold: float = DEFAULT_MAC_THRESHOLD,
    n_singular_values: int = DEFAULT_SINGULAR_VALUES_COUNT
) -> ModalParams:
    from .efdd import efdd_analysis
    
    n_samples, n_channels = data.shape
    
    if fft_params is None:
        fft_params = get_default_fft_params(n_samples)
    
    if peak_params is None:
        peak_params = {}
    
    frequencies, cpsd_matrix = compute_cpsd_matrix(
        data, sampling_rate,
        window_length=fft_params.get('window_length'),
        overlap=fft_params.get('overlap', DEFAULT_FFT_OVERLAP),
        window_type=fft_params.get('window_type', 'hann'),
        nfft=fft_params.get('nfft')
    )
    
    singular_values, mode_vectors = compute_svd(
        cpsd_matrix, n_singular_values=n_singular_values
    )
    
    peak_freqs, peak_indices = find_peaks(
        frequencies, singular_values,
        height_ratio=peak_params.get('height_ratio', DEFAULT_PEAK_HEIGHT_RATIO),
        distance_ratio=peak_params.get('distance_ratio', DEFAULT_PEAK_DISTANCE_RATIO)
    )
    
    mode_shapes = []
    for peak_idx, freq in zip(peak_indices, peak_freqs):
        mode_vec = np.real(mode_vectors[peak_idx, :, 0])
        mode_vec = mode_vec / np.max(np.abs(mode_vec))
        
        mode_shapes.append(ModeShape(
            frequency=freq,
            damping_ratio=0.0,
            mode_vector=mode_vec,
            mac_value=None
        ))
    
    params = {
        **fft_params,
        **peak_params,
        'mac_threshold': mac_threshold,
        'n_singular_values': n_singular_values
    }
    
    modal_params = ModalParams(
        event_id="",
        frequencies=frequencies,
        singular_values=singular_values,
        mode_shapes=mode_shapes,
        psd_matrix=cpsd_matrix,
        params=params
    )
    
    modal_params = efdd_analysis(
        modal_params, data, sampling_rate,
        mac_threshold=mac_threshold
    )
    
    return modal_params
