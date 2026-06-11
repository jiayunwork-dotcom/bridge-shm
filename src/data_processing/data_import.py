import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Optional
from config import MAX_FILE_SIZE


def check_file_size(filepath: str) -> Tuple[bool, str]:
    if not os.path.exists(filepath):
        return False, "文件不存在"
    file_size = os.path.getsize(filepath)
    if file_size > MAX_FILE_SIZE:
        return False, f"文件大小({file_size/1024/1024:.1f}MB)超过限制200MB，请分段导入"
    return True, "文件大小符合要求"


def detect_channels(df: pd.DataFrame) -> int:
    return df.shape[1]


def detect_sampling_rate(df: pd.DataFrame) -> Optional[float]:
    if df.shape[1] < 1:
        return None
    
    first_col = df.columns[0]
    if first_col.lower() in ['time', 't', 'timestamp']:
        time_col = df[first_col].values
        if len(time_col) > 1:
            dt = np.median(np.diff(time_col))
            if dt > 0:
                return 1.0 / dt
    return None


def import_csv(
    filepath: str,
    has_time_column: bool = True,
    custom_sampling_rate: Optional[float] = None,
    encoding: str = 'utf-8'
) -> Tuple[pd.DataFrame, float, List[str]]:
    valid, msg = check_file_size(filepath)
    if not valid:
        raise ValueError(msg)
    
    df = pd.read_csv(filepath, encoding=encoding)
    
    sampling_rate = custom_sampling_rate
    if sampling_rate is None and has_time_column:
        sampling_rate = detect_sampling_rate(df)
    
    if has_time_column and df.shape[1] > 0:
        first_col = df.columns[0]
        if first_col.lower() in ['time', 't', 'timestamp']:
            df = df.drop(columns=[first_col])
    
    channel_names = list(df.columns)
    n_channels = detect_channels(df)
    
    if sampling_rate is None:
        raise ValueError("无法自动检测采样率，请手动指定采样频率")
    
    if sampling_rate < 100 or sampling_rate > 1000:
        raise ValueError(f"采样率{sampling_rate:.1f}Hz不在有效范围(100Hz-1000Hz)内")
    
    return df, sampling_rate, channel_names
