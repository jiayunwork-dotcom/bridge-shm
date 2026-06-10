import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_bridge_test_data(
    output_dir: str = "test_data",
    n_events: int = 5,
    n_channels: int = 8,
    sampling_rate: float = 200.0,
    duration: float = 10.0,
    base_frequencies: list = None,
    seed: int = 42
):
    """
    生成桥梁振动测试数据
    
    Args:
        output_dir: 输出目录
        n_events: 生成的测试事件数量
        n_channels: 通道数量
        sampling_rate: 采样率 (Hz)
        duration: 每个测试的时长 (秒)
        base_frequencies: 基频列表 (Hz)
        seed: 随机种子
    """
    np.random.seed(seed)
    
    if base_frequencies is None:
        base_frequencies = [2.5, 5.8, 9.2, 13.5]
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = int(sampling_rate * duration)
    t = np.arange(n_samples) / sampling_rate
    
    # 振型 (各通道的相对振幅)
    mode_shapes = []
    for mode_idx in range(len(base_frequencies)):
        shape = np.sin(np.pi * (mode_idx + 1) * np.arange(1, n_channels + 1) / (n_channels + 1))
        mode_shapes.append(shape / np.max(np.abs(shape)))
    
    # 阻尼比
    damping_ratios = [0.02, 0.015, 0.012, 0.01]
    
    base_time = datetime.now() - timedelta(days=n_events)
    
    for event_idx in range(n_events):
        # 模拟温度变化 (10-30°C)
        temperature = 10 + 20 * np.random.rand()
        
        # 温度对频率的影响 (温度升高, 频率降低)
        temp_factor = 1 - 0.0005 * (temperature - 20)
        
        # 随机损伤模拟 (后面的测试可能有微小损伤)
        damage_factor = 1.0
        if event_idx > n_events // 2:
            damage_factor = 1 - 0.003 * (event_idx - n_events // 2)
        
        # 生成数据
        data = np.zeros((n_samples, n_channels))
        
        for mode_idx, (base_freq, damp, shape) in enumerate(zip(base_frequencies, damping_ratios, mode_shapes)):
            freq = base_freq * temp_factor * damage_factor
            omega = 2 * np.pi * freq
            omega_d = omega * np.sqrt(1 - damp**2)
            phase = np.random.uniform(0, 2 * np.pi)
            
            decay = np.exp(-damp * omega * t)
            amplitude = 1.0 / (mode_idx + 1)
            
            signal = amplitude * decay * np.sin(omega_d * t + phase)
            
            for ch in range(n_channels):
                data[:, ch] += signal * shape[ch]
        
        # 添加噪声
        data += 0.05 * np.random.randn(n_samples, n_channels)
        
        # 创建DataFrame (包含时间列)
        df = pd.DataFrame(data, columns=[f'CH_{i+1:02d}' for i in range(n_channels)])
        df.insert(0, 'Time', t)
        
        # 生成文件名
        event_time = base_time + timedelta(days=event_idx)
        filename = f"bridge_test_{event_idx+1:03d}_{event_time.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # 保存CSV
        df.to_csv(filepath, index=False)
        
        # 生成元信息文件
        meta = {
            'event_id': f"event_{event_idx+1:03d}",
            'collection_time': event_time.isoformat(),
            'temperature': round(temperature, 1),
            'weather': ['sunny', 'cloudy', 'rain'][np.random.randint(0, 3)],
            'wind_speed': round(np.random.uniform(0, 10), 1),
            'traffic_status': ['normal', 'busy'][np.random.randint(0, 2)],
            'sampling_rate': sampling_rate,
            'duration': duration,
            'n_channels': n_channels,
            'description': f"测试数据 {event_idx+1}, 模拟温度 {temperature:.1f}°C"
        }
        
        meta_filepath = os.path.join(output_dir, f"meta_{event_idx+1:03d}.json")
        pd.Series(meta).to_json(meta_filepath, force_ascii=False, indent=2)
        
        print(f"生成测试文件: {filename}")
        print(f"  采样率: {sampling_rate} Hz, 通道数: {n_channels}, 时长: {duration}s")
        print(f"  温度: {temperature:.1f}°C, 损伤因子: {damage_factor:.4f}")
        print()
    
    print(f"\n✓ 成功生成 {n_events} 个测试数据文件，保存在: {output_dir}/")
    print("\n使用说明:")
    print("1. 启动系统: python app.py")
    print("2. 在浏览器中打开 http://localhost:8050")
    print("3. 先在'系统配置'中创建桥梁和测点")
    print("4. 在'数据导入'页面导入生成的CSV文件")
    print("5. 进行模态分析、损伤检测等操作")


if __name__ == "__main__":
    generate_bridge_test_data(
        output_dir="test_data",
        n_events=5,
        n_channels=8,
        sampling_rate=200.0,
        duration=10.0
    )
