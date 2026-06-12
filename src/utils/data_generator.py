import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_station_data(station_name, capacity_mw, sunniness, anomaly_rate, num_days=180):
    """
    模拟生成单个电站的时序气象数据与运维事件数据。
    """
    np.random.seed(hash(station_name) % 2**32)
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(num_days * 24)]
    
    # 1. 模拟气象数据
    # 辐照度 (W/m^2): 日周期 + 天气波动 + 季节趋势
    irradiance = []
    temperature = []
    
    for t in timestamps:
        hour = t.hour
        day_of_year = t.timetuple().tm_yday
        
        # 基础日照强度 (正弦函数模拟白昼)
        if 6 <= hour <= 18:
            day_progress = (hour - 6) / 12.0
            base_irr = 1000 * np.sin(day_progress * np.pi)
        else:
            base_irr = 0
            
        # 天气噪声 (多云、阴天等波动)
        weather_factor = np.random.choice([1.0, 0.8, 0.5, 0.1], p=[0.6, 0.2, 0.1, 0.1])
        if sunniness == "high":
            # 晴朗地区，天气波动少
            weather_factor = np.random.choice([1.0, 0.9, 0.7], p=[0.8, 0.15, 0.05])
        elif sunniness == "low":
            # 多雨阴天地区
            weather_factor = np.random.choice([1.0, 0.7, 0.4, 0.1], p=[0.3, 0.4, 0.2, 0.1])
            
        irr = base_irr * weather_factor * (1 + np.random.normal(0, 0.05))
        irr = max(0, irr)
        
        # 温度 (Celsius): 与辐照度正相关，有滞后，伴随季节和日变化
        seasonal_temp = 15 + 10 * np.sin((day_of_year - 100) / 365.0 * 2 * np.pi)
        daily_temp_var = 5 * np.sin((hour - 9) / 24.0 * 2 * np.pi)
        temp = seasonal_temp + daily_temp_var + (irr * 0.01) + np.random.normal(0, 1.0)
        
        irradiance.append(irr)
        temperature.append(temp)
        
    irradiance = np.array(irradiance)
    temperature = np.array(temperature)
    
    # 2. 计算理论功率
    # 功率受辐照度影响，同时温度过高会导致效率下降
    temp_efficiency_loss = np.maximum(0, (temperature - 25) * 0.004) # 超过25度，效率降低
    theoretical_power = (irradiance / 1000.0) * capacity_mw * (1.0 - temp_efficiency_loss)
    theoretical_power = np.maximum(0, theoretical_power)
    
    # 3. 模拟离散运维事件 (O&M Events)
    # 0: Normal, 1: Scheduled Maintenance (计划检修), 2: Device Fault (设备故障), 3: Grid Curtailment (电网限电)
    event_types = np.zeros(len(timestamps), dtype=int)
    
    i = 0
    while i < len(timestamps):
        # 随机触发事件
        prob = np.random.rand()
        if prob < anomaly_rate * 0.2: # 计划检修：持续时间长，提前已知
            duration = np.random.randint(6, 15) # 持续6到15小时
            end_idx = min(i + duration, len(timestamps))
            event_types[i:end_idx] = 1
            i += duration
        elif prob < anomaly_rate * 0.5: # 设备故障：突发，部分功率损失
            duration = np.random.randint(2, 8) # 持续2到8小时
            end_idx = min(i + duration, len(timestamps))
            event_types[i:end_idx] = 2
            i += duration
        elif prob < anomaly_rate * 0.8: # 电网限电：突发或周期性
            duration = np.random.randint(4, 10)
            end_idx = min(i + duration, len(timestamps))
            event_types[i:end_idx] = 3
            i += duration
        else:
            i += 1
            
    # 4. 根据事件计算实际功率
    actual_power = theoretical_power.copy()
    for idx in range(len(timestamps)):
        event = event_types[idx]
        if event == 1: # 计划检修：断电全停
            actual_power[idx] = 0.0
        elif event == 2: # 设备故障：比如逆变器跳闸，损失60%功率
            actual_power[idx] *= 0.4
        elif event == 3: # 电网限电：最大功率限制在30%
            actual_power[idx] = min(actual_power[idx], capacity_mw * 0.25)
            
        # 加入微小系统噪声
        actual_power[idx] += np.random.normal(0, capacity_mw * 0.01)
        actual_power[idx] = max(0.0, actual_power[idx])
        
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Irradiance": irradiance,
        "Temperature": temperature,
        "Theoretical_Power": theoretical_power,
        "Event_Type": event_types,
        "Actual_Power": actual_power
    })
    
    return df

def main():
    # 创建 data 文件夹
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成三个不同特点的电站
    # 场站 A：西北晴朗地区，大容量，低故障率
    print("Generating data for Station A...")
    df_A = generate_station_data("Station_A", capacity_mw=100.0, sunniness="high", anomaly_rate=0.01)
    df_A.to_csv(os.path.join(data_dir, "station_A.csv"), index=False)
    
    # 场站 B：西南多云地区，中等容量，常态故障
    print("Generating data for Station B...")
    df_B = generate_station_data("Station_B", capacity_mw=50.0, sunniness="medium", anomaly_rate=0.02)
    df_B.to_csv(os.path.join(data_dir, "station_B.csv"), index=False)
    
    # 场站 C：南方多雨且设备老化电站，小容量，高故障与检修率
    print("Generating data for Station C...")
    df_C = generate_station_data("Station_C", capacity_mw=20.0, sunniness="low", anomaly_rate=0.04)
    df_C.to_csv(os.path.join(data_dir, "station_C.csv"), index=False)
    
    print("All simulated dataset files generated successfully in src/data/")

if __name__ == "__main__":
    main()
