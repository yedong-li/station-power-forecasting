import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class PowerStationDataset(Dataset):
    def __init__(self, csv_path, history_len=24, forecast_len=24, train_ratio=0.8, is_train=True, scaler=None):
        """
        电站时序数据集
        csv_path: CSV文件路径
        history_len: 历史窗口长度 (T)
        forecast_len: 预测未来窗口长度 (H)
        train_ratio: 训练集比例
        is_train: 是否是训练集
        scaler: 用于特征归一化的 StandardScalers 字典。如果为 None 且是训练集，则计算并创建。
        """
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.is_train = is_train
        
        # 读取CSV数据
        df = pd.read_csv(csv_path)
        
        # 划分训练集和测试集 (按时间顺序)
        split_idx = int(len(df) * train_ratio)
        if self.is_train:
            self.data = df.iloc[:split_idx].reset_index(drop=True)
        else:
            self.data = df.iloc[split_idx:].reset_index(drop=True)
            
        # 连续型特征与目标特征
        cont_features = ["Irradiance", "Temperature", "Actual_Power"]
        
        # 拟合或应用归一化
        if scaler is None and self.is_train:
            self.scaler = {}
            for col in cont_features:
                s = StandardScaler()
                s.fit(self.data[[col]].values)
                self.scaler[col] = s
        else:
            self.scaler = scaler
            
        # 转换连续特征
        self.scaled_data = {}
        for col in cont_features:
            self.scaled_data[col] = self.scaler[col].transform(self.data[[col]].values).squeeze()
            
        # 离散运维事件特征 (0-3)
        self.events = self.data["Event_Type"].values
        
        # 计算归一化后的物理零值 (对于 StandardScaler 为 -mean / std)
        mean_power = self.scaler["Actual_Power"].mean_[0]
        std_power = self.scaler["Actual_Power"].scale_[0]
        self.zero_value = -mean_power / std_power
        
    def __len__(self):
        # 确保有足够的历史和未来窗口
        return len(self.data) - self.history_len - self.forecast_len + 1
        
    def __getitem__(self, idx):
        # 历史特征起点和终点
        hist_start = idx
        hist_end = idx + self.history_len
        
        # 未来特征起点和终点
        fut_start = hist_end
        fut_end = hist_end + self.forecast_len
        
        # 1. 历史连续特征 (Irradiance, Temperature, Actual_Power)
        hist_irr = self.scaled_data["Irradiance"][hist_start:hist_end]
        hist_temp = self.scaled_data["Temperature"][hist_start:hist_end]
        hist_power = self.scaled_data["Actual_Power"][hist_start:hist_end]
        hist_cont = np.stack([hist_irr, hist_temp, hist_power], axis=-1) # (T, 3)
        
        # 2. 历史事件特征 (Event_Type)
        hist_event = self.events[hist_start:hist_end] # (T,)
        
        # 3. 未来连续特征 (未来气象 NWP 信息: Irradiance, Temperature)
        fut_irr = self.scaled_data["Irradiance"][fut_start:fut_end]
        fut_temp = self.scaled_data["Temperature"][fut_start:fut_end]
        fut_cont = np.stack([fut_irr, fut_temp], axis=-1) # (H, 2)
        
        # 4. 未来事件特征 (例如已知的未来检修计划等)
        # 在实际中，未来的故障无法预知，但未来的计划检修是已知的。
        # 这里直接加载对应的事件类型，但在实际评估时，我们会假设非计划性的设备故障(Event_Type=2)
        # 和电网限电(Event_Type=3)在未来时段不可知（因此设为0/Normal），而计划检修(Event_Type=1)是可知的。
        fut_event = self.events[fut_start:fut_end].copy()
        # 模拟“未知故障”和“未知限电”，所以在预测未来时，将非计划性事件掩盖为正常
        # 只有计划检修 (1) 保留
        fut_event_masked = np.where((fut_event == 2) | (fut_event == 3), 0, fut_event)
        
        # 5. 预测目标 (未归一化的实际功率，或者归一化的实际功率均可，这里返回归一化后的作为模型输出，并在计算指标时逆归一化)
        target = self.scaled_data["Actual_Power"][fut_start:fut_end] # (H,)
        target = np.expand_dims(target, axis=-1) # (H, 1)
        
        # 转换为张量
        return {
            "hist_cont": torch.tensor(hist_cont, dtype=torch.float32),
            "hist_event": torch.tensor(hist_event, dtype=torch.long),
            "future_cont": torch.tensor(fut_cont, dtype=torch.float32),
            "future_event": torch.tensor(fut_event_masked, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            # 同时也把真实的未来事件传回去，便于在评估时单独统计故障时段的误差
            "future_event_real": torch.tensor(fut_event, dtype=torch.long),
            "zero_value": torch.tensor(self.zero_value, dtype=torch.float32)
        }

def get_dataloaders(csv_path, history_len=24, forecast_len=24, batch_size=32, train_ratio=0.8):
    """
    便捷获取训练和测试数据加载器的函数
    """
    train_dataset = PowerStationDataset(
        csv_path=csv_path, 
        history_len=history_len, 
        forecast_len=forecast_len, 
        train_ratio=train_ratio, 
        is_train=True
    )
    
    # 测试集共用训练集的 scaler
    test_dataset = PowerStationDataset(
        csv_path=csv_path, 
        history_len=history_len, 
        forecast_len=forecast_len, 
        train_ratio=train_ratio, 
        is_train=False,
        scaler=train_dataset.scaler
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader, train_dataset.scaler
