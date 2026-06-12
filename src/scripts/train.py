import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.dataset import get_dataloaders
from src.models.baseline_transformer import BaselineTransformer
from src.models.om_aware_transformer import OMAwareTransformer

def train_model(model_type, csv_path, history_len=24, forecast_len=24, epochs=15, batch_size=32, lr=0.001, device="cpu"):
    """
    模型训练主函数
    model_type: "baseline" 或 "om_aware"
    """
    train_loader, test_loader, scaler = get_dataloaders(
        csv_path=csv_path,
        history_len=history_len,
        forecast_len=forecast_len,
        batch_size=batch_size
    )
    
    # 初始化模型
    if model_type == "baseline":
        model = BaselineTransformer(hist_dim=3, future_dim=2, out_dim=1, d_model=64)
    elif model_type == "om_aware":
        model = OMAwareTransformer(hist_dim=3, future_dim=2, out_dim=1, d_model=64)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"--- Training {model_type} on {os.path.basename(csv_path)} (Device: {device}) ---")
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # 移至 device
            batch_cuda = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            preds = model(batch_cuda)
            loss = criterion(preds, batch_cuda["target"])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}")
            
    # 评估模型
    metrics = evaluate_model(model, test_loader, scaler, device)
    return model, metrics

def evaluate_model(model, test_loader, scaler, device="cpu"):
    """
    模型测试与分类指标统计（总体指标、常规时段指标、运维事件时段指标）
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_events = [] # 真实未来事件类别
    
    power_scaler = scaler["Actual_Power"]
    
    with torch.no_grad():
        for batch in test_loader:
            batch_cuda = {k: v.to(device) for k, v in batch.items()}
            preds = model(batch_cuda)
            
            # 转回 CPU 数组
            preds_np = preds.cpu().numpy() # (batch, H, 1)
            target_np = batch["target"].numpy() # (batch, H, 1)
            events_np = batch["future_event_real"].numpy() # (batch, H)
            
            all_preds.append(preds_np)
            all_targets.append(target_np)
            all_events.append(events_np)
            
    # 拼接数据
    all_preds = np.concatenate(all_preds, axis=0).squeeze(-1) # (N, H)
    all_targets = np.concatenate(all_targets, axis=0).squeeze(-1) # (N, H)
    all_events = np.concatenate(all_events, axis=0) # (N, H)
    
    # 逆归一化，恢复成真实功率（MW）
    all_preds_mw = power_scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
    all_targets_mw = power_scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)
    
    # 负值截断（物理常识，功率必为非负）
    all_preds_mw = np.maximum(0, all_preds_mw)
    
    # 展平为一维数组计算指标
    preds_flat = all_preds_mw.flatten()
    targets_flat = all_targets_mw.flatten()
    events_flat = all_events.flatten()
    
    # 分类别进行性能指标统计
    # 1. 总体指标 (Overall)
    overall_mae = mean_absolute_error(targets_flat, preds_flat)
    overall_rmse = np.sqrt(mean_squared_error(targets_flat, preds_flat))
    overall_r2 = r2_score(targets_flat, preds_flat)
    
    # 2. 正常时段指标 (Normal Periods: Event_Type == 0)
    normal_idx = (events_flat == 0)
    if np.sum(normal_idx) > 0:
        normal_mae = mean_absolute_error(targets_flat[normal_idx], preds_flat[normal_idx])
        normal_rmse = np.sqrt(mean_squared_error(targets_flat[normal_idx], preds_flat[normal_idx]))
    else:
        normal_mae, normal_rmse = 0.0, 0.0
        
    # 3. 运维事件时段指标 (O&M Event Periods: Event_Type != 0)
    event_idx = (events_flat != 0)
    if np.sum(event_idx) > 0:
        event_mae = mean_absolute_error(targets_flat[event_idx], preds_flat[event_idx])
        event_rmse = np.sqrt(mean_squared_error(targets_flat[event_idx], preds_flat[event_idx]))
    else:
        event_mae, event_rmse = 0.0, 0.0
        
    # 4. 细分特定事件类型误差：计划检修 (Event_Type == 1)
    maint_idx = (events_flat == 1)
    if np.sum(maint_idx) > 0:
        maint_mae = mean_absolute_error(targets_flat[maint_idx], preds_flat[maint_idx])
    else:
        maint_mae = 0.0
        
    metrics = {
        "overall": {"mae": overall_mae, "rmse": overall_rmse, "r2": overall_r2},
        "normal": {"mae": normal_mae, "rmse": normal_rmse},
        "event": {"mae": event_mae, "rmse": event_rmse},
        "maintenance": {"mae": maint_mae},
        "raw_preds_mw": all_preds_mw,
        "raw_targets_mw": all_targets_mw,
        "raw_events": all_events
    }
    
    return metrics
