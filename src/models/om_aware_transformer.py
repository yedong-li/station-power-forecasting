import torch
import torch.nn as nn
import math
from .baseline_transformer import PositionalEncoding

class OMAwareTransformer(nn.Module):
    def __init__(self, hist_dim=3, future_dim=2, out_dim=1, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, dropout=0.1, num_events=4, use_gate=True):
        super(OMAwareTransformer, self).__init__()
        
        self.d_model = d_model
        self.use_gate = use_gate
        
        # 1. 运维事件嵌入层 (O&M Event Embedding)
        # 0: 正常, 1: 计划检修, 2: 设备故障, 3: 电网限电
        self.event_embedding = nn.Embedding(num_embeddings=num_events, embedding_dim=d_model)
        
        # 2. 连续特征嵌入层
        self.enc_cont_embedding = nn.Linear(hist_dim, d_model)
        self.dec_cont_embedding = nn.Linear(future_dim, d_model)
        
        # 3. 多源数据融合层 (将连续特征与离散事件特征拼接融合)
        self.hist_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fut_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        
        # Transformer 核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 4. 预测输出层
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, out_dim)
        )
        
        # 5. 核心创新点：可微运维门控机制 (O&M Gate)
        self.gate_projection = nn.Linear(d_model, 1)
        # 对不同事件的门控偏置初始化，使其在反向传播前即具备物理常识偏置：
        # 正常(0)->无减免(极低门控), 计划检修(1)->强制置零(高门控), 设备故障(2)->部分减免(中等门控), 限电(3)->小幅调整
        self.event_gate_bias = nn.Parameter(torch.tensor([-10.0, 10.0, 0.0, -1.0], dtype=torch.float32))
        
    def forward(self, batch):
        """
        batch 字典，包含:
            hist_cont: (batch, T, hist_dim)
            hist_event: (batch, T)
            future_cont: (batch, H, future_dim)
            future_event: (batch, H)  -- 在未来已知的事件类型（如计划检修已填入，未知故障为0）
        """
        x_hist_cont = batch["hist_cont"]
        x_hist_event = batch["hist_event"]
        x_fut_cont = batch["future_cont"]
        x_fut_event = batch["future_event"]
        
        batch_size = x_hist_cont.size(0)
        
        # 1. 提取并融合历史特征
        h_cont = self.enc_cont_embedding(x_hist_cont) # (batch, T, d_model)
        h_event = self.event_embedding(x_hist_event) # (batch, T, d_model)
        h_fused = self.hist_fusion(torch.cat([h_cont, h_event], dim=-1)) # (batch, T, d_model)
        h_fused = self.pos_encoder(h_fused)
        
        # Transformer 编码器
        memory = self.encoder(h_fused) # (batch, T, d_model)
        
        # 2. 提取并融合未来特征
        f_cont = self.dec_cont_embedding(x_fut_cont) # (batch, H, d_model)
        f_event = self.event_embedding(x_fut_event) # (batch, H, d_model)
        f_fused = self.fut_fusion(torch.cat([f_cont, f_event], dim=-1)) # (batch, H, d_model)
        f_fused = self.pos_decoder(f_fused)
        
        # 因果自注意力掩码
        sz = x_fut_cont.size(1)
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=x_fut_cont.device), diagonal=1)
        
        # Transformer 解码器
        dec_out = self.decoder(
            tgt=f_fused, 
            memory=memory, 
            tgt_mask=mask
        ) # (batch, H, d_model)
        
        # 3. 原始预测值
        raw_preds = self.out_projection(dec_out) # (batch, H, 1)
        
        # 4. 计算运维门控值
        # 投影出基础门控 logits
        gate_logits = self.gate_projection(dec_out).squeeze(-1) # (batch, H)
        
        # 根据未来的事件类别索引，获取对应的先验偏置值
        # x_fut_event 形状为 (batch, H)，其值范围是 0~3
        # 获取偏置：[batch, H]
        bias = self.event_gate_bias[x_fut_event]
        
        # 结合神经网络学习到的时空状态与事件先验偏置
        g = torch.sigmoid(gate_logits + bias) # (batch, H)
        g = g.unsqueeze(-1) # (batch, H, 1)
        
        # 5. 应用门控调整最终预测值：当g趋近1时，实际预测被置为归一化零值
        if self.use_gate:
            zero_value = batch["zero_value"].view(-1, 1, 1) # (batch, 1, 1)
            final_preds = raw_preds * (1.0 - g) + g * zero_value
        else:
            final_preds = raw_preds
        
        return final_preds
