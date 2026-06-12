import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class BaselineTransformer(nn.Module):
    def __init__(self, hist_dim=3, future_dim=2, out_dim=1, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, dropout=0.1):
        super(BaselineTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 输入映射层
        self.enc_embedding = nn.Linear(hist_dim, d_model)
        self.dec_embedding = nn.Linear(future_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        
        # Transformer 架构
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
        
        # 输出线性映射层
        self.out_projection = nn.Linear(d_model, out_dim)
        
    def forward(self, batch):
        """
        batch: 字典，包含:
            hist_cont: (batch_size, T, hist_dim)
            future_cont: (batch_size, H, future_dim)
        """
        x_hist = batch["hist_cont"]
        x_fut = batch["future_cont"]
        
        # 1. 编码历史特征
        enc_input = self.enc_embedding(x_hist) # (batch, T, d_model)
        enc_input = self.pos_encoder(enc_input)
        memory = self.encoder(enc_input) # (batch, T, d_model)
        
        # 2. 解码未来特征
        dec_input = self.dec_embedding(x_fut) # (batch, H, d_model)
        dec_input = self.pos_decoder(dec_input)
        
        # 因果掩码 (Causal Mask) 用于解码器，防止看到未来的特征（这里因为未来气象已知，可选，但标准 Transformer 使用它）
        # 这里可以使用下三角掩码
        sz = x_fut.size(1)
        # 生成掩码，返回 (sz, sz) 形式
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=x_fut.device), diagonal=1)
        
        out = self.decoder(
            tgt=dec_input, 
            memory=memory, 
            tgt_mask=mask
        ) # (batch, H, d_model)
        
        # 3. 预测未来功率
        preds = self.out_projection(out) # (batch, H, 1)
        return preds
