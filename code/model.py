import torch
import torch.nn as nn

class ConvBiGRUAutoencoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, n_layers, dropout):
        super().__init__()
        # 使用多层卷积提取更复杂的特征
        self.conv = nn.Sequential(
            nn.Conv1d(inp_dim, inp_dim * 2, kernel_size=3, padding=1),  # 增加通道数
            nn.ReLU(),
            nn.BatchNorm1d(inp_dim * 2),
            nn.Conv1d(inp_dim * 2, inp_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(inp_dim * 2)
        )
        # 双向 GRU 编码器
        self.enc = nn.GRU(
            inp_dim * 2,  # 输入维度与卷积输出相匹配
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        # 解码器（输入 hid_dim*2 -> 输出 inp_dim）
        self.dec = nn.GRU(
            hid_dim * 2,
            inp_dim,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)  # -> (B, C, T)
        x = self.conv(x)  # -> (B, C, T)
        x = x.permute(0, 2, 1)  # -> (B, T, C)

        B = x.size(0)
        h0 = torch.zeros(
            self.enc.num_layers * 2,  # 双向
            B,
            self.enc.hidden_size,
            device=x.device
        )
        enc_out, _ = self.enc(x, h0)
        dec_out, _ = self.dec(enc_out)
        return dec_out
