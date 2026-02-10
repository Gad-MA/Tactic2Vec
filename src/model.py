import torch
import torch.nn as nn

class GatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                padding=(kernel_size-1)*dilation//2, 
                                dilation=dilation)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                padding=(kernel_size-1)*dilation//2, 
                                dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        out = f * g
        out = self.dropout(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class SiameseTCN(nn.Module):
    def __init__(self, input_channels=44, hidden_channels=32, embedding_dim=64):
        super().__init__()
        
        # TCN Backbone
        self.tcn = nn.Sequential(
            GatedTCNBlock(input_channels, hidden_channels, kernel_size=3, dilation=1),
            GatedTCNBlock(hidden_channels, hidden_channels, kernel_size=3, dilation=2),
            GatedTCNBlock(hidden_channels, hidden_channels, kernel_size=3, dilation=4),
            GatedTCNBlock(hidden_channels, hidden_channels, kernel_size=3, dilation=8)
        )
        
        # Embedding Head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward_one(self, x):
        # x: (B, 44, T) - batch of flattened scene tensors (22 players × 2 coords)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)  # (B, hidden)
        x = self.fc(x)
        return x
    
    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2
