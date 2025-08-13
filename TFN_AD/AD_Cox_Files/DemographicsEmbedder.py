import torch
import torch.nn as nn

class DemographicsEmbedder(nn.Module):
    """
    Structured features (numeric + one-hot) → MLP → z_d, with optional time broadcast.
    Inputs:
        x: [B, d_in]       or  [B, T, d_in]
        seq_len (optional): if provided and x is [B, d_in], broadcast to [B, seq_len, d_dim]
    Output:
        z_d: [B, T, d_dim] (T=1 if no time/broadcast)
    """
    def __init__(self, in_dim: int, out_dim: int = 16, hidden: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = max(2 * out_dim, 16)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        if x.dim() == 3:
            # [B, T, d_in] → [B*T, d_in] → MLP → [B*T, d_dim] → [B, T, d_dim]
            B, T, D = x.shape
            z = self.mlp(x.view(B * T, D)).view(B, T, -1)
            return z
        elif x.dim() == 2:
            # [B, d_in] → [B, d_dim] (→ optional broadcast to [B, seq_len, d_dim])
            B, D = x.shape
            z = self.mlp(x)  # [B, d_dim]
            z = z.unsqueeze(1)  # [B, 1, d_dim]
            if seq_len is not None and seq_len > 1:
                z = z.repeat(1, seq_len, 1)  # broadcast to time axis
            return z
        else:
            raise ValueError(f"DemographicsEmbedder expects [B,d_in] or [B,T,d_in], got {x.shape}")