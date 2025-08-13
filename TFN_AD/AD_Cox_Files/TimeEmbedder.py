import torch
import torch.nn as nn

class TimeEmbedder(nn.Module):
    """
    Per-visit time features (e.g., Δt since baseline, absolute time, interval) → MLP → z_τ
    Inputs:
        t: [B, t_in]  or  [B, T, t_in]
    Output:
        z_τ: [B, T, t_dim] (T=1 if no time axis in input)
    """
    def __init__(self, in_dim: int = 1, out_dim: int = 8, hidden: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = max(2 * out_dim, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 3:
            # [B, T, t_in] → [B*T, t_in] → MLP → [B*T, t_dim] → [B, T, t_dim]
            B, T, Din = t.shape
            z = self.mlp(t.view(B * T, Din)).view(B, T, -1)
            return z
        elif t.dim() == 2:
            # [B, t_in] → [B, t_dim] → [B, 1, t_dim]
            B, Din = t.shape
            z = self.mlp(t).unsqueeze(1)
            return z
        else:
            raise ValueError(f"TimeEmbedder expects [B,t_in] or [B,T,t_in], got {t.shape}")