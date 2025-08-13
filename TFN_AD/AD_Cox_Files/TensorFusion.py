import torch
import torch.nn as nn

def _align_time_dims(z_v: torch.Tensor, z_d: torch.Tensor, z_tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ensure all three tensors have the same time length.
    If any has T=1 while another has T>1, repeat along time.
    """
    if not (z_v.dim() == z_d.dim() == z_tau.dim() == 3):
        raise ValueError(f"Expected all embeddings as [B,T,D] (or [B,1,D]), got shapes: "
                         f"z_v={tuple(z_v.shape)}, z_d={tuple(z_d.shape)}, z_tau={tuple(z_tau.shape)}")

    B = z_v.size(0)
    T = max(z_v.size(1), z_d.size(1), z_tau.size(1))

    def _maybe_repeat(x):
        if x.size(1) == T:
            return x
        if x.size(1) == 1:
            return x.repeat(1, T, 1)
        raise ValueError(f"Time length mismatch: got T={x.size(1)} but target T={T}")

    return _maybe_repeat(z_v), _maybe_repeat(z_d), _maybe_repeat(z_tau)

class TensorFusion(nn.Module):
    """
    (1 ⊕ z_v) ⊗ (1 ⊕ z_d) ⊗ (1 ⊕ z_tau) per visit, flattened.
    Optionally project to a smaller dimension to control size.

    Args:
        v_dim (int): dimension of z_v
        d_dim (int): dimension of z_d
        t_dim (int): dimension of z_tau
        proj_dim (int|None): if set, applies a Linear(+Dropout) to reduce fused size
        dropout (float): dropout rate applied after projection (if proj_dim is set)
    """
    def __init__(self, v_dim: int, d_dim: int, t_dim: int, proj_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.v_dim = v_dim
        self.d_dim = d_dim
        self.t_dim = t_dim
        self.fused_dim = (1 + v_dim) * (1 + d_dim) * (1 + t_dim)

        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(self.fused_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.out_dim = proj_dim
        else:
            self.proj = None
            self.out_dim = self.fused_dim

    def forward(self, z_v: torch.Tensor, z_d: torch.Tensor, z_tau: torch.Tensor) -> torch.Tensor:
        # Ensure [B,T,D] and align time dimensions
        if z_v.dim() == 2: z_v = z_v.unsqueeze(1)  # [B,1,v]
        if z_d.dim() == 2: z_d = z_d.unsqueeze(1)  # [B,1,d]
        if z_tau.dim() == 2: z_tau = z_tau.unsqueeze(1)  # [B,1,t]
        z_v, z_d, z_tau = _align_time_dims(z_v, z_d, z_tau)  # [B,T,·]

        B, T, _ = z_v.shape
        device, dtype = z_v.device, z_v.dtype

        ones_v = torch.ones(B, T, 1, device=device, dtype=dtype)
        ones_d = torch.ones(B, T, 1, device=device, dtype=dtype)
        ones_t = torch.ones(B, T, 1, device=device, dtype=dtype)

        zv = torch.cat([ones_v, z_v], dim=-1)     # [B,T,1+v]
        zd = torch.cat([ones_d, z_d], dim=-1)     # [B,T,1+d]
        zt = torch.cat([ones_t, z_tau], dim=-1)   # [B,T,1+t]

        # Outer product per-visit via einsum: (B,T,I) x (B,T,J) x (B,T,K) -> (B,T,I,J,K)
        fused_3d = torch.einsum('bti,btj,btk->btijk', zv, zd, zt)
        fused = fused_3d.reshape(B, T, -1)        # [B,T,(1+v)(1+d)(1+t)]

        if self.proj is not None:
            fused = self.proj(fused)              # [B,T,proj_dim]

        return fused