import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

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

class VisualEmbedder(nn.Module):
    """
    VGG16-based visual embedder (sequence-ready).
    Inputs:
        x: [B, 1, H, W]  or  [B, T, 1, H, W]  (grayscale MRI)
    Output:
        z_v: [B, T, v_dim]  (T=1 if input had no time axis)
    Notes:
        - Internally resizes to 224×224, replicates to 3ch, and applies ImageNet normalization.
        - Set train_backbone=True to fine-tune VGG; False to freeze.
    """
    def __init__(self, out_dim: int = 256, train_backbone: bool = False, pool: str = "gap"):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights)
        self.backbone = vgg.features  # conv blocks only

        # freeze / unfreeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = bool(train_backbone)

        # global pooling → 512-d
        if pool.lower() == "gap":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))

        # projection to desired v_dim
        self.proj = nn.Linear(512, out_dim)

        # register ImageNet normalization buffers
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _preprocess_rgb224(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*, 1, H, W] in [0,1] range (recommendation).
        Convert to 3ch, resize to 224, ImageNet normalize.
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # gray→RGB
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Expect inputs in [0,1]; clamp just in case
        x = x.clamp(0, 1)
        x = (x - self.mean) / self.std
        return x

    def _forward_single(self, x4d: torch.Tensor) -> torch.Tensor:
        # x4d: [B, 1, H, W]
        x4d = self._preprocess_rgb224(x4d)          # → [B,3,224,224]
        feat = self.backbone(x4d)                   # → [B,512,7,7]
        feat = self.pool(feat).flatten(1)           # → [B,512]
        z = self.proj(feat)                         # → [B,out_dim]
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            z = self._forward_single(x)             # [B*T, out_dim]
            return z.view(B, T, -1)                 # [B, T, out_dim]
        elif x.dim() == 4:
            z = self._forward_single(x)             # [B, out_dim]
            return z.unsqueeze(1)                   # [B, 1, out_dim]
        else:
            raise ValueError(f"Expected [B,1,H,W] or [B,T,1,H,W], got {tuple(x.shape)}")
