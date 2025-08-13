# --- 4) Sequence Cox Model (LSTM + Cox Head) ---
# Uses per-visit Tensor Fusion → LSTM → Cox head, and the agreed EfronLossPenalty for training.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class CoxPHHead(nn.Module):
    """Linear Cox head: outputs log-risk β^T S (no activation)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.beta = nn.Linear(in_dim, 1, bias=False)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.beta(S).squeeze(-1)  # [B]

class SequenceCoxModel(nn.Module):
    """
    Sequence Cox model:
      - z_v^(t) = VisualEmbedder(imgs_t)
      - z_d     = DemographicsEmbedder(demos[, broadcast to T])
      - z_τ^(t) = TimeEmbedder(times_t)
      - h^(t)   = TensorFusion(z_v^(t), z_d, z_τ^(t))
      - S       = LSTM({h^(t)}) summary (last hidden)
      - log_risk = β^T S
    """
    def __init__(
        self,
        v_embedder: nn.Module,
        d_embedder: nn.Module,
        t_embedder: nn.Module,
        fusion: nn.Module,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.v = v_embedder
        self.d = d_embedder
        self.t = t_embedder
        self.fusion = fusion

        self.lstm = nn.LSTM(
            input_size=self.fusion.out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        head_in = lstm_hidden * (2 if bidirectional else 1)
        self.head = CoxPHHead(in_dim=head_in)

    @torch.no_grad()
    def _infer_seq_len(self, x: torch.Tensor) -> int:
        # imgs expected as [B,T,1,H,W]
        return x.size(1)

    def forward(
        self,
        imgs: torch.Tensor,              # [B,T,1,H,W]
        demos: torch.Tensor,             # [B,d_in] or [B,T,d_in]
        times: torch.Tensor,             # [B,T,t_in]
        lengths: torch.Tensor | None = None  # [B] true lengths (before padding)
    ) -> torch.Tensor:
        B = imgs.size(0)
        T = self._infer_seq_len(imgs)

        # Embeddings
        z_v = self.v(imgs)  # [B,T,v]
        z_d = self.d(demos, seq_len=T) if demos.dim() == 2 else self.d(demos)  # [B,T,d]
        z_tau = self.t(times)  # [B,T,t]

        # Per-visit fusion → H
        H = self.fusion(z_v, z_d, z_tau)  # [B,T,F]

        # Sequence encoder
        if lengths is not None:
            lengths = lengths.to('cpu').to(dtype=torch.int64)
            packed = pack_padded_sequence(H, lengths, batch_first=True, enforce_sorted=False)
            _, (hn, _) = self.lstm(packed)  # hn: [layers*dir, B, hidden]
        else:
            _, (hn, _) = self.lstm(H)

        S = hn[-1]  # final layer's hidden: [B, hidden] (bidir already combined by layer stacking)
        log_risk = self.head(S)  # [B]
        return log_risk


# Convenience builder that wires modules and returns (model, loss_fn)
def build_sequence_cox_model(
    visual_embedder: nn.Module,
    demo_embedder: nn.Module,
    time_embedder: nn.Module,
    fusion_module: nn.Module,
    lstm_hidden: int = 64,
    lstm_layers: int = 1,
    bidirectional: bool = False,
    dropout: float = 0.0,
    penalty: float = 0.0,          # λ for EfronLossPenalty
    return_stats: bool = False
):
    """
    Returns:
      model   : SequenceCoxModel
      loss_fn : EfronLossPenalty(penalty=λ)
    Note: EfronLossPenalty must be defined/imported from step 4-loss cell.
    """
    model = SequenceCoxModel(
        v_embedder=visual_embedder,
        d_embedder=demo_embedder,
        t_embedder=time_embedder,
        fusion=fusion_module,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    # Use the agreed penalty-based Efron loss
    loss_fn = EfronLossPenalty(penalty=penalty, return_stats=return_stats)
    return model, loss_fn

# --- Efron loss with per-tie-block penalty  ---
class EfronLossPenalty(nn.Module):
    def __init__(self, penalty: float = 0.0, return_stats: bool = False, eps: float = 1e-12):
        """
        Args:
            penalty: λ ≥ 0. If >0, adds λ * sum(|log_risk|) per tie block.
            return_stats: if True, also returns (tie_counts, cum_exp_risk, failure_times) as numpy arrays.
            eps: small constant for numerical stability.
        """
        super().__init__()
        self.penalty = float(penalty)
        self.return_stats = bool(return_stats)
        self.eps = float(eps)

    def forward(self, times: torch.Tensor, events: torch.Tensor, log_risk: torch.Tensor):
        """
        Inputs:
            times    : [N] float tensor (larger = later)
            events   : [N] float/bool (1=event, 0=censor)
            log_risk : [N] β^T x (model output)
        Returns:
            loss (scalar), and optionally diagnostics if return_stats=True
        """
        # Ensure shapes/dtypes/devices consistent
        log_risk = log_risk.view(-1)
        times    = times.view(-1).to(dtype=log_risk.dtype, device=log_risk.device)
        events   = events.view(-1).to(dtype=log_risk.dtype, device=log_risk.device)

        if events.sum().item() == 0:
            return (log_risk * 0.0).sum()

        # Sort by time descending so equal times are contiguous; risk set is cumulative up to block end
        order = torch.argsort(times, descending=True)
        log_risk = log_risk[order]
        times    = times[order]
        events   = events[order]

        risk = torch.exp(log_risk)              # [N]
        cum_risk = torch.cumsum(risk, dim=0)    # [N]

        # Tie blocks (consecutive equal times)
        _, counts = torch.unique_consecutive(times, return_counts=True)

        idx_start = 0
        total = log_risk.new_tensor(0.0)

        # (optional) diagnostics
        tie_counts_list = []
        cum_exp_risk_list = []
        failure_times_list = []

        for c in counts:
            idx_end = idx_start + c - 1
            block = slice(idx_start, idx_end + 1)

            e_mask = (events[block] > 0.5)
            d = int(e_mask.sum().item())

            if d > 0:
                # Sum of log_risk for events within this time block
                log_risk_events = log_risk[block][e_mask].sum()

                # Risk set sum at the end of the block
                risk_set_sum = cum_risk[idx_end]

                # Sum of risks among the events in the block
                events_risk_sum = risk[block][e_mask].sum()

                # Efron correction: average over l = 0..d-1
                if d == 1:
                    denom = torch.log(risk_set_sum + self.eps)
                else:
                    l = torch.arange(d, device=log_risk.device, dtype=log_risk.dtype)
                    denom_terms = (risk_set_sum - (l / d) * events_risk_sum).clamp_min(self.eps)
                    denom = torch.log(denom_terms).sum()

                block_loss = denom - log_risk_events

                # Penalty per tie block on |log_risk|
                if self.penalty > 0.0:
                    block_loss = block_loss + self.penalty * log_risk[block].abs().sum()

                total = total + block_loss

                if self.return_stats:
                    tie_counts_list.append(d)
                    cum_exp_risk_list.append(risk_set_sum.detach().cpu())
                    failure_times_list.append(times[idx_end].detach().cpu())
            else:
                if self.return_stats:
                    tie_counts_list.append(0)
                    cum_exp_risk_list.append(cum_risk[idx_end].detach().cpu())
                    failure_times_list.append(times[idx_end].detach().cpu())

            idx_start = idx_end + 1

        num_events = events.sum().clamp(min=1.0)
        loss = total / num_events

        if self.return_stats:
            import numpy as np
            tie_counts = np.array(tie_counts_list)
            cum_exp_risk = np.array([x.item() for x in cum_exp_risk_list])
            failure_times = np.array([x.item() for x in failure_times_list])
            return loss, tie_counts, cum_exp_risk, failure_times
        else:
            return loss
