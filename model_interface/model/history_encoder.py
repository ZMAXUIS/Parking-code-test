import torch
from torch import nn

from utils.config import Configuration


class HistoryEncoder(nn.Module):
    """
    Simple history trajectory encoder.
    Input: history_traj (B, hist_len, 2) in ego coordinates (meters)
    Output: sequence features (B, hist_len, D) where D == cfg.tf_de_dim by default.

    Implementation: a small GRU + linear projection to cfg.tf_de_dim.
    """

    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.hist_len = int(getattr(cfg, 'history_len', 10))
        self.input_dim = 2
        self.hidden_dim = int(getattr(cfg, 'history_hidden_dim', 64))
        # determine output dim safely
        out_dim_cfg = getattr(cfg, 'history_out_dim', None)
        if out_dim_cfg is None:
            out_dim_cfg = getattr(cfg, 'tf_de_dim', 64)
        self.out_dim = int(out_dim_cfg)

        # small projection to hidden dim
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=False)
        # project GRU hidden back to decoder d_model
        self.out_proj = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, history_traj):
        """
        history_traj: tensor or None
        if None -> return None
        expected shape: (B, hist_len, 2)
        returns: hist_feats (B, hist_len, out_dim)
        """
        if history_traj is None:
            return None
        # ensure float
        x = history_traj.float()
        # if shape mismatch, try to trim/pad
        if x.dim() != 3 or x.size(-1) != 2:
            # try to reshape if (B, 2, hist_len)
            if x.dim() == 3 and x.size(1) == 2 and x.size(2) <= self.hist_len:
                x = x.permute(0, 2, 1)
            else:
                # fallback: return None
                return None

        # if too long, trim; if too short, pad zeros
        if x.size(1) > self.hist_len:
            x = x[:, -self.hist_len:, :]
        elif x.size(1) < self.hist_len:
            pad = torch.zeros((x.size(0), self.hist_len - x.size(1), x.size(2)), device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)

        h = self.input_proj(x)
        gru_out, _ = self.gru(h)
        out = self.out_proj(gru_out)  # (B, hist_len, out_dim)
        return out
