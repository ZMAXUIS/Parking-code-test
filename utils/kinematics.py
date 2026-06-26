import torch
import math
from typing import Tuple

from utils.trajectory_utils import detokenize_traj_point


def detokenize_pred_tokens(pred_tokens: torch.Tensor, token_nums: int, item_number: int, autoregressive_points: int, xy_max: float = 12.0) -> torch.Tensor:
    """Convert predicted discrete tokens into continuous trajectory points.
    pred_tokens: (B, S) int tensor containing predicted token ids (including BOS/EOS/PAD)
    We locate tokens < token_nums (i.e. real spatial/progress tokens), group them in order by item_number
    and take the first autoregressive_points groups. Returns (B, autoregressive_points, item_number) float tensor.
    """
    B, S = pred_tokens.shape
    n_needed = autoregressive_points * item_number
    out = []
    for b in range(B):
        seq = pred_tokens[b].tolist()
        # filter only real tokens (0..token_nums-1)
        real = [t for t in seq if int(t) < token_nums]
        # pad or trim to required length
        if len(real) < n_needed:
            real = real + [token_nums - 1] * (n_needed - len(real))
        else:
            real = real[:n_needed]
        t = torch.tensor(real, dtype=torch.long)
        coords = detokenize_traj_point(t, token_nums, item_number, xy_max=xy_max)  # shape (num_points, item_number)
        coords = coords.view(autoregressive_points, -1)
        out.append(coords)
    out = torch.stack(out, dim=0)  # (B, autoregressive_points, item_number)
    return out


def compute_acceleration_loss(pred_pts: torch.Tensor) -> torch.Tensor:
    """Compute acceleration (second difference) squared loss.
    pred_pts: (B, T, 2) continuous coordinates
    Returns mean squared accel.
    """
    if pred_pts.size(1) < 3:
        return torch.tensor(0.0, device=pred_pts.device)
    v1 = pred_pts[:, 1:, :] - pred_pts[:, :-1, :]   # (B, T-1, 2)
    a = v1[:, 1:, :] - v1[:, :-1, :]               # (B, T-2, 2)
    loss = (a ** 2).sum(dim=-1).mean()
    return loss


def compute_curvature_loss(pred_pts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Approximate curvature loss using heading differences normalized by arc length.
    pred_pts: (B, T, 2)
    Returns mean squared curvature.
    """
    B, T, _ = pred_pts.shape
    if T < 3:
        return torch.tensor(0.0, device=pred_pts.device)
    # headings
    delta = pred_pts[:, 1:, :] - pred_pts[:, :-1, :]  # (B, T-1, 2)
    headings = torch.atan2(delta[..., 1], delta[..., 0])  # (B, T-1)
    # angle differences
    dhead = headings[:, 1:] - headings[:, :-1]  # (B, T-2)
    # wrap to [-pi, pi]
    dhead = (dhead + math.pi) % (2 * math.pi) - math.pi
    # approximate arc length per segment
    s1 = torch.norm(delta[:, :-1, :], dim=-1)  # (B, T-2)
    s2 = torch.norm(delta[:, 1:, :], dim=-1)
    s = 0.5 * (s1 + s2) + eps
    curvature = dhead / s
    loss = (curvature ** 2).mean()
    return loss

