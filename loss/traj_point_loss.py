from torch import nn
import torch

from utils.config import Configuration


class TokenTrajPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(TokenTrajPointLoss, self).__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.PAD_token)

    def forward(self, pred, data):
        pred = pred[:, :-1,:]
        pred_traj_point = pred.reshape(-1, pred.shape[-1])
        # move gt tokens to same device as predictions
        device = pred_traj_point.device
        gt_traj_point_token = data['gt_traj_point_token'][:, 1:-1].reshape(-1).to(device)

        traj_point_loss = self.ce_loss(pred_traj_point, gt_traj_point_token)
        return traj_point_loss


class TrajPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(TrajPointLoss, self).__init__()
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, data):
        gt = data['gt_traj_point'].view(-1, self.cfg.autoregressive_points, 2)
        # Ensure no NaN or Inf in predictions and ground truth
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e5, neginf=-1e5)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=1e5, neginf=-1e5)
        traj_point_loss = self.mse_loss(pred, gt)
        return traj_point_loss