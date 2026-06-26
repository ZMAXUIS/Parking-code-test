import pytorch_lightning as pl
import torch

from loss.traj_point_loss import TokenTrajPointLoss, TrajPointLoss
from loss.depth_loss import DepthLoss
from loss.seg_loss import SegmentationLoss
from model_interface.model.parking_model_real import ParkingModelReal
from utils.config import Configuration
from utils.metrics import CustomizedMetric
from utils.kinematics import detokenize_pred_tokens, compute_acceleration_loss, compute_curvature_loss
import torch.nn as nn


def _sanitize_metrics(metrics: dict):
    """Convert tensors or arrays in metrics dict to Python scalars where possible.
    - If value is a torch.Tensor with a single element, convert to float(item).
    - If tensor has >1 elements, convert to float(tensor.mean().detach().cpu().item()) and keep a warning key.
    - If value is numpy array, convert to float(np.mean(value)).
    Returns sanitized dict of primitives (floats/ints) suitable for PL logging.
    """
    import numpy as _np
    out = {}
    for k, v in metrics.items():
        try:
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    out[k] = float(v.detach().cpu().item())
                else:
                    out[k] = float(v.detach().cpu().mean().item())
                    out[f"{k}_was_multi"] = True
            elif isinstance(v, (float, int)):
                out[k] = v
            elif isinstance(v, _np.ndarray):
                out[k] = float(_np.mean(v))
            elif v is None:
                out[k] = None
            else:
                # fallback: try to cast
                out[k] = float(v)
        except Exception:
            # as last resort, store string repr
            out[k] = str(v)
    return out


class ParkingTrainingModuleReal(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(ParkingTrainingModuleReal, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.traj_point_loss_func = self.get_loss_function()
        # optional losses
        self.use_segmentation_loss = getattr(self.cfg, 'use_segmentation', False)
        self.use_depth_loss = getattr(self.cfg, 'use_depth', False)
        if self.use_depth_loss:
            self.depth_loss_func = DepthLoss(self.cfg)
        else:
            self.depth_loss_func = None
        if self.use_segmentation_loss:
            # prepare class weights (may be None, list, or tensor); SegmentationLoss handles conversion
            seg_weights = getattr(self.cfg, 'seg_vehicle_weights', None)
            self.seg_loss_func = SegmentationLoss(seg_weights)
        else:
            self.seg_loss_func = None

        self.parking_model = ParkingModelReal(self.cfg)

        # optional learnable loss weights (Kendall)
        # Safety: we initialize log_sigma values from config, and will clamp them during forward to avoid divergence.
        self.use_learnable_loss_weights = getattr(self.cfg, 'use_learnable_loss_weights', False)
        if self.use_learnable_loss_weights:
            init_val = float(getattr(self.cfg, 'learnable_log_sigma_init', 0.0))
            self.log_sigma_traj = nn.Parameter(torch.full((1,), init_val))
            self.log_sigma_smooth = nn.Parameter(torch.full((1,), init_val))
            self.log_sigma_curv = nn.Parameter(torch.full((1,), init_val))
            # clamp bounds
            self._log_sigma_min = float(getattr(self.cfg, 'learnable_log_sigma_min', -8.0))
            self._log_sigma_max = float(getattr(self.cfg, 'learnable_log_sigma_max', 8.0))
        else:
            self.log_sigma_traj = None
            self.log_sigma_smooth = None
            self.log_sigma_curv = None

        # Warmup / staged auxiliary-loss options
        self.use_aux_warmup = getattr(self.cfg, 'use_aux_loss_warmup', False)
        self.aux_warmup_epochs = int(getattr(self.cfg, 'aux_warmup_epochs', 5))
        self.aux_warmup_multiplier = float(getattr(self.cfg, 'aux_warmup_multiplier', 0.1))
        # whether learnable weights should be delayed until warmup completes
        self.learnable_warmup = getattr(self.cfg, 'learnable_warmup', False)

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_traj_point, pred_segmentation, pred_depth = self.parking_model(batch)

        # primary trajectory loss (token or regression)
        traj_loss = self.traj_point_loss_func(pred_traj_point, batch)
        train_loss = traj_loss
        loss_dict['traj_loss'] = traj_loss

        # optional depth and segmentation losses
        if self.use_depth_loss and (pred_depth is not None) and ('depth' in batch):
            depth_loss = self.depth_loss_func(pred_depth, batch['depth']) * float(getattr(self.cfg, 'depth_loss_weight', 1.0))
            train_loss = train_loss + depth_loss
            loss_dict.update({'depth_loss': depth_loss})

        if self.use_segmentation_loss and (pred_segmentation is not None) and ('segmentation' in batch):
            seg_loss = self.seg_loss_func(pred_segmentation, batch['segmentation']) * float(getattr(self.cfg, 'seg_loss_weight', 1.0))
            train_loss = train_loss + seg_loss
            loss_dict.update({'seg_loss': seg_loss})

        # ---- Kinematics / smoothness losses (computed from predicted tokens -> coords) ----
        smooth_loss = torch.tensor(0.0, device=self.device)
        curv_loss = torch.tensor(0.0, device=self.device)
        if getattr(self.cfg, 'use_smoothness_loss', False) or getattr(self.cfg, 'use_curvature_loss', False):
            # detokenize predicted tokens to coordinates for computing these losses
            # pred_traj_point is logits over tokens with shape (B, S, V)
            pred_tokens = pred_traj_point.argmax(dim=-1)  # (B, S)
            coords = detokenize_pred_tokens(pred_tokens, self.cfg.token_nums, self.cfg.item_number, self.cfg.autoregressive_points, xy_max=self.cfg.xy_max)
            # coords shape: (B, T, item_number) where item_number >= 2 (x,y,...)
            # ensure we take only x,y
            coords_xy = coords[:, :, :2]
            # Adjust smoothness and curvature loss weights for better stability
            smoothness_loss_weight = float(getattr(self.cfg, 'smoothness_loss_weight', 0.001))
            curvature_loss_weight = float(getattr(self.cfg, 'curvature_loss_weight', 0.001))
            # apply warmup multiplier if configured and we are within warmup epochs
            aux_multiplier = 1.0
            if self.use_aux_warmup and hasattr(self, 'current_epoch') and (self.current_epoch < self.aux_warmup_epochs):
                aux_multiplier = float(self.aux_warmup_multiplier)
            smoothness_loss_weight = smoothness_loss_weight * aux_multiplier
            curvature_loss_weight = curvature_loss_weight * aux_multiplier

            if getattr(self.cfg, 'use_smoothness_loss', False):
                smooth_loss = compute_acceleration_loss(coords_xy) * smoothness_loss_weight
                loss_dict['smooth_loss'] = smooth_loss
            if getattr(self.cfg, 'use_curvature_loss', False):
                curv_loss = compute_curvature_loss(coords_xy) * curvature_loss_weight
                loss_dict['curv_loss'] = curv_loss

        # combine with optional learnable weights
        if self.use_learnable_loss_weights and self.log_sigma_traj is not None:
            # Optionally delay use of learnable weights until warmup completes
            if self.learnable_warmup and self.use_aux_warmup and hasattr(self, 'current_epoch') and (self.current_epoch < self.aux_warmup_epochs):
                # During warmup, use fixed weighted sum (no learnable scalars)
                train_loss = train_loss + smooth_loss + curv_loss
            else:
                # Kendall-style weighting with safety clamps and 0.5 scaling: 0.5*(exp(-s) * L + s)
                total = 0.0
                s_t = torch.clamp(self.log_sigma_traj, min=self._log_sigma_min, max=self._log_sigma_max)
                total = total + 0.5 * (torch.exp(-s_t) * traj_loss + s_t)
                if getattr(self.cfg, 'use_smoothness_loss', False):
                    s_s = torch.clamp(self.log_sigma_smooth, min=self._log_sigma_min, max=self._log_sigma_max)
                    total = total + 0.5 * (torch.exp(-s_s) * smooth_loss + s_s)
                if getattr(self.cfg, 'use_curvature_loss', False):
                    s_c = torch.clamp(self.log_sigma_curv, min=self._log_sigma_min, max=self._log_sigma_max)
                    total = total + 0.5 * (torch.exp(-s_c) * curv_loss + s_c)
                train_loss = total
        else:
            train_loss = train_loss + smooth_loss + curv_loss

        # ---- Offline RL (dataset-only) fine-tune loss ----
        if getattr(self.cfg, 'use_offline_rl', False):
            try:
                # sample actions from current policy
                sampled_seq, logp_sums, _, _, _ = self.parking_model.sample_transformer(batch, max_samples=self.cfg.autoregressive_points * self.cfg.item_number)
                # compute reward from sampled sequence vs GT
                with torch.no_grad():
                    pred_tokens = sampled_seq
                    coords = detokenize_pred_tokens(pred_tokens, self.cfg.token_nums, self.cfg.item_number,
                                                    self.cfg.autoregressive_points, xy_max=self.cfg.xy_max)
                    coords_xy = coords[:, :, :2]
                    gt_flat = batch['gt_traj_point']
                    gt_xy = gt_flat.view(-1, self.cfg.autoregressive_points, self.cfg.item_number)[:, :, :2]
                    # reward: negative L2 distance over trajectory
                    diff = coords_xy - gt_xy.to(coords_xy.device)
                    reward = -torch.norm(diff, dim=-1).mean(dim=1)
                    reward = reward * float(getattr(self.cfg, 'offline_rl_reward_scale', 1.0))
                    if getattr(self.cfg, 'offline_rl_adv_normalize', True):
                        reward = (reward - reward.mean()) / (reward.std() + 1e-8)
                rl_weight = float(getattr(self.cfg, 'offline_rl_weight', 0.1))
                rl_loss = -(logp_sums * reward.detach()).mean() * rl_weight
                train_loss = train_loss + rl_loss
                loss_dict['offline_rl_loss'] = rl_loss
                loss_dict['offline_rl_reward'] = reward.mean()
            except Exception:
                # Keep training robust if offline RL path fails
                pass

        loss_dict.update({"train_loss": train_loss})
        self.log_dict(loss_dict)

        # Optional gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}
        pred_traj_point, pred_segmentation, pred_depth = self.parking_model(batch)

        val_traj_loss = self.traj_point_loss_func(pred_traj_point, batch)
        val_loss = val_traj_loss
        val_loss_dict['traj_loss'] = val_traj_loss

        if self.use_depth_loss and (pred_depth is not None) and ('depth' in batch):
            depth_loss = self.depth_loss_func(pred_depth, batch['depth']) * float(getattr(self.cfg, 'depth_loss_weight', 1.0))
            val_loss = val_loss + depth_loss
            val_loss_dict.update({'depth_loss': depth_loss})

        if self.use_segmentation_loss and (pred_segmentation is not None) and ('segmentation' in batch):
            seg_loss = self.seg_loss_func(pred_segmentation, batch['segmentation']) * float(getattr(self.cfg, 'seg_loss_weight', 1.0))
            val_loss = val_loss + seg_loss
            val_loss_dict.update({'seg_loss': seg_loss})

        # validation smooth/curvature
        val_smooth_loss = torch.tensor(0.0, device=self.device)
        val_curv_loss = torch.tensor(0.0, device=self.device)
        if getattr(self.cfg, 'use_smoothness_loss', False) or getattr(self.cfg, 'use_curvature_loss', False):
            pred_tokens = pred_traj_point.argmax(dim=-1)
            coords = detokenize_pred_tokens(pred_tokens, self.cfg.token_nums, self.cfg.item_number, self.cfg.autoregressive_points, xy_max=self.cfg.xy_max)
            coords_xy = coords[:, :, :2]
            if getattr(self.cfg, 'use_smoothness_loss', False):
                val_smooth_loss = compute_acceleration_loss(coords_xy) * float(getattr(self.cfg, 'smoothness_loss_weight', 0.1))
                val_loss = val_loss + val_smooth_loss
                val_loss_dict['smooth_loss'] = val_smooth_loss
            if getattr(self.cfg, 'use_curvature_loss', False):
                val_curv_loss = compute_curvature_loss(coords_xy) * float(getattr(self.cfg, 'curvature_loss_weight', 0.01))
                val_loss = val_loss + val_curv_loss
                val_loss_dict['curv_loss'] = val_curv_loss

        # if learning weights, apply same Kendall style at validation (use current log_sigma)
        if self.use_learnable_loss_weights and self.log_sigma_traj is not None:
            total = 0.0
            total = total + (torch.exp(-self.log_sigma_traj) * val_traj_loss + self.log_sigma_traj)
            if getattr(self.cfg, 'use_smoothness_loss', False):
                total = total + (torch.exp(-self.log_sigma_smooth) * val_smooth_loss + self.log_sigma_smooth)
            if getattr(self.cfg, 'use_curvature_loss', False):
                total = total + (torch.exp(-self.log_sigma_curv) * val_curv_loss + self.log_sigma_curv)
            val_loss = total

        # If learnable loss weights are used, log their current scalar values (avoid logging tensors with >1 elem)
        if self.use_learnable_loss_weights and self.log_sigma_traj is not None:
            try:
                val_loss_dict['log_sigma_traj'] = float(self.log_sigma_traj.detach().cpu().item())
            except Exception:
                val_loss_dict['log_sigma_traj'] = None
            try:
                val_loss_dict['log_sigma_smooth'] = float(self.log_sigma_smooth.detach().cpu().item()) if self.log_sigma_smooth is not None else 0.0
            except Exception:
                val_loss_dict['log_sigma_smooth'] = None
            try:
                val_loss_dict['log_sigma_curv'] = float(self.log_sigma_curv.detach().cpu().item()) if self.log_sigma_curv is not None else 0.0
            except Exception:
                val_loss_dict['log_sigma_curv'] = None

        val_loss_dict.update({"val_loss": val_loss})

        customized_metric = CustomizedMetric(self.cfg, pred_traj_point, batch)
        val_loss_dict.update(customized_metric.calculate_distance(pred_traj_point, batch))

        val_loss_dict = _sanitize_metrics(val_loss_dict)
        self.log_dict(val_loss_dict)

        return val_loss

    def configure_optimizers(self):
        # If learnable loss weights are used, put their params in a separate param group with smaller lr
        base_lr = float(self.cfg.learning_rate)
        weight_decay = float(self.cfg.weight_decay)
        if self.use_learnable_loss_weights and self.log_sigma_traj is not None:
            # collect log_sigma params
            sigma_params = [p for p in [self.log_sigma_traj, self.log_sigma_smooth, self.log_sigma_curv] if p is not None]
            # compare by id to avoid ambiguous tensor boolean comparisons
            sigma_param_ids = {id(p) for p in sigma_params}
            other_params = [p for n, p in self.named_parameters() if id(p) not in sigma_param_ids]
            optimizer = torch.optim.Adam([
                {'params': other_params},
                {'params': sigma_params, 'lr': base_lr * 0.1}
            ], lr=base_lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=base_lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_loss_function(self):
        traj_point_loss_func = None
        if self.cfg.decoder_method == "transformer":
            traj_point_loss_func = TokenTrajPointLoss(self.cfg)
        elif self.cfg.decoder_method == "gru":
            traj_point_loss_func = TrajPointLoss(self.cfg)
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        return traj_point_loss_func