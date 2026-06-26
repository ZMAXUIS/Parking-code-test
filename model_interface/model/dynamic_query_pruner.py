import torch
from torch import nn


class DynamicQueryPruner(nn.Module):
    """Compute relevance of BEV spatial locations w.r.t. the target encoding and apply
    a soft or hard pruning mask.

    Config (cfg) options (all optional):
      - use_query_pruner (bool): enable/disable (parking_model_real will check)
      - query_prune_topk (int): if set, keep top-k spatial locations (hard prune)
      - query_prune_ratio (float in (0,1]): alternatively keep top-k = ratio * L
      - query_prune_soft (bool): if True, use soft weighting (multiply by normalized scores)
      - query_prune_temperature (float): temperature for softmax when computing weights

    The module is intentionally conservative: if no cfg flags present, it returns
    the input unchanged.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # read config options with safe fallbacks
        self.topk = getattr(cfg, 'query_prune_topk', None)
        self.ratio = getattr(cfg, 'query_prune_ratio', None)
        self.soft = getattr(cfg, 'query_prune_soft', True)
        self.temperature = float(getattr(cfg, 'query_prune_temperature', 1.0))

        # normalize parameters
        if self.topk is None and self.ratio is None:
            # default: do not prune, just soft-weight
            self._enabled = getattr(cfg, 'use_query_pruner', False)
        else:
            self._enabled = True

    def forward(self, bev_feature, bev_target_encoder=None):
        """bev_feature: (B, C, H, W)
           bev_target_encoder: (B, C, H, W) or None

           Returns same-shape tensor (B, C, H, W). If hard-pruning is configured,
           low-importance locations are zeroed out (mask). If soft, locations are
           multiplied by normalized importance scores.
        """
        if not getattr(self.cfg, 'use_query_pruner', False):
            return bev_feature

        B, C, H, W = bev_feature.shape
        device = bev_feature.device
        dtype = bev_feature.dtype

        L = H * W
        # compute a target prototype vector; if bev_target_encoder provided, pool it
        if bev_target_encoder is not None:
            # (B, C, H, W) -> (B, C)
            target_pool = bev_target_encoder.mean(dim=(2, 3))
        else:
            # fallback: global average of bev_feature
            target_pool = bev_feature.mean(dim=(2, 3))

        # flatten spatial
        feat = bev_feature.view(B, C, -1)  # (B, C, L)
        # compute relevance as dot product between each spatial vector and target_pool
        # target_pool: (B, C) -> (B, C, 1)
        tp = target_pool.unsqueeze(-1)  # (B, C, 1)
        scores = (feat * tp).sum(dim=1)  # (B, L)

        # normalize scores
        if self.temperature != 1.0:
            scores = scores / self.temperature

        # soft weights
        weights = torch.softmax(scores, dim=-1)  # (B, L)

        # decide hard/topk
        if self.topk is None and self.ratio is not None:
            k = max(1, int(L * float(self.ratio)))
        elif self.topk is not None:
            k = int(self.topk)
        else:
            k = None

        if k is not None and not self.soft:
            # hard top-k: produce mask with 1 for topk and 0 elsewhere
            topk_vals, topk_idx = torch.topk(weights, k=k, dim=-1)
            mask = torch.zeros_like(weights, device=device)
            arange = torch.arange(B, device=device).unsqueeze(1)
            mask[arange, topk_idx] = 1.0
            mask = mask.unsqueeze(1)  # (B,1,L)
            feat = feat * mask
            out = feat.view(B, C, H, W)
            return out
        else:
            # soft weighting (default). Multiply features by weights
            weights = weights.unsqueeze(1)  # (B,1,L)
            feat = feat * weights
            out = feat.view(B, C, H, W)
            return out

