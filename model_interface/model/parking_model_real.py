import torch
from torch import nn

from model_interface.model.bev_encoder import BevEncoder, BevQuery
from model_interface.model.gru_trajectory_decoder import GRUTrajectoryDecoder
from model_interface.model.lss_bev_model import LssBevModel
from model_interface.model.trajectory_decoder import TrajectoryDecoder
from model_interface.model.segmentation_head import SegmentationHead
from model_interface.model.dynamic_query_pruner import DynamicQueryPruner
from model_interface.model.history_encoder import HistoryEncoder
from utils.config import Configuration


class ParkingModelReal(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        # Camera Encoder
        self.lss_bev_model = LssBevModel(self.cfg)
        self.image_res_encoder = BevEncoder(in_channel=self.cfg.bev_encoder_in_channel)

        # Target Encoder
        self.target_res_encoder = BevEncoder(in_channel=1)

        # BEV Query
        self.bev_query = BevQuery(self.cfg)

        # optional segmentation head
        self.segmentation_head = SegmentationHead(self.cfg) if getattr(self.cfg, 'use_segmentation', False) else None

        # optional dynamic query pruner
        self.query_pruner = DynamicQueryPruner(self.cfg) if getattr(self.cfg, 'use_query_pruner', False) else None

        # History encoder (optional)
        self.history_encoder = HistoryEncoder(self.cfg) if getattr(self.cfg, 'use_history', False) else None

        # Trajectory Decoder
        self.trajectory_decoder = self.get_trajectory_decoder()

        # Value head for critic (used by PPO). We'll pool BEV feature in forward and pass through this head.
        # Use cfg.query_en_dim as expected BEV channel size; if mismatch occurs we will adapt at runtime.
        vh_in = getattr(self.cfg, 'query_en_dim', 256)
        self.value_head = nn.Sequential(
            nn.Linear(vh_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self._value_proj = None

    def forward(self, data):
        # Encoder (now also returns bev_camera_encoder to avoid re-encoding raw images)
        bev_feature, pred_depth, bev_target, pred_segmentation, hist_feats, bev_camera_encoder = self.encoder(data,
                                                                                                              mode="train")

        # Decoder
        # ensure gt token on same device as bev_feature
        device = bev_feature.device
        # bev_camera_encoder is provided by encoder; compute target encoder from bev_target
        bev_target_encoder = self.target_res_encoder(self.get_target_bev(data['target_point'].to(device), mode="train"),
                                                     flatten=False) if hasattr(self, 'target_res_encoder') else None
        pred_traj_point = self.trajectory_decoder(bev_feature, data['gt_traj_point_token'].to(device),
                                                  bev_camera_encoder, bev_target_encoder, hist_feats)

        return pred_traj_point, pred_segmentation, pred_depth

    def predict_transformer(self, data, predict_token_num):
        # Encoder
        bev_feature, pred_depth, bev_target, pred_segmentation, hist_feats, bev_camera_encoder = self.encoder(data,
                                                                                                              mode="predict")

        # Auto Regressive Decoder
        # During inference, we regard BOS as gt_traj_point_token.
        device = bev_feature.device
        autoregressive_point = data['gt_traj_point_token'].to(device)
        # bev_camera_encoder already returned by encoder; prepare bev_target_encoder
        bev_target_encoder = self.target_res_encoder(bev_target, flatten=False) if hasattr(self,
                                                                                           'target_res_encoder') else None
        for _ in range(predict_token_num):
            pred_traj_point = self.trajectory_decoder.predict(bev_feature, autoregressive_point, bev_camera_encoder,
                                                              bev_target_encoder, hist_feats)
            autoregressive_point = torch.cat([autoregressive_point, pred_traj_point], dim=1)

        return autoregressive_point, pred_segmentation, pred_depth, bev_target

    def predict_gru(self, data):
        # Encoder
        bev_feature, pred_depth, bev_target, pred_segmentation, hist_feats, bev_camera_encoder = self.encoder(data,
                                                                                                              mode="predict")

        # Decoder
        # provide encoders for possible gru decoder using same interface
        bev_target_encoder = self.target_res_encoder(bev_target, flatten=False) if hasattr(self,
                                                                                           'target_res_encoder') else None
        autoregressive_point = self.trajectory_decoder(bev_feature, bev_camera_encoder=bev_camera_encoder,
                                                       bev_target_encoder=bev_target_encoder, hist_feats=hist_feats)
        return autoregressive_point, pred_segmentation, pred_depth, bev_target

    def encoder(self, data, mode):
        # Camera Encoder
        # Move images/intrinsics/extrinsics to cfg.device initially; lss_bev_model may place outputs on that device
        cfg_device = getattr(self.cfg, 'device', torch.device('cpu'))
        images = data['image'].to(cfg_device, non_blocking=True)
        intrinsics = data['intrinsics'].to(cfg_device, non_blocking=True)
        extrinsics = data['extrinsics'].to(cfg_device, non_blocking=True)
        bev_camera, pred_depth = self.lss_bev_model(images, intrinsics, extrinsics)
        # derive canonical device from bev_camera (ensures encoders and target maps are placed consistently)
        device = bev_camera.device
        # ensure encoder modules are on the same device as bev_camera to avoid dtype/device mismatch
        try:
            self.image_res_encoder.to(device)
        except Exception:
            pass
        try:
            self.target_res_encoder.to(device)
        except Exception:
            pass
        if self.segmentation_head is not None:
            try:
                self.segmentation_head.to(device)
            except Exception:
                pass
        if self.history_encoder is not None:
            try:
                self.history_encoder.to(device)
            except Exception:
                pass
        bev_camera_encoder = self.image_res_encoder(bev_camera.to(device), flatten=False)

        # Target Encoder
        target_point = data['fuzzy_target_point'] if self.cfg.use_fuzzy_target else data['target_point']
        target_point = target_point.to(device, non_blocking=True)
        bev_target = self.get_target_bev(target_point, mode=mode)
        bev_target_encoder = self.target_res_encoder(bev_target, flatten=False)

        # apply dynamic pruning on camera BEV feature guided by target encoder (if enabled)
        if self.query_pruner is not None:
            bev_camera_encoder = self.query_pruner(bev_camera_encoder, bev_target_encoder)

        # History encoder (optional)
        hist_feats = None
        if self.history_encoder is not None:
            # data may contain 'history_traj' as (B, hist_len, 2)
            raw_hist = data.get('history_traj', None)
            if raw_hist is not None:
                hist_feats = self.history_encoder(raw_hist.to(device, non_blocking=True))
            else:
                # if missing, provide None (decoder will ignore)
                hist_feats = None

        # Feature Fusion
        bev_feature = self.get_feature_fusion(bev_target_encoder, bev_camera_encoder)

        bev_feature = torch.flatten(bev_feature, 2)

        # optional segmentation prediction
        pred_segmentation = None
        if self.segmentation_head is not None:
            pred_segmentation = self.segmentation_head(bev_feature)

        # return bev_camera_encoder as additional output so callers can reuse it without re-encoding raw images
        return bev_feature, pred_depth, bev_target, pred_segmentation, hist_feats, bev_camera_encoder

    def get_target_bev(self, target_point, mode):
        """Create a BEV target map using a Gaussian distribution centered at the target point
        instead of a hard binary square. The spread (sigma) and kernel range are configurable
        via cfg: 'target_sigma_pixels' and 'target_range'. If these are missing, reasonable
        defaults are used.
        """
        # derive device from provided target_point tensor if available, otherwise fall back to cfg.device
        if isinstance(target_point, torch.Tensor):
            device = target_point.device
        else:
            device = getattr(self.cfg, 'device', torch.device('cpu'))

        h = int((self.cfg.bev_y_bound[1] - self.cfg.bev_y_bound[0]) / self.cfg.bev_y_bound[2])
        w = int((self.cfg.bev_x_bound[1] - self.cfg.bev_x_bound[0]) / self.cfg.bev_x_bound[2])
        b = self.cfg.batch_size if mode == "train" else 1

        bev_target = torch.zeros((b, 1, h, w), dtype=torch.float, device=device)

        # compute pixel coordinates for target points
        # expect target_point shape: (B, 2) with (x, y) in meters
        x_pixel = ((h / 2.0) + target_point[:, 0] / self.cfg.bev_x_bound[2]).long()
        y_pixel = ((w / 2.0) + target_point[:, 1] / self.cfg.bev_y_bound[2]).long()
        coord = torch.stack([x_pixel, y_pixel], dim=1)

        # optional noise
        if getattr(self.cfg, 'add_noise_to_target', False) and mode == "train":
            noise_threshold = int(self.cfg.target_noise_threshold / self.cfg.bev_x_bound[2])
            noise = (torch.rand_like(coord, dtype=torch.float,
                                     device=device) * noise_threshold * 2 - noise_threshold).long()
            coord = coord + noise

        # gaussian kernel parameters
        sigma = float(getattr(self.cfg, 'target_sigma_pixels', max(1.0, getattr(self.cfg, 'target_range', 1) / (
            self.cfg.bev_x_bound[2] if self.cfg.bev_x_bound[2] != 0 else 1))))
        # radius: how far from center we apply kernel (in pixels)
        radius = int(
            getattr(self.cfg, 'target_range', 1) / (self.cfg.bev_x_bound[2] if self.cfg.bev_x_bound[2] != 0 else 1))
        radius = max(1, radius)

        # precompute a local grid used for kernel generation (square of size 2*radius+1)
        ks = 2 * radius + 1
        dx = torch.arange(-radius, radius + 1, device=device).view(1, -1).float()
        dy = torch.arange(-radius, radius + 1, device=device).view(-1, 1).float()
        # dx and dy will be broadcastable to (ks, ks)

        for batch in range(b):
            cx, cy = int(coord[batch, 0].item()), int(coord[batch, 1].item())
            # clamp center inside image
            if cx < 0 or cx >= h or cy < 0 or cy >= w:
                continue

            x0 = max(0, cx - radius)
            x1 = min(h - 1, cx + radius)
            y0 = max(0, cy - radius)
            y1 = min(w - 1, cy + radius)

            # compute local ranges and corresponding indices in kernel
            local_x = torch.arange(x0, x1 + 1, device=device).float()
            local_y = torch.arange(y0, y1 + 1, device=device).float()
            gx = local_x.unsqueeze(1) - cx
            gy = local_y.unsqueeze(0) - cy
            # gaussian on local grid
            # note: gx is (nx,1), gy is (1,ny) so we want (nx, ny) distances
            dist2 = gx.pow(2) + gy.pow(2)
            kernel = torch.exp(-0.5 * dist2 / (sigma * sigma))

            # normalize kernel to max 1.0 (keeps scale comparable to previous binary map)
            if kernel.max() > 0:
                kernel = kernel / kernel.max()

            bev_target[batch, 0, x0:x1 + 1, y0:y1 + 1] = torch.maximum(bev_target[batch, 0, x0:x1 + 1, y0:y1 + 1],
                                                                       kernel)

        return bev_target

    def get_feature_fusion(self, bev_target_encoder, bev_camera_encoder):
        if self.cfg.fusion_method == "query":
            bev_feature = self.bev_query(bev_target_encoder, bev_camera_encoder)
        elif self.cfg.fusion_method == "plus":
            bev_feature = bev_target_encoder + bev_camera_encoder
        elif self.cfg.fusion_method == "concat":
            concat_feature = torch.concatenate([bev_target_encoder, bev_camera_encoder], dim=1)
            # create conv on the same device to avoid forcing CUDA in library code
            device = bev_target_encoder.device
            conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False).to(device)
            bev_feature = conv(concat_feature)
        else:
            raise ValueError(f"Don't support fusion_method '{self.cfg.fusion_method}'!")

        return bev_feature

    def get_trajectory_decoder(self):
        if self.cfg.decoder_method == "transformer":
            trajectory_decoder = TrajectoryDecoder(self.cfg)
        elif self.cfg.decoder_method == "gru":
            trajectory_decoder = GRUTrajectoryDecoder(self.cfg)
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")

        return trajectory_decoder

    def sample_transformer(self, data, max_samples=None, temperature=1.0):
        """Sequentially sample tokens from transformer policy. Returns sampled sequence (B, S)
        and sum_log_probs (B,).
        """
        device = getattr(self.cfg, 'device', torch.device('cpu'))
        bev_feature, pred_depth, bev_target, pred_segmentation, hist_feats, bev_camera_encoder = self.encoder(data,
                                                                                                              mode="predict")
        bev_target_encoder = self.target_res_encoder(bev_target, flatten=False) if hasattr(self,
                                                                                           'target_res_encoder') else None

        B = bev_feature.size(0)
        # seed with BOS token
        S = self.cfg.autoregressive_points * self.cfg.item_number + int(self.cfg.append_token)
        sampled = []
        logp_sums = torch.zeros(B, device=device)
        autoregressive_point = data['gt_traj_point_token'].to(device)
        max_iter = max_samples if max_samples is not None else (self.cfg.autoregressive_points * self.cfg.item_number)
        for _ in range(max_iter):
            logits = self.trajectory_decoder.predict_logits(bev_feature, autoregressive_point, bev_camera_encoder,
                                                            bev_target_encoder, hist_feats)
            # apply temperature and categorical sampling
            probs = torch.softmax(logits / temperature, dim=-1)
            distrib = torch.distributions.Categorical(probs)
            sample = distrib.sample().view(B, 1)
            logp = distrib.log_prob(sample.view(-1)).view(B)
            logp_sums = logp_sums + logp
            sampled.append(sample)
            autoregressive_point = torch.cat([autoregressive_point, sample], dim=1)

        sampled_seq = torch.cat(sampled, dim=1)
        return sampled_seq, logp_sums, pred_segmentation, pred_depth, bev_target

    def value(self, bev_feature):
        """Compute a scalar value per batch from bev_feature (B, C, L).
        We pool over spatial dim to (B, C) then apply value_head. If channel dim mismatch,
        adapt with a linear projection created lazily.
        """
        pooled = bev_feature.mean(dim=2)  # (B, C)
        C = pooled.size(1)
        expected = getattr(self.cfg, 'query_en_dim', C)
        if C != expected:
            if self._value_proj is None:
                # create projection to expected dim
                self._value_proj = nn.Linear(C, expected).to(pooled.device)
            pooled = self._value_proj(pooled)
        value = self.value_head(pooled).squeeze(-1)
        return value

    def compute_logprob_transformer(self, data, actions):
        """Compute sum log-prob of given actions (tokens) under current policy for each batch.
        actions: (B, T) long tensor containing successive tokens sampled earlier.
        Returns: logp_sums tensor shape (B,)
        """
        device = getattr(self.cfg, 'device', torch.device('cpu'))
        bev_feature, pred_depth, bev_target, pred_segmentation, hist_feats, bev_camera_encoder = self.encoder(data,
                                                                                                              mode="predict")
        bev_target_encoder = self.target_res_encoder(bev_target, flatten=False) if hasattr(self,
                                                                                           'target_res_encoder') else None

        B = bev_feature.size(0)
        logp_sums = torch.zeros(B, device=device)
        autoregressive_point = data['gt_traj_point'].to(device)
        max_iter = actions.size(1)
        for t in range(max_iter):
            logits = self.trajectory_decoder.predict_logits(bev_feature, autoregressive_point, bev_camera_encoder,
                                                            bev_target_encoder, hist_feats)
            probs = torch.softmax(logits, dim=-1)
            # actions[:, t] is the token to evaluate
            act = actions[:, t]
            # avoid invalid indices
            act = act.clamp(0, probs.size(-1) - 1)
            dist = torch.distributions.Categorical(probs)
            logp = dist.log_prob(act)
            logp_sums = logp_sums + logp
            autoregressive_point = torch.cat([autoregressive_point, act.view(B, 1)], dim=1)
        return logp_sums
