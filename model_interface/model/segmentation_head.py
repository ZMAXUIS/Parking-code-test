import math

import torch
import torch.nn.functional as F

from torch import nn
from tool.config import Configuration


class SegmentationHead(nn.Module):
    def __init__(self, cfg: Configuration):
        super(SegmentationHead, self).__init__()
        self.cfg = cfg

        # output channel to reduce to bev encoder in channel for top-down decoder
        self.out_channel = self.cfg.bev_encoder_in_channel
        # support both config types (tool.config.Configuration or utils.config.Configuration)
        # fall back to default 3 classes if not provided
        self.seg_classes = getattr(self.cfg, 'seg_classes', getattr(self.cfg, 'seg_vehicle_classes', None))
        if self.seg_classes is None:
            self.seg_classes = 3

        # we may not know the exact input channel (depends on BEV encoder/backbone), so defer creation
        # of the initial 1x1 conv until we see the actual tensor in forward
        self.c5_conv = None
        self.relu = nn.ReLU(inplace=True)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # defer creation of up_conv* and segmentation_head until we know input device/dtype
        self.up_conv5 = None
        self.up_conv4 = None
        self.up_conv3 = None
        self.segmentation_head = None

    def top_down(self, x):
        # ensure c5_conv is created with correct in_channels
        if self.c5_conv is None or self.c5_conv.in_channels != x.shape[1]:
            in_ch = x.shape[1]
            # create conv and immediately move it to the device/dtype of input x to avoid
            # mismatches when modules are created after model.to(device).
            device = x.device
            dtype = x.dtype
            conv = nn.Conv2d(in_ch, self.out_channel, (1, 1))
            conv = conv.to(device=device, dtype=dtype)
            self.c5_conv = conv
        # create up_conv and segmentation_head lazily on the same device/dtype as input x
        device = x.device
        dtype = x.dtype
        if self.up_conv5 is None:
            self.up_conv5 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=(1, 1)).to(device=device, dtype=dtype)
        if self.up_conv4 is None:
            self.up_conv4 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=(1, 1)).to(device=device, dtype=dtype)
        if self.up_conv3 is None:
            self.up_conv3 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=(1, 1)).to(device=device, dtype=dtype)
        if self.segmentation_head is None:
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channel, self.seg_classes, kernel_size=1, padding=0)
            ).to(device=device, dtype=dtype)

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.up_sample(p5)))
        p3 = self.relu(self.up_conv4(self.up_sample(p4)))
        p2 = self.relu(self.up_conv3(self.up_sample(p3)))
        p1 = F.interpolate(p2, size=(200, 200), mode="bilinear", align_corners=False)
        return p1

    def forward(self, fuse_feature):
        # fuse_feature: [batch, channel, seq_len] -> transpose -> [batch, seq_len, channel]
        fuse_feature_t = fuse_feature.transpose(1, 2)
        b, s, c = fuse_feature_t.shape
        # reshape to BEV: [batch, channel, H, W], where H=W=sqrt(seq_len)
        h_w = int(math.sqrt(s))
        fuse_bev = torch.reshape(fuse_feature_t, (b, c, h_w, h_w))
        fuse_bev = self.top_down(fuse_bev)
        pred_segmentation = self.segmentation_head(fuse_bev)
        return pred_segmentation
