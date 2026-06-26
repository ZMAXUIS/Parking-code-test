import torch
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet18

from utils.config import Configuration


class BevEncoder(nn.Module):
    def __init__(self, in_channel, use_interactive_attention=False, cfg=None):
        super().__init__()
        self.use_interactive_attention = use_interactive_attention

        if self.use_interactive_attention:
            self.encoder = InteractiveAttentionEncoder(cfg)
        else:
            trunk = resnet18(weights=None, zero_init_residual=True)

            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = trunk.bn1
            self.relu = trunk.relu
            self.max_pool = trunk.maxpool

            self.layer1 = trunk.layer1
            self.layer2 = trunk.layer2
            self.layer3 = trunk.layer3
            self.layer4 = trunk.layer4

    def forward(self, x, target_feature=None, flatten=True):
        if self.use_interactive_attention:
            return self.encoder(target_feature, x)

        # tolerate inputs without batch dim (C,H,W) -> add batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"BevEncoder.forward expected 4D input (N,C,H,W), got shape {tuple(x.shape)}")
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if flatten:
            x = torch.flatten(x, 2)
        return x


class BevQuery(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.query_en_dim, nhead=self.cfg.query_en_heads, batch_first=True, dropout=self.cfg.query_en_dropout)
        self.tf_query = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.query_en_layers)

        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.query_en_bev_length, self.cfg.query_en_dim) * .02)


        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, tgt_feature, img_feature):
        # allow mismatched shapes (channels or spatial) by adapting img_feature to tgt_feature
        if tgt_feature.shape != img_feature.shape:
            # adapt spatial size first
            tb, tc, th, tw = tgt_feature.shape
            ib, ic, ih, iw = img_feature.shape
            # if batch size mismatches, try to broadcast or raise
            if tb != ib:
                raise ValueError(f"Batch size mismatch between target ({tb}) and image ({ib}) features")
            # adapt channels: truncate or pad zeros
            if ic != tc:
                if ic > tc:
                    img_feature = img_feature[:, :tc, :, :]
                else:
                    pad = torch.zeros((ib, tc-ic, ih, iw), device=img_feature.device, dtype=img_feature.dtype)
                    img_feature = torch.cat([img_feature, pad], dim=1)
                    ic = tc
            # adapt spatial size: interpolate image feature to target spatial resolution
            if (ih, iw) != (th, tw):
                img_feature = F.interpolate(img_feature, size=(th, tw), mode='bilinear', align_corners=False)
        batch_size, channel, h, w = tgt_feature.shape

        tgt_feature = tgt_feature.view(batch_size, channel, -1)
        img_feature = img_feature.view(batch_size, channel, -1)
        tgt_feature = tgt_feature.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]
        img_feature = img_feature.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]

        # adapt pos_embed to (1, seq_len, embed_dim) if necessary
        seq_len = tgt_feature.size(1)
        embed_dim = tgt_feature.size(2)
        pos = self.pos_embed
        # pos shape: (1, P_seq, P_dim)
        P_seq = pos.size(1)
        P_dim = pos.size(2)
        # adjust seq length via interpolation along seq dim if needed
        if P_seq != seq_len:
            # interpolate: reshape to (1, dim, seq) -> (1, dim, seq_len)
            pos_t = pos.permute(0, 2, 1)  # (1, P_dim, P_seq)
            pos_t = F.interpolate(pos_t, size=seq_len, mode='linear', align_corners=False)
            pos = pos_t.permute(0, 2, 1)
        # adjust embedding dim by truncation or padding
        if P_dim != embed_dim:
            if P_dim > embed_dim:
                pos = pos[:, :, :embed_dim]
            else:
                pad = torch.zeros((1, seq_len, embed_dim - P_dim), device=pos.device, dtype=pos.dtype)
                pos = torch.cat([pos, pad], dim=2)

        tgt_feature = tgt_feature + pos
        img_feature = img_feature + pos

        bev_feature = self.tf_query(tgt_feature, memory=img_feature)
        bev_feature = bev_feature.permute(0, 2, 1)

        bev_feature = bev_feature.view(batch_size, channel, h, w)
        return bev_feature


class InteractiveAttentionEncoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        # Transformer layers for bidirectional attention
        self.target_to_bev = nn.TransformerDecoderLayer(d_model=self.cfg.query_en_dim, nhead=self.cfg.query_en_heads, batch_first=True, dropout=self.cfg.query_en_dropout)
        self.bev_to_target = nn.TransformerDecoderLayer(d_model=self.cfg.query_en_dim, nhead=self.cfg.query_en_heads, batch_first=True, dropout=self.cfg.query_en_dropout)

        # GRU for dynamic fusion
        self.gru = nn.GRU(input_size=self.cfg.query_en_dim, hidden_size=self.cfg.query_en_dim, batch_first=True)

        # Optional convolutional layer, group normalization, and activation
        self.conv = nn.Conv2d(self.cfg.query_en_dim, self.cfg.query_en_dim, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(1, self.cfg.query_en_dim)
        self.activation = nn.GELU()  # Changed activation to GELU

    def forward(self, target_feature, bev_feature):
        # Bidirectional attention
        target_to_bev = self.target_to_bev(target_feature, memory=bev_feature)
        bev_to_target = self.bev_to_target(bev_feature, memory=target_feature)

        # Dynamic fusion using GRU
        fused_feature, _ = self.gru(target_to_bev + bev_to_target)

        # Optional convolutional layer, group normalization, and activation
        fused_feature = fused_feature.permute(0, 2, 1).view(fused_feature.size(0), -1, int(fused_feature.size(1)**0.5), int(fused_feature.size(1)**0.5))
        fused_feature = self.conv(fused_feature)
        fused_feature = self.gn(fused_feature)
        fused_feature = self.activation(fused_feature)

        return fused_feature


# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         return x * y


# class BEVTfEncoder(nn.Module):
#     def __init__(self, cfg: Configuration, input_dim):
#         super().__init__()
#         self.cfg = cfg

#         tf_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=self.cfg.tf_en_heads)
#         self.tf_encoder = nn.TransformerEncoder(tf_layer, num_layers=self.cfg.tf_en_layers)

#         self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.tf_en_bev_length, input_dim) * .02)
#         self.pos_drop = nn.Dropout(self.cfg.tf_en_dropout)

#         self.init_weights()

#     def init_weights(self):
#         for name, p in self.named_parameters():
#             if 'pos_embed' in name:
#                 continue
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         trunc_normal_(self.pos_embed, std=.02)

#     def forward(self, bev_feature, mode):
#         bev_feature = bev_feature
#         if mode == "train":
#             bev_feature = self.pos_drop(bev_feature)
#         bev_feature = bev_feature.transpose(0, 1)
#         bev_feature = self.tf_encoder(bev_feature)
#         bev_feature = bev_feature.transpose(0, 1)
#         return bev_feature
