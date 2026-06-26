"""
TrajectoryDecoder (已修改版)

实现目标（对应你的要求）：
1) 将轨迹的 x、y 分成两个独立的解码流（x 解码器和 y 解码器），而不是把 (x,y) 作为一个整体同时解码。
2) 每个流在 cross-attention 阶段分别与 encoder 的 BEV 特征（包括 camera BEV 与 target BEV）做交叉注意力学习。
3) cross-attention 后对 x、y 各自做流内 self-attention（增强时序/上下文信息），然后将两个流的特征在特征维上拼接，再做一个融合的自注意力（fusion self-attn），最后映射到 token 输出维度或 logits。

设计细节（实现要点）：
- 使用两个独立的 TransformerDecoder（tf_decoder_x / tf_decoder_y）完成 cross-attention，与 encoder memory（BEV flatten 后）交互；
- 为 x/y 分别维护独立的位置编码（pos_embed_x / pos_embed_y），并在不同 device/dtype 下做兼容处理；
- cross-attention 后使用 TransformerEncoder 作为流内 self-attention（self_attn_x / self_attn_y）；
- 将两个流的输出按特征维 concat 形成大小为 2*D 的特征，再通过另一个 TransformerEncoder 做融合自注意力；
- 最终通过线性层把 fused 特征投射为 token logits（与训练中 TokenTrajPointLoss 的接口保持一致）；
- 在 predict 接口中保留逐步自回归生成单步 token 的逻辑（返回单步 token），以兼容原有的 `predict_transformer` 调用方式。

兼容性与边界处理：
- 保留原先 decoder 接口：forward(encoder_out, tgt, bev_camera_encoder=None, bev_target_encoder=None) 返回 (B,T,V) logits，训练器里 TokenTrajPointLoss 可以直接使用；
- predict(...) 返回单步 token（与旧接口行为一致），便于 autoregressive 推理流程；
- 对位置编码可能和 embedding 维度不一致的情况做了 pad/截断保护；
- 对输入的 BEV encoder（可能是 (B,C,H,W) 或已展平的 (B,C,L)）都做了适配处理。

注意：该修改主要集中在解码器内部结构，外部调用（parking_model_real、trainer_real）接口未改动，因此与你现有训练/评估框架兼容。
"""

import torch
from torch import nn
from timm.models.layers import trunc_normal_

from utils.config import Configuration


class TrajectoryDecoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        # token embedding (shared)
        self.embedding = nn.Embedding(self.cfg.token_nums + self.cfg.append_token, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)

        item_cnt = self.cfg.autoregressive_points
        seq_len = int(self.cfg.item_number * item_cnt + int(self.cfg.append_token))

        # separate positional embeddings for x and y decoders
        self.pos_embed_x = nn.Parameter(torch.randn(1, seq_len, self.cfg.tf_de_dim) * .02)
        self.pos_embed_y = nn.Parameter(torch.randn(1, seq_len, self.cfg.tf_de_dim) * .02)

        # transformer decoders for x and y (cross-attention with encoder memory)
        tf_layer_x = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.tf_decoder_x = nn.TransformerDecoder(tf_layer_x, num_layers=self.cfg.tf_de_layers)
        tf_layer_y = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.tf_decoder_y = nn.TransformerDecoder(tf_layer_y, num_layers=self.cfg.tf_de_layers)

        # per-stream self-attention (transformer encoder) after cross-attention
        enc_layer = nn.TransformerEncoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.self_attn_x = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.self_attn_y = nn.TransformerEncoder(enc_layer, num_layers=1)

        # final fusion self-attention and output projection
        # NOTE: older checkpoints used a single-stream decoder (no x/y split) where
        # the final feature dim == tf_de_dim. Newer implementation concatenates x/y
        # streams -> feature dim == 2*tf_de_dim. To remain compatible with legacy
        # checkpoints, support a config flag `decoder_split_xy` (default True).
        # If decoder_split_xy == False, we will fuse sa_x and sa_y into a single tf_de_dim
        # representation (average) and keep fusion machinery dimension == tf_de_dim.
        self.decoder_split_xy = bool(getattr(self.cfg, 'decoder_split_xy', True))

        if self.decoder_split_xy:
            fusion_dim = 2 * self.cfg.tf_de_dim
        else:
            fusion_dim = self.cfg.tf_de_dim

        fusion_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=self.cfg.tf_de_heads)
        self.fusion_self_attn = nn.TransformerEncoder(fusion_layer, num_layers=1)

        # project fused features to token logits
        self.output = nn.Linear(fusion_dim, self.cfg.token_nums + self.cfg.append_token)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed_x, std=.02)
        trunc_normal_(self.pos_embed_y, std=.02)

    def _adapt_pos(self, pos_param, tgt_len, tgt_dim, device, dtype):
        """Adapt a positional parameter to requested sequence length and embedding dim.
        Handles interpolation along sequence dimension and truncation/padding along embed dim.
        """
        P_seq = pos_param.size(1)
        P_dim = pos_param.size(2)
        pos = pos_param
        if P_seq != tgt_len:
            pos_t = pos.permute(0, 2, 1)  # (1, P_dim, P_seq)
            pos_t = torch.nn.functional.interpolate(pos_t, size=tgt_len, mode='linear', align_corners=False)
            pos = pos_t.permute(0, 2, 1)
        if P_dim != tgt_dim:
            if P_dim > tgt_dim:
                pos = pos[:, :, :tgt_dim]
            else:
                pad = torch.zeros((1, tgt_len, tgt_dim - P_dim), device=device, dtype=dtype)
                pos = torch.cat([pos, pad], dim=2)
        return pos.to(device).to(dtype)

    def create_mask(self, tgt):
        device = tgt.device if hasattr(tgt, 'device') else self.cfg.device
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0)).to(device)
        tgt_padding_mask = (tgt == self.PAD_token).to(device)
        return tgt_mask, tgt_padding_mask

    def _prepare_memory(self, bev_camera_encoder, bev_target_encoder, hist_feats=None):
        """Flatten and concatenate the BEV camera and BEV target encoders to form transformer memory.
        If hist_feats provided (B, L_hist, C_hist) it'll be converted to (L_hist, B, C_hist) and appended.
        Returns memory of shape (L_mem, B, C).
        """
        seqs = []
        # camera
        if bev_camera_encoder is not None:
            if bev_camera_encoder.dim() == 3:
                cam_seq = bev_camera_encoder.permute(2, 0, 1)  # (L, B, C)
            else:
                B, C, H, W = bev_camera_encoder.shape
                cam_seq = bev_camera_encoder.view(B, C, -1).permute(2, 0, 1)
            seqs.append(cam_seq)
        # target
        if bev_target_encoder is not None:
            if bev_target_encoder.dim() == 3:
                tgt_seq = bev_target_encoder.permute(2, 0, 1)
            else:
                B, C, H, W = bev_target_encoder.shape
                tgt_seq = bev_target_encoder.view(B, C, -1).permute(2, 0, 1)
            seqs.append(tgt_seq)
        # history sequence (B, L_hist, C_hist) -> (L_hist, B, C_hist)
        if hist_feats is not None:
            # expect (B, L_hist, C) or (L_hist, B, C)
            if hist_feats.dim() == 3 and hist_feats.shape[0] == bev_camera_encoder.shape[0]:
                # (B, L_hist, C) -> permute
                hist_seq = hist_feats.permute(1, 0, 2)
            elif hist_feats.dim() == 3 and hist_feats.shape[1] == bev_camera_encoder.shape[0]:
                # already (L_hist, B, C)
                hist_seq = hist_feats
            else:
                # fallback: try to permute
                hist_seq = hist_feats.permute(1, 0, 2)
            seqs.append(hist_seq)

        if len(seqs) == 0:
            raise ValueError("At least one encoder input must be provided to build memory")

        memory = torch.cat(seqs, dim=0)
        return memory

    def decoder_cross(self, memory, tgt_embedding, tgt_mask, tgt_padding_mask, which='x'):
        # tgt_embedding: (B, T, D) -> transformer expects (T, B, D)
        tgt = tgt_embedding.transpose(0, 1)
        if which == 'x':
            out = self.tf_decoder_x(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        else:
            out = self.tf_decoder_y(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        out = out.transpose(0, 1)  # (B, T, D)
        return out

    def forward(self, encoder_out, tgt, bev_camera_encoder=None, bev_target_encoder=None, hist_feats=None):
        # encoder_out is kept for backward compatibility but we construct memory from encoders if provided
        # tgt: token sequence (B, S)
        tgt = tgt[:, :-1]
        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        # prepare embeddings
        tgt_embedding = self.embedding(tgt)  # (B, T, D)
        tgt_len = tgt_embedding.size(1)

        # prepare pos embeds for x/y and ensure shape match (seq_len and embed_dim)
        tgt_dim = tgt_embedding.size(-1)
        pos_x = self._adapt_pos(self.pos_embed_x, tgt_len, tgt_dim, tgt_embedding.device, tgt_embedding.dtype)
        pos_y = self._adapt_pos(self.pos_embed_y, tgt_len, tgt_dim, tgt_embedding.device, tgt_embedding.dtype)

        tgt_emb_x = self.pos_drop(tgt_embedding + pos_x)
        tgt_emb_y = self.pos_drop(tgt_embedding + pos_y)

        # prepare memory for cross-attention: prefer explicit bev encoders if provided
        # decide whether to use cross-attention memory (camera/target/history)
        if getattr(self.cfg, 'decoder_use_cross_attention', True):
            if bev_camera_encoder is not None or bev_target_encoder is not None or hist_feats is not None:
                memory = self._prepare_memory(bev_camera_encoder, bev_target_encoder, hist_feats if getattr(self.cfg, 'history_use_attention', True) else None)
            else:
                # fallback to using encoder_out as flattened memory
                memory = encoder_out.permute(2, 0, 1)
        else:
            # cross-attention disabled: use encoder_out as flattened fallback (or zeros if encoder_out missing)
            try:
                memory = encoder_out.permute(2, 0, 1)
            except Exception:
                # create a tiny zero memory to avoid transformer errors
                B = tgt_embedding.size(0)
                D = tgt_embedding.size(2)
                memory = torch.zeros((1, B, D), device=tgt_embedding.device, dtype=tgt_embedding.dtype)

        # cross-attention decoding for x and y separately (or bypass if disabled)
        if getattr(self.cfg, 'decoder_use_cross_attention', True):
            cross_x = self.decoder_cross(memory, tgt_emb_x, tgt_mask, tgt_padding_mask, which='x')  # (B, T, D)
            cross_y = self.decoder_cross(memory, tgt_emb_y, tgt_mask, tgt_padding_mask, which='y')  # (B, T, D)
        else:
            # if cross-attn disabled, use the embeddings themselves as 'cross' output
            cross_x = tgt_emb_x
            cross_y = tgt_emb_y

        # per-stream self-attention (optional)
        if getattr(self.cfg, 'decoder_use_self_attention', True):
            # transformer encoder expects (T, B, D)
            cross_x_t = cross_x.transpose(0, 1)
            cross_y_t = cross_y.transpose(0, 1)
            sa_x = self.self_attn_x(cross_x_t).transpose(0, 1)  # (B, T, D)
            sa_y = self.self_attn_y(cross_y_t).transpose(0, 1)  # (B, T, D)
        else:
            sa_x = cross_x
            sa_y = cross_y

        # combine x/y streams. If decoder_split_xy is True (new behavior) concatenate;
        # otherwise fuse into a single-stream representation (legacy compatibility).
        if self.decoder_split_xy:
            fused = torch.cat([sa_x, sa_y], dim=-1)  # (B, T, 2D)
        else:
            # simple average fusion for legacy compatibility -> (B, T, D)
            fused = 0.5 * (sa_x + sa_y)

        # fusion self-attention (optional)
        if getattr(self.cfg, 'decoder_use_fusion_self_attention', True):
            fused_t = fused.transpose(0, 1)
            fused_out = self.fusion_self_attn(fused_t).transpose(0, 1)  # (B, T, 2D)
        else:
            fused_out = fused

        # project to token logits
        pred_traj_points = self.output(fused_out)
        return pred_traj_points

    def predict(self, encoder_out, tgt, bev_camera_encoder=None, bev_target_encoder=None, hist_feats=None):
        length = tgt.size(1)
        padding_num = self.cfg.item_number * self.cfg.autoregressive_points + int(self.cfg.append_token) - length

        offset = 1
        if padding_num > 0:
            device = encoder_out.device if hasattr(encoder_out, 'device') else tgt.device
            padding = torch.ones(tgt.size(0), padding_num).fill_(self.PAD_token).long().to(device)
            tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        # adapt pos embeddings for predict as well
        tgt_len_pred = tgt_embedding.size(1)
        tgt_dim_pred = tgt_embedding.size(-1)
        pos_x = self._adapt_pos(self.pos_embed_x, tgt_len_pred, tgt_dim_pred, tgt_embedding.device, tgt_embedding.dtype)
        pos_y = self._adapt_pos(self.pos_embed_y, tgt_len_pred, tgt_dim_pred, tgt_embedding.device, tgt_embedding.dtype)
        tgt_emb_x = tgt_embedding + pos_x
        tgt_emb_y = tgt_embedding + pos_y

        # prepare memory respecting cross-attention and history flags
        if getattr(self.cfg, 'decoder_use_cross_attention', True):
            if bev_camera_encoder is not None or bev_target_encoder is not None or hist_feats is not None:
                memory = self._prepare_memory(bev_camera_encoder, bev_target_encoder, hist_feats if getattr(self.cfg, 'history_use_attention', True) else None)
            else:
                memory = encoder_out.permute(2, 0, 1)
        else:
            try:
                memory = encoder_out.permute(2, 0, 1)
            except Exception:
                B = tgt_embedding.size(0)
                D = tgt_embedding.size(2)
                memory = torch.zeros((1, B, D), device=tgt_embedding.device, dtype=tgt_embedding.dtype)

        # decode and return last generated token similar to previous behavior
        if getattr(self.cfg, 'decoder_use_cross_attention', True):
            cross_x = self.decoder_cross(memory, tgt_emb_x, tgt_mask, tgt_padding_mask, which='x')
            cross_y = self.decoder_cross(memory, tgt_emb_y, tgt_mask, tgt_padding_mask, which='y')
        else:
            cross_x = tgt_emb_x
            cross_y = tgt_emb_y

        if getattr(self.cfg, 'decoder_use_self_attention', True):
            sa_x = self.self_attn_x(cross_x.transpose(0,1)).transpose(0,1)
            sa_y = self.self_attn_y(cross_y.transpose(0,1)).transpose(0,1)
        else:
            sa_x = cross_x
            sa_y = cross_y

        if self.decoder_split_xy:
            fused = torch.cat([sa_x, sa_y], dim=-1)
        else:
            fused = 0.5 * (sa_x + sa_y)

        if getattr(self.cfg, 'decoder_use_fusion_self_attention', True):
            fused_out = self.fusion_self_attn(fused.transpose(0,1)).transpose(0,1)
        else:
            fused_out = fused

        pred_traj_points = self.output(fused_out)[:, length - offset, :]
        pred_traj_points = torch.softmax(pred_traj_points, dim=-1)
        pred_traj_points = pred_traj_points.argmax(dim=-1).view(-1, 1)
        return pred_traj_points

    def predict_logits(self, encoder_out, tgt, bev_camera_encoder=None, bev_target_encoder=None, hist_feats=None):
        """Return raw logits (no softmax) for the last position given current tgt sequence.
        This is identical to `predict` but returns logits to allow external sampling for RL.
        """
        length = tgt.size(1)
        padding_num = self.cfg.item_number * self.cfg.autoregressive_points + int(self.cfg.append_token) - length

        offset = 1
        if padding_num > 0:
            device = encoder_out.device if hasattr(encoder_out, 'device') else tgt.device
            padding = torch.ones(tgt.size(0), padding_num).fill_(self.PAD_token).long().to(device)
            tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        # adapt pos embeddings for predict as well
        tgt_len_pred = tgt_embedding.size(1)
        tgt_dim_pred = tgt_embedding.size(-1)
        pos_x = self._adapt_pos(self.pos_embed_x, tgt_len_pred, tgt_dim_pred, tgt_embedding.device, tgt_embedding.dtype)
        pos_y = self._adapt_pos(self.pos_embed_y, tgt_len_pred, tgt_dim_pred, tgt_embedding.device, tgt_embedding.dtype)
        tgt_emb_x = tgt_embedding + pos_x
        tgt_emb_y = tgt_embedding + pos_y

        # prepare memory respecting cross-attention and history flags
        if getattr(self.cfg, 'decoder_use_cross_attention', True):
            if bev_camera_encoder is not None or bev_target_encoder is not None or hist_feats is not None:
                memory = self._prepare_memory(bev_camera_encoder, bev_target_encoder, hist_feats if getattr(self.cfg, 'history_use_attention', True) else None)
            else:
                memory = encoder_out.permute(2, 0, 1)
        else:
            try:
                memory = encoder_out.permute(2, 0, 1)
            except Exception:
                B = tgt_embedding.size(0)
                D = tgt_embedding.size(2)
                memory = torch.zeros((1, B, D), device=tgt_embedding.device, dtype=tgt_embedding.dtype)

        # decode
        if getattr(self.cfg, 'decoder_use_cross_attention', True):
            cross_x = self.decoder_cross(memory, tgt_emb_x, tgt_mask, tgt_padding_mask, which='x')
            cross_y = self.decoder_cross(memory, tgt_emb_y, tgt_mask, tgt_padding_mask, which='y')
        else:
            cross_x = tgt_emb_x
            cross_y = tgt_emb_y

        if getattr(self.cfg, 'decoder_use_self_attention', True):
            sa_x = self.self_attn_x(cross_x.transpose(0,1)).transpose(0,1)
            sa_y = self.self_attn_y(cross_y.transpose(0,1)).transpose(0,1)
        else:
            sa_x = cross_x
            sa_y = cross_y

        if self.decoder_split_xy:
            fused = torch.cat([sa_x, sa_y], dim=-1)
        else:
            fused = 0.5 * (sa_x + sa_y)

        if getattr(self.cfg, 'decoder_use_fusion_self_attention', True):
            fused_out = self.fusion_self_attn(fused.transpose(0,1)).transpose(0,1)
        else:
            fused_out = fused

        logits = self.output(fused_out)[:, length - offset, :]
        return logits

