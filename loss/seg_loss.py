import torch
from torch import nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights=None):
        """class_weights may be:
           - None: no weighting
           - list or numpy array: converted to torch.Tensor
           - torch.Tensor: used directly
        The actual device/dtype conversion is done in forward when the target device is known.
        """
        super(SegmentationLoss, self).__init__()
        self.ignore_index = 255
        # store raw class_weights; conversion to tensor happens in forward where device is known
        self.class_weights = class_weights

    def forward(self, pred, target):
        # Ensure no NaN or Inf in predictions
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e5, neginf=-1e5)
        # normalize pred to 5D (b, s, c, h, w)
        if pred.dim() == 4:
            b, c, h, w = pred.shape
            s = 1
            pred = pred.unsqueeze(1)  # (b,1,c,h,w)
        elif pred.dim() == 5:
            b, s, c, h, w = pred.shape
        else:
            raise ValueError(f'Unsupported pred tensor dims: {pred.dim()}, expected 4 or 5.')

        # normalize target to (b, s, 1, h, w) or (b, 1, h, w) or (b, h, w)
        if target.dim() == 5:
            # (b, s, 1, h, w)
            if target.shape[2] != 1:
                raise ValueError('segmentation label must be index label with channel dim = 1')
            gt = target
        elif target.dim() == 4:
            # could be (b, 1, h, w) or (b, h, w, ?unlikely)
            if target.shape[1] == 1:
                # (b,1,h,w) -> expand to (b,1,1,h,w)
                gt = target.unsqueeze(1)  # (b,1,1,h,w)
            else:
                # assume (b,h,w) incorrectly shaped; try to handle
                gt = target.unsqueeze(1).unsqueeze(1)  # (b,1,1,h,w)
        elif target.dim() == 3:
            # (b, h, w) -> (b,1,1,h,w)
            gt = target.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f'Unsupported target tensor dims: {target.dim()}, expected 3,4 or 5.')

        # Now gt should be (b, s_or_1, 1, h, w)
        # If gt has s==1 but pred has s>1, try to broadcast gt along s
        gt_b, gt_s, gt_c, gt_h, gt_w = gt.shape
        if gt_c != 1:
            raise ValueError('segmentation label must be index label with channel dim = 1')

        if (gt_b != b) or (gt_h != h) or (gt_w != w):
            raise ValueError('Mismatch between pred and target spatial/batch dimensions')

        if gt_s == 1 and s > 1:
            gt = gt.repeat(1, s, 1, 1, 1)
            gt_s = s

        if gt_s != s:
            raise ValueError('Mismatch between pred sequence length and target sequence length')

        # reshape for cross entropy: pred -> (b*s, c, h, w), gt -> (b*s, h, w)
        pred_seg = pred.view(b * s, c, h, w)
        gt_seg = gt.view(b * s, h, w)

        # prepare weight tensor if provided
        weight = None
        if self.class_weights is not None:
            # accept python list / numpy array / torch tensor
            if not isinstance(self.class_weights, torch.Tensor):
                try:
                    weight = torch.tensor(self.class_weights, dtype=torch.float32, device=gt_seg.device)
                except Exception:
                    # fallback: try converting via torch.as_tensor
                    weight = torch.as_tensor(self.class_weights, dtype=torch.float32, device=gt_seg.device)
            else:
                weight = self.class_weights.to(dtype=torch.float32, device=gt_seg.device)

            # sanity: number of classes in weight should match pred channels
            if weight.numel() != pred_seg.size(1):
                # If mismatch, do not use weight (safer) and log via stderr
                print(f"[SegmentationLoss] class_weights length ({weight.numel()}) != num_classes ({pred_seg.size(1)}); ignoring weights.")
                weight = None

        seg_loss = F.cross_entropy(pred_seg,
                                   gt_seg,
                                   reduction='none',
                                   ignore_index=self.ignore_index,
                                   weight=weight)

        return torch.mean(seg_loss)
