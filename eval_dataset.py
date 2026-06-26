"""
Dataset-based evaluation script.
This script evaluates the pretrained model on the dataset under `e2e_dataset` (train/ or val/),
without using ROS or rosbag. It computes L2 mean / Hausdorff / Fourier difference per-sample
and a summary mean, saving results to JSON files under `out_dir`.

Usage (examples):
  python eval_dataset.py --inference_config ./config/inference_real.yaml --ckpt ./ckpt/pretrained_model.ckpt \
      --data_dir ./e2e_dataset --cam_info_dir ./catkin_ws/src/core/config --out_dir ./log/eval_dataset

The script reuses model code in `model_interface/model` and dataset_interface.
"""

import os
import time
import json
import argparse
from collections import OrderedDict

import torch
import numpy as np

from utils.config import get_inference_config_obj
from dataset_interface.dataset_real import ParkingDataModuleReal
from model_interface.model.parking_model_real import ParkingModelReal
from utils.trajectory_utils import TrajectoryDistance
from utils.traj_post_process import fitting_curve
from utils.trajectory_utils import detokenize_traj_point


def load_model_from_ckpt(cfg, ckpt_path, device):
    model = ParkingModelReal(cfg.train_meta_config)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        sd = ckpt['model']
    else:
        sd = ckpt
    # normalize keys (strip prefixes used in some checkpoints)
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_k = k
        if new_k.startswith('parking_model.'):
            new_k = new_k.replace('parking_model.', '')
        if new_k.startswith('model.'):
            new_k = new_k.replace('model.', '')
        if new_k.startswith('state_dict.'):
            new_k = new_k.replace('state_dict.', '')
        new_sd[new_k] = v

    # Adapt model decoder structure if checkpoint uses different decoder fusion dim
    try:
        # look for trajectory_decoder.output.weight key in checkpoint
        output_key = None
        for k in new_sd.keys():
            if k.endswith('trajectory_decoder.output.weight') or k == 'trajectory_decoder.output.weight':
                output_key = k
                break
        if output_key is not None:
            ck_in_features = new_sd[output_key].shape[1]
            # current model expected in_features
            model_out = model.trajectory_decoder.output
            model_in_features = model_out.weight.shape[1]
            if ck_in_features != model_in_features:
                # decide whether checkpoint corresponds to single-stream legacy decoder
                tf_de_dim = getattr(cfg.train_meta_config, 'tf_de_dim', None)
                # if checkpoint in_features equals tf_de_dim, it's likely legacy (no x/y split)
                from torch import nn as _nn
                if tf_de_dim is not None and ck_in_features == int(tf_de_dim):
                    model.trajectory_decoder.decoder_split_xy = False
                else:
                    model.trajectory_decoder.decoder_split_xy = True

                fusion_dim = ck_in_features
                # rebuild fusion self-attention and output projection to match checkpoint
                fusion_layer = _nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=model.trajectory_decoder.cfg.tf_de_heads)
                model.trajectory_decoder.fusion_self_attn = _nn.TransformerEncoder(fusion_layer, num_layers=1)
                model.trajectory_decoder.output = _nn.Linear(fusion_dim, model.trajectory_decoder.cfg.token_nums + model.trajectory_decoder.cfg.append_token)
                # Note: weights will be loaded from checkpoint below (strict=False)
    except Exception:
        # non-fatal: keep existing model structure and attempt to load
        pass

    model.load_state_dict(new_sd, strict=False)
    model.to(device)
    model.eval()
    return model


def single_sample_infer(model, cfg, sample):
    """Run inference on a single dataset sample (no batch) and return predicted points (Nx2 numpy).
    sample is the output of ParkingDataModuleReal.__getitem__.
    """
    # derive device from model parameters (robust if cfg doesn't carry device)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare tensors and add batch dim
    images = sample['image'].unsqueeze(0).to(device)
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device)
    extrinsics = sample['extrinsics'].unsqueeze(0).to(device)

    data = {
        'image': images,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'fuzzy_target_point': sample['fuzzy_target_point'].unsqueeze(0).to(device),
        'target_point': sample['target_point'].unsqueeze(0).to(device),
    }

    # prepare BOS token sequence for transformer decoding
    BOS_token = cfg.train_meta_config.token_nums
    start_token = torch.tensor([[BOS_token]], dtype=torch.int64).to(device)
    data['gt_traj_point_token'] = start_token

    # run model inference (choose decoder)
    with torch.no_grad():
        if cfg.train_meta_config.decoder_method == 'transformer':
            predict_ret = model.predict_transformer(data, predict_token_num=cfg.train_meta_config.item_number * cfg.train_meta_config.autoregressive_points)
            # predict_transformer may return (tokens, pred_segmentation, pred_depth, bev_target)
            if isinstance(predict_ret, tuple) or isinstance(predict_ret, list):
                pred_token_seq = predict_ret[0]
            else:
                pred_token_seq = predict_ret
            # pred_token_seq shape: (1, seq_len)
            pred_token_update = pred_token_seq[0][1:]  # remove initial bos
            # remove invalid tokens if any (only available on inference wrapper)
            if hasattr(model, 'remove_invalid_content'):
                pred_token_update = model.remove_invalid_content(pred_token_update)
            else:
                # local fallback: find EOS token and truncate (same logic as inference_real)
                EOS_token = cfg.train_meta_config.token_nums + cfg.train_meta_config.append_token - 2
                eos_idx = torch.where(pred_token_update == EOS_token)[0]
                if len(eos_idx) > 0:
                    finish_index = eos_idx[0].item()
                    finish_index = finish_index - finish_index % cfg.train_meta_config.item_number
                    if finish_index > 0:
                        pred_token_update = pred_token_update[:finish_index]
            pred_tokens = pred_token_update.cpu()
            pred_pts = detokenize_traj_point(pred_tokens, cfg.train_meta_config.token_nums, cfg.train_meta_config.item_number, cfg.train_meta_config.xy_max)
            pred_pts = pred_pts.numpy()
        elif cfg.train_meta_config.decoder_method == 'gru':
            predict_ret = model.predict_gru(data)
            # predict_gru may return (pred_pts, pred_seg, pred_depth, bev_target) in our wrapper
            if isinstance(predict_ret, tuple) or isinstance(predict_ret, list):
                pred_pts = predict_ret[0]
            else:
                pred_pts = predict_ret
            if isinstance(pred_pts, torch.Tensor):
                pred_pts = pred_pts.cpu().numpy()
        else:
            raise ValueError('Unsupported decoder method')

    # post-process - fitting curve like inference
    pred_pts = fitting_curve(pred_pts, num_points=cfg.train_meta_config.autoregressive_points, item_number=cfg.train_meta_config.item_number)
    # ensure Nx2
    pred_pts = np.array(pred_pts)
    if pred_pts.ndim == 1:
        pred_pts = pred_pts.reshape(-1, cfg.train_meta_config.item_number)
    if pred_pts.shape[1] > 2:
        pred_pts = pred_pts[:, :2]
    return pred_pts


def evaluate_dataset(inference_cfg, ckpt_path, data_dir, cam_info_dir, out_dir, is_train=0):
    # set up output
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load model
    model = load_model_from_ckpt(inference_cfg, ckpt_path, device)

    # prepare dataset (validation by default)
    train_meta_cfg = inference_cfg.train_meta_config
    train_meta_cfg.data_dir = data_dir
    train_meta_cfg.training_dir = 'train'
    train_meta_cfg.validation_dir = 'val'
    dataset = ParkingDataModuleReal(train_meta_cfg, is_train)

    # evaluate over dataset

    metrics_list = []
    event_id = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        # ground truth points (dataset stores flattened list of length autoregressive_points*item_number)
        gt_flat = sample['gt_traj_point'].numpy()
        gt_pts = gt_flat.reshape(-1, train_meta_cfg.item_number)
        # predicted points
        pred_pts = single_sample_infer(model, inference_cfg, sample)

        # align lengths
        min_len = min(pred_pts.shape[0], gt_pts.shape[0])
        if min_len == 0:
            continue
        pred_al = pred_pts[:min_len]
        gt_al = gt_pts[:min_len]

        td = TrajectoryDistance(pred_al, gt_al)
        # compute metrics safely; trajectory_utils now returns zeros or NaN for empty cases
        l2 = td.get_l2_distance()
        try:
            haus = td.get_haus_distance()
        except Exception:
            haus = float('nan')
        try:
            fourier = td.get_fourier_difference()
        except Exception:
            fourier = float('nan')
        metrics = {
            'l2_mean': float(l2) if not (l2 is None) else float('nan'),
            'hausdorff': float(haus) if not (haus is None) else float('nan'),
            'fourier': float(fourier) if not (fourier is None) else float('nan')
        }
        metrics_list.append(metrics)

        event = {
            'event_id': str(event_id).zfill(5),
            'index': idx,
            'metrics': metrics
        }
        with open(os.path.join(out_dir, f"event_{event['event_id']}.json"), 'w') as f:
            json.dump(event, f, indent=2)
        event_id += 1

    # summary
    if len(metrics_list) == 0:
        summary = {'total': 0, 'valid': 0, 'metrics_mean': None}
    else:
        l2s = np.array([m['l2_mean'] for m in metrics_list], dtype=float)
        hauss = np.array([m['hausdorff'] for m in metrics_list], dtype=float)
        fds = np.array([m['fourier'] for m in metrics_list], dtype=float)
        # use nanmean to ignore NaNs resulting from empty/invalid samples
        summary = {
            'total': len(metrics_list),
            'valid': int(np.sum(~np.isnan(l2s))),
            'metrics_mean': {
                'l2_mean': float(np.nanmean(l2s)) if np.any(~np.isnan(l2s)) else None,
                'hausdorff': float(np.nanmean(hauss)) if np.any(~np.isnan(hauss)) else None,
                'fourier': float(np.nanmean(fds)) if np.any(~np.isnan(fds)) else None
            }
        }

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('Saved summary to', os.path.join(out_dir, 'summary.json'))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_config', default='./config/inference_real.yaml')
    parser.add_argument('--ckpt', default='./ckpt/pretrained_model.ckpt')
    parser.add_argument('--data_dir', default='./e2e_dataset')
    parser.add_argument('--cam_info_dir', default='./catkin_ws/src/core/config')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    inference_cfg = get_inference_config_obj(args.inference_config)
    if args.out_dir is None:
        args.out_dir = os.path.join('log', 'eval_dataset', str(int(time.time())))

    # override cam info path
    inference_cfg.cam_info_dir = args.cam_info_dir

    evaluate_dataset(inference_cfg, args.ckpt, args.data_dir, args.cam_info_dir, args.out_dir, is_train=0)
