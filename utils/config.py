import datetime
import os
from dataclasses import dataclass
from typing import List

import torch
import yaml
from loguru import logger


@dataclass
class Configuration:
    data_mode: str
    num_gpus: int
    cuda_device_index: str
    data_dir: str
    log_root_dir: str
    checkpoint_root_dir: str
    log_every_n_steps: int
    check_val_every_n_epoch: int

    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_workers: int

    training_dir: str
    validation_dir: str
    autoregressive_points: int
    item_number: int
    token_nums: int
    xy_max: float
    process_dim: List[int]

    use_fuzzy_target: bool
    bev_encoder_in_channel: int

    bev_x_bound: List[float]
    bev_y_bound: List[float]
    bev_z_bound: List[float]
    d_bound: List[float]
    final_dim: List[int]
    bev_down_sample: int
    backbone: str

    tf_de_dim: int
    tf_de_heads: int
    tf_de_layers: int
    tf_de_dropout: float

    append_token: int
    traj_downsample_stride: int

    add_noise_to_target: bool
    target_noise_threshold: float

    fusion_method: str
    decoder_method: str
    query_en_dim: int
    query_en_heads: int
    query_en_layers: int
    query_en_dropout: float
    query_en_bev_length: int
    target_range: float

    # new: segmentation and depth usage flags and loss weights
    use_segmentation: bool = False
    seg_loss_weight: float = 1.0
    use_depth: bool = False
    depth_loss_weight: float = 1.0
    # segmentation specifics
    seg_classes: int = 3
    seg_vehicle_weights: object = None

    # New target & pruning configuration defaults
    target_dist: str = 'gaussian'
    target_sigma_pixels: float = 2.0

    use_query_pruner: bool = False
    query_prune_soft: bool = True
    query_prune_topk: int = None
    query_prune_ratio: float = 0.1
    query_prune_temperature: float = 1.0

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume_path: str = None
    # optional pretrained weights (weights-only) path — can be set in training yaml
    pretrained_ckpt: str = None
    config_path: str = None
    log_dir: str = None
    checkpoint_dir: str = None
    use_depth_distribution: bool = False
    tf_en_motion_length: str = None

    # history encoding options
    use_history: bool = False
    history_len: int = 10
    history_hidden_dim: int = 64
    history_out_dim: int = None  # if None, default to tf_de_dim at runtime
    # history attention control: when True, history features are appended to decoder memory for cross-attention
    history_use_attention: bool = True

    # Decoder attention ablation flags
    # If False, the decoder will skip cross-attention between queries and encoder memory
    decoder_use_cross_attention: bool = True
    # If False, the per-stream self-attention (x/y) is disabled
    decoder_use_self_attention: bool = True
    # If False, the final fusion self-attention after concatenation of x/y features is disabled
    decoder_use_fusion_self_attention: bool = True

    # Kinematics / smoothness losses
    use_smoothness_loss: bool = False
    smoothness_loss_weight: float = 0.1
    use_curvature_loss: bool = False
    curvature_loss_weight: float = 0.01
    # If true, learnable task weights (Kendall) will be used instead of fixed weights
    use_learnable_loss_weights: bool = False

    # Reinforcement Learning fine-tuning options (if True, use RL fine-tune after BC pretraining)
    use_rl_finetune: bool = False
    # RL algorithm: currently supported: 'pg' (simple policy gradient). Placeholder for 'ppo' or others.
    rl_algorithm: str = 'pg'
    rl_env: str = 'carla'  # name of environment; used by RL scripts to select env wrapper
    rl_timesteps: int = 10000
    rl_epochs: int = 10
    rl_rollout_length: int = 200
    rl_batch_size: int = 32
    rl_lr: float = 1e-5
    rl_gamma: float = 0.99
    # whether to only train decoder (keep encoder frozen) during RL fine-tune
    rl_train_decoder_only: bool = True
    # path to save RL fine-tuned checkpoint
    rl_save_path: str = './ckpt/rl_finetuned.ckpt'

    # Offline RL options (dataset-only fine-tune during training)
    use_offline_rl: bool = False
    # weight for offline RL loss added to BC loss
    offline_rl_weight: float = 0.1
    # reward type for offline RL: currently supports 'l2'
    offline_rl_reward: str = 'l2'
    # optional reward scale applied before advantage normalization
    offline_rl_reward_scale: float = 1.0
    # whether to normalize per-batch advantage
    offline_rl_adv_normalize: bool = True


@dataclass
class InferenceConfiguration:
    model_ckpt_path: str
    training_config: str
    predict_mode: str

    trajectory_pub_frequency: int
    cam_info_dir: str
    progress_threshold: float

    train_meta_config: Configuration = None

def get_train_config_obj(config_path: str):
    exp_name = get_exp_name()
    with open(config_path, 'r') as yaml_file:
        try:
            config_yaml = yaml.safe_load(yaml_file)
            # Backwards/alternate key compatibility:
            # Some configs use 'training_map'/'validation_map' while the dataclass expects
            # 'training_dir'/'validation_dir'. Normalize both directions so downstream
            # code (and older configs) work.
            if 'training_map' in config_yaml and 'training_dir' not in config_yaml:
                config_yaml['training_dir'] = config_yaml['training_map']
            if 'validation_map' in config_yaml and 'validation_dir' not in config_yaml:
                config_yaml['validation_dir'] = config_yaml['validation_map']
            # Also provide the reverse keys for modules that still reference 'training_map'/'validation_map'
            if 'training_dir' in config_yaml and 'training_map' not in config_yaml:
                config_yaml['training_map'] = config_yaml['training_dir']
            if 'validation_dir' in config_yaml and 'validation_map' not in config_yaml:
                config_yaml['validation_map'] = config_yaml['validation_dir']

            # Filter YAML keys to only those accepted by Configuration to avoid
            # TypeError when users include legacy/extra keys.
            accepted_keys = set(Configuration.__dataclass_fields__.keys())
            filtered_cfg = {k: v for k, v in config_yaml.items() if k in accepted_keys}

            config_obj = Configuration(**filtered_cfg)
            # Keep backward-compatible attributes on the returned object
            # so other modules that expect training_map/validation_map still work.
            config_obj.training_map = config_yaml.get('training_map', config_obj.training_dir)
            config_obj.validation_map = config_yaml.get('validation_map', config_obj.validation_dir)
            config_obj.config_path = config_path
            config_obj.log_dir = os.path.join(config_obj.log_root_dir, exp_name)
            config_obj.checkpoint_dir = os.path.join(config_obj.checkpoint_root_dir, exp_name)
            # Backfill histogram/future frame settings if missing. Older codebases used
            # hist_frame_nums=10 by default; future_frame_nums should at least cover
            # required autoregressive prediction horizon.
            # Provide sensible defaults to keep backwards compatibility.
            config_obj.hist_frame_nums = config_yaml.get('hist_frame_nums', 10)

            if 'future_frame_nums' in config_yaml:
                config_obj.future_frame_nums = config_yaml['future_frame_nums']
            else:
                # derive required future frames from autoregressive settings when possible
                try:
                    afr = getattr(config_obj, 'autoregressive_points', None)
                    stride = getattr(config_obj, 'traj_downsample_stride', 1)
                    if afr is not None:
                        config_obj.future_frame_nums = int(1 + (int(afr) - 1) * int(stride))
                    else:
                        config_obj.future_frame_nums = 30
                except Exception:
                    config_obj.future_frame_nums = 30
        except yaml.YAMLError:
            logger.exception("Open {} failed!", config_path)
    return config_obj


def get_exp_name():
    today = datetime.datetime.now()
    today_str = "{}_{}_{}_{}_{}_{}".format(today.year, today.month, today.day,
                                           today.hour, today.minute, today.second)
    exp_name = "exp_{}".format(today_str)
    return exp_name


def get_inference_config_obj(config_path: str):
    with open(config_path, 'r') as yaml_file:
        try:
            config_yaml = yaml.safe_load(yaml_file)
            inference_config_obj = InferenceConfiguration(**config_yaml)
        except yaml.YAMLError:
            logger.exception("Open {} failed!", config_path)
    training_config_path = os.path.join(os.path.dirname(config_path), "{}.yaml".format(inference_config_obj.training_config))
    inference_config_obj.train_meta_config = get_train_config_obj(training_config_path)
    return inference_config_obj