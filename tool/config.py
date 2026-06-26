import os
import torch

from datetime import datetime


class Configuration:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = None
    log_dir = None
    checkpoint_dir = None
    log_every_n_steps = None
    check_val_every_n_epoch = None

    epochs = None
    learning_rate = None
    weight_decay = None
    batch_size = None

    training_map = None
    validation_map = None
    future_frame_nums = None
    hist_frame_nums = None
    token_nums = None
    image_crop = None

    bev_encoder_in_channel = None
    bev_encoder_out_channel = None

    bev_x_bound = None
    bev_y_bound = None
    bev_z_bound = None
    d_bound = None
    final_dim = None
    bev_down_sample = None
    use_depth_distribution = None
    backbone = None

    seg_classes = None
    seg_vehicle_weights = None

    tf_en_dim = None
    tf_en_heads = None
    tf_en_layers = None
    tf_en_dropout = None
    tf_en_bev_length = None
    tf_en_motion_length = None

    tf_de_dim = None
    tf_de_heads = None
    tf_de_layers = None
    tf_de_dropout = None
    tf_de_tgt_dim = None

    # trajectory specific
    autoregressive_points = None
    traj_downsample_stride = None
    item_number = None
    append_token = None
    xy_max = None

    # target noise / BEV trajectory concat
    add_noise_to_target = None
    target_noise_threshold = None
    target_range = None
    use_fuzzy_target = None
    concat_traj_to_bev = None
    traj_motion_mode = None

    # vehicle controller params
    lon_kp = None
    lon_ki = None
    lon_kd = None
    wheel_base = None
    lookahead_gain = None
    min_lookahead = None
    max_steer_angle = None
    max_accel = None


def get_cfg(cfg_yaml: dict):
    today = datetime.now()
    today_str = "{}_{}_{}_{}_{}_{}".format(today.year, today.month, today.day,
                                           today.hour, today.minute, today.second)
    exp_name = "exp_{}".format(today_str)

    # Accept either a dict with top-level keys or a wrapper {'parking_model': {...}}
    config = cfg_yaml.get('parking_model', cfg_yaml) if isinstance(cfg_yaml, dict) else cfg_yaml
    cfg = Configuration()

    # support both log_dir and log_root_dir naming
    log_root = config.get('log_dir', config.get('log_root_dir', './log'))
    ckpt_root = config.get('checkpoint_dir', config.get('checkpoint_root_dir', './ckpt'))
    cfg.log_dir = os.path.join(log_root, exp_name)
    cfg.checkpoint_dir = os.path.join(ckpt_root, exp_name)

    cfg.log_every_n_steps = config.get('log_every_n_steps', 10)
    cfg.check_val_every_n_epoch = config.get('check_val_every_n_epoch', 1)

    cfg.epochs = config.get('epochs', 100)
    cfg.learning_rate = config.get('learning_rate', 1e-4)
    cfg.weight_decay = config.get('weight_decay', 0.0)
    cfg.batch_size = config.get('batch_size', 8)

    # training/validation map naming: support training_map or training_dir
    cfg.training_map = config.get('training_map', config.get('training_dir', None))
    cfg.validation_map = config.get('validation_map', config.get('validation_dir', None))

    cfg.future_frame_nums = config.get('future_frame_nums', None)
    cfg.hist_frame_nums = config.get('hist_frame_nums', None)
    cfg.token_nums = config.get('token_nums', None)
    # image_crop: if not provided, fall back to first element of process_dim (if present) or 200
    proc_dim = config.get('process_dim', None)
    if config.get('image_crop', None) is not None:
        cfg.image_crop = config.get('image_crop')
    elif proc_dim and isinstance(proc_dim, (list, tuple)) and len(proc_dim) >= 1:
        try:
            cfg.image_crop = int(proc_dim[0])
        except Exception:
            cfg.image_crop = 200
    else:
        cfg.image_crop = 200

    cfg.bev_encoder_in_channel = config.get('bev_encoder_in_channel', None)
    cfg.bev_encoder_out_channel = config.get('bev_encoder_out_channel', None)

    cfg.bev_x_bound = config.get('bev_x_bound', None)
    cfg.bev_y_bound = config.get('bev_y_bound', None)
    cfg.bev_z_bound = config.get('bev_z_bound', None)
    cfg.d_bound = config.get('d_bound', None)
    cfg.final_dim = config.get('final_dim', None)
    cfg.bev_down_sample = config.get('bev_down_sample', None)
    cfg.use_depth_distribution = bool(config.get('use_depth_distribution', False))
    cfg.backbone = config.get('backbone', None)

    cfg.seg_classes = config.get('seg_classes', None)
    cfg.seg_vehicle_weights = config.get('seg_vehicle_weights', None)

    # support both tf_en_* and query_en_* naming used in some configs
    cfg.tf_en_dim = config.get('tf_en_dim', config.get('query_en_dim', None))
    cfg.tf_en_heads = config.get('tf_en_heads', config.get('query_en_heads', None))
    cfg.tf_en_layers = config.get('tf_en_layers', config.get('query_en_layers', None))
    cfg.tf_en_dropout = config.get('tf_en_dropout', config.get('query_en_dropout', None))
    cfg.tf_en_bev_length = config.get('tf_en_bev_length', config.get('query_en_bev_length', None))
    cfg.tf_en_motion_length = config.get('tf_en_motion_length', config.get('query_en_motion_length', None))

    cfg.tf_de_dim = config.get('tf_de_dim', None)
    cfg.tf_de_heads = config.get('tf_de_heads', None)
    cfg.tf_de_layers = config.get('tf_de_layers', None)
    cfg.tf_de_dropout = config.get('tf_de_dropout', None)
    cfg.tf_de_tgt_dim = config.get('tf_de_tgt_dim', None)

    # trajectory specific
    cfg.autoregressive_points = config.get('autoregressive_points', None)
    cfg.traj_downsample_stride = config.get('traj_downsample_stride', 1)
    cfg.item_number = config.get('item_number', 2)
    cfg.append_token = config.get('append_token', 0)
    cfg.xy_max = config.get('xy_max', None)

    # target noise
    cfg.add_noise_to_target = bool(config.get('add_noise_to_target', False))
    cfg.target_noise_threshold = float(config.get('target_noise_threshold', 0.0))
    cfg.target_range = float(config.get('target_range', 0.0))
    cfg.use_fuzzy_target = bool(config.get('use_fuzzy_target', False))

    # trajectory concat options
    cfg.concat_traj_to_bev = bool(config.get('concat_traj_to_bev', False))
    cfg.traj_motion_mode = config.get('traj_motion_mode', 'first')

    # vehicle controller defaults
    cfg.lon_kp = float(config.get('lon_kp', 1.0))
    cfg.lon_ki = float(config.get('lon_ki', 0.0))
    cfg.lon_kd = float(config.get('lon_kd', 0.0))
    cfg.wheel_base = float(config.get('wheel_base', 2.7))
    cfg.lookahead_gain = float(config.get('lookahead_gain', 1.0))
    cfg.min_lookahead = float(config.get('min_lookahead', 2.0))
    cfg.max_steer_angle = float(config.get('max_steer_angle', 0.7854))
    cfg.max_accel = float(config.get('max_accel', 3.0))

    # data_dir support (some configs call it data_dir, others use data_root or similar)
    cfg.data_dir = config.get('data_dir', config.get('data_root', None))

    return cfg
