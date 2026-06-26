import sys
sys.path.insert(0,'/-your path-/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
import carla
import math
import pathlib
import yaml
import torch
import logging
import time
import pygame

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from collections import OrderedDict

from tool.geometry import update_intrinsics
from tool.config import Configuration, get_cfg
from dataset.carla_dataset import ProcessImage, convert_slot_coord, ProcessSemantic
from dataset.carla_dataset import detokenize
from data_generation.network_evaluator import NetworkEvaluator
from data_generation.tools import encode_npy_to_pil
# from model.parking_model import ParkingModel
from tool.pid_controller import VehicleController


from model_interface.model.parking_model_real import ParkingModelReal
from utils.config import get_train_config_obj
from utils.trajectory_utils import detokenize_traj_point
from utils.traj_post_process import fitting_curve


def show_control_info(window, control, steering_wheel_image, width, height, font):
    histogram_width = 15

    t_x = width - 30
    t_y = height - 50

    b_x = t_x - 30
    b_y = t_y

    s_x = t_x - 80
    s_y = t_y - 40

    r_x = t_x - 140
    r_y = t_y

    # throttle max = 0.5 in data gen
    throttle_height = (control['throttle'] * 200) * 0.8
    throttle_rect = pygame.Rect(t_x, t_y - throttle_height, histogram_width, throttle_height)
    pygame.draw.rect(window, (0, 255, 0), throttle_rect)

    brake_height = (control['brake'] * 100) * 0.8
    brake_rect = pygame.Rect(b_x, b_y - brake_height, histogram_width, brake_height)
    pygame.draw.rect(window, (255, 0, 0), brake_rect)

    steer = -control['steer'] * 90
    rotated_steering_wheel = pygame.transform.rotate(steering_wheel_image, steer)
    rotated_rect = rotated_steering_wheel.get_rect(center=(s_x, s_y))
    window.blit(rotated_steering_wheel, rotated_rect)

    reverse = bool(control['reverse'])
    rect = pygame.Rect((r_x, r_y - 10), (10, 10))
    pygame.draw.rect(window, (0, 0, 0), rect, 0 if reverse else 1)

    # show text
    throttle_text = font.render("T", True, (0, 255, 0))
    brake_text = font.render("B", True, (255, 0, 0))
    steer_text = font.render("S", True, (0, 0, 0))
    reverse_text = font.render("R", True, (0, 0, 0))

    window.blit(throttle_text, (t_x + 2, t_y + 10))
    window.blit(brake_text, (b_x + 2, b_y + 10))
    window.blit(steer_text, (s_x - 4, s_y + 50))
    window.blit(reverse_text, (r_x, r_y + 10))


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


def grid_show(to_shows, cols):
    it = iter(to_shows)
    fig, axs = plt.subplots(1, cols, figsize=(cols * 2, cols))
    for j in range(cols):
        try:
            image, title = next(it)
        except StopIteration:
            image = np.zeros_like(to_shows[0][0])
            title = 'pad'
        axs[j].imshow(image)
        axs[j].set_title(title)
        axs[j].set_yticks([])
        axs[j].set_xticks([])
    plt.show()


def visualize_heads(att_map):
    to_shows = []
    att_map = att_map.squeeze()
    cols = att_map.shape[0] + 1
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image


def get_atten_avg_map(att_map, grid_index, image, grid_size=16):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_image = highlight_grid(image, [grid_index], grid_size)

    att_map = att_map.squeeze()
    average_att_map = att_map.mean(axis=0)
    atten_avg = average_att_map[grid_index].reshape(grid_size[0], grid_size[1])
    atten_avg = Image.fromarray(atten_avg.numpy()).resize(image.size)
    return grid_image, atten_avg


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=16, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_image = highlight_grid(image, [grid_index], grid_size)

    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        mask = att_map[i][grid_index].reshape(grid_size[0], grid_size[1])
        mask = Image.fromarray(mask.numpy()).resize(image.size)
        to_shows.append((mask, f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    average_mask = average_att_map[grid_index].reshape(grid_size[0], grid_size[1])
    average_mask = Image.fromarray(average_mask.numpy()).resize(image.size)
    to_shows.append((average_mask, 'Head Average'))

    plt.subplots_adjust(wspace=0.08, hspace=0.08, left=0.04, bottom=0.0, right=0.95, top=0.97)
    rows = 1
    cols = 7

    it = iter(to_shows)
    for j in range(cols):
        try:
            mask, title = next(it)
        except StopIteration:
            mask = np.zeros_like(to_shows[0][0])
            title = 'pad'
        ax_attem = plt.subplot(rows, cols, j + 1)
        ax_attem.axis('off')
        ax_attem.set_title(title, fontsize=10)
        ax_attem.imshow(grid_image)
        ax_attem.imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')

    plt.pause(0.1)
    plt.clf()


class ParkingAgent:
    def __init__(self, network_evaluator: NetworkEvaluator, args):

        self.show_eva_imgs = args.show_eva_imgs

        self.args = args

        self.atten_avg = None
        self.grid_image = None

        self.rgb_rear = None
        self.rgb_right = None
        self.rgb_left = None
        self.rgb_front = None
        self.seg_bev = None
        self.target_bev = None

        self.pre_target_point = None

        self.model = None
        self.device = None

        self.cfg = Configuration()
        self.load_cfg(args)

        self.log_path = pathlib.Path(self.cfg.log_dir)
        if not self.log_path.exists():
            self.log_path.mkdir()

        self.BOS_token = self.cfg.token_nums - 3

        self.hist_frame_nums = self.cfg.hist_frame_nums

        self.net_eva = network_evaluator
        self.world = network_evaluator.world
        self.player = network_evaluator.world.player

        self.is_init = False
        self.intrinsic_crop = None
        self.extrinsic = None
        self.image_process = None
        self.semantic_process = ProcessSemantic(self.cfg)

        self.process_frequency = 3  # process sensor data for every 3 steps 0.1s
        self.step = -1

        self.prev_xy_thea = None

        self.trans_control = carla.VehicleControl()
        self.gru_control = carla.VehicleControl()

        self.save_output = SaveOutput()
        self.hook_handle = None
        self.load_model(args.model_path)

        # vehicle lateral/longitudinal controller used during evaluation
        self.vehicle_controller = VehicleController(self.cfg)

        # storage for last predicted trajectory point (to inject into BEV on next step)
        self.prev_pred_traj_point = None

        self.stop_count = 0
        self.boost = False
        self.boot_step = 0

        self.init_agent()

        plt.ion()

    def load_cfg(self, args):

        with open(args.model_config_path, 'r') as config_file:
            try:
                cfg_yaml = (yaml.safe_load(config_file))
            except yaml.YAMLError:
                logging.exception('Invalid YAML Config file {}', args.config)
        self.cfg = get_cfg(cfg_yaml)

    def load_model(self, parking_pth_path):
        """Try to load new `ParkingModelReal` first; if it fails, fall back to legacy `ParkingModel`.
        For ParkingModelReal we attach a `predict(data)` wrapper that returns
        (pred_traj(B,T,2), pred_segmentation_or_None, pred_depth_or_None, target_bev_or_None).
        """
        # prepare device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # attempt to load ParkingModelReal using training config file specified by args
        try:
            # try load train config compatible with ParkingModelReal
            train_cfg = get_train_config_obj(self.args.model_config_path)
            # instantiate model
            model_real = ParkingModelReal(train_cfg)

            # load checkpoint robustly
            ckpt = torch.load(parking_pth_path, map_location=self.device)
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    sd = ckpt['state_dict']
                elif 'model_state_dict' in ckpt:
                    sd = ckpt['model_state_dict']
                elif 'model' in ckpt and isinstance(ckpt['model'], dict):
                    sd = ckpt['model']
                else:
                    sd = ckpt
            else:
                sd = ckpt

            # normalize keys
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

            model_real.load_state_dict(new_sd, strict=False)
            model_real.to(self.device)
            model_real.eval()

            # create a predict wrapper to match legacy agent expectation
            def predict_wrapper(data):
                # ensure tensors are on device
                for k in ['image', 'intrinsics', 'extrinsics', 'target_point', 'fuzzy_target_point']:
                    if k in data and isinstance(data[k], torch.Tensor):
                        data[k] = data[k].to(self.device)

                # create BOS token if missing
                BOS_token = train_cfg.token_nums
                if 'gt_traj_point_token' not in data:
                    data['gt_traj_point_token'] = torch.tensor([[BOS_token]], dtype=torch.int64).to(self.device)

                # choose decoder
                if train_cfg.decoder_method == 'transformer':
                    pred_token_seq, pred_segmentation, pred_depth, bev_target = model_real.predict_transformer(
                        data, predict_token_num=train_cfg.item_number * train_cfg.autoregressive_points)
                    # pred_token_seq: tensor (1, seq_len)
                    pred_token_update = pred_token_seq[0][1:]
                    # remove EOS if present
                    EOS_token = train_cfg.token_nums + train_cfg.append_token - 2
                    eos_idx = torch.where(pred_token_update == EOS_token)[0]
                    if len(eos_idx) > 0:
                        finish_index = eos_idx[0].item()
                        finish_index = finish_index - finish_index % train_cfg.item_number
                        if finish_index > 0:
                            pred_token_update = pred_token_update[:finish_index]

                    pred_tokens = pred_token_update.cpu()
                    pred_pts = detokenize_traj_point(pred_tokens, train_cfg.token_nums,
                                                     train_cfg.item_number, train_cfg.xy_max)
                    pred_pts = fitting_curve(pred_pts.numpy(), num_points=train_cfg.autoregressive_points,
                                             item_number=train_cfg.item_number)
                    pred_pts = np.array(pred_pts)
                    if pred_pts.ndim == 1:
                        pred_pts = pred_pts.reshape(-1, train_cfg.item_number)
                    if pred_pts.shape[1] > 2:
                        pred_pts = pred_pts[:, :2]

                    pred_traj = torch.tensor(pred_pts, dtype=torch.float).unsqueeze(0).to(self.device)  # (1,T,2)
                    # use segmentation returned by model if present
                    pred_seg = pred_segmentation if 'pred_segmentation' in locals() else None
                    return pred_traj, pred_seg, pred_depth, bev_target

                elif train_cfg.decoder_method == 'gru':
                    pred_pts = model_real.predict_gru(data)
                    if isinstance(pred_pts, torch.Tensor):
                        pred_traj = pred_pts.unsqueeze(0).to(self.device)
                    else:
                        pred_traj = torch.tensor(pred_pts, dtype=torch.float).unsqueeze(0).to(self.device)
                    pred_seg = None
                    pred_depth = None
                    bev_target = None
                    return pred_traj, pred_seg, pred_depth, bev_target

                else:
                    raise ValueError(f"Unsupported decoder {train_cfg.decoder_method}")

            # attach wrapper and finish
            model_real.predict = predict_wrapper
            self.model = model_real

            logging.info('Loaded ParkingModelReal from %s', parking_pth_path)
            return

        except Exception as e:
            logging.warning('Load ParkingModelReal failed (%s), fallback to legacy ParkingModel', e)

     def save_seg_img(self, pred_segmentation):
        if pred_segmentation is None:
            self.seg_bev = None
            return
        pred_segmentation = pred_segmentation[0]
        pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=True)
        pred_segmentation = pred_segmentation.detach().cpu().numpy()
        pred_segmentation[pred_segmentation == 1] = 128
        pred_segmentation[pred_segmentation == 2] = 255
        pred_seg_img = pred_segmentation[0, :, :][::-1]
        # image_file = pathlib.Path(self.cfg.log_dir) / ('%04d.png' % self.step)
        # Image.fromarray(np.uint8(pred_seg_img), mode='L').save(image_file)
        self.seg_bev = pred_seg_img

    def save_target_bev_img(self, target_bev):
        if target_bev is None:
            self.target_bev = None
            return
        try:
            target_bev = target_bev[0]
            target_bev = target_bev.detach().cpu().numpy()
            target_bev[target_bev == 1] = 255
            target_bev_img = target_bev[0, :, :][::-1]
            self.target_bev = target_bev_img
        except Exception:
            self.target_bev = None

    def save_prev_target(self, pred_segmentation):
        pred_segmentation = pred_segmentation[0]
        pred_segmentation = torch.argmax(pred_segmentation, dim=0, keepdim=True)
        pred_segmentation = pred_segmentation.detach().cpu().numpy()
        pred_segmentation[pred_segmentation == 1] = 128
        pred_segmentation[pred_segmentation == 2] = 255
        pred_seg_img = pred_segmentation[0, :, :][::-1]

        h, w = pred_seg_img.shape
        target_slot_x = []
        target_slot_y = []
        for row_idx in range(h):
            for col_idx in range(w):
                if pred_seg_img[row_idx, col_idx] == 255:
                    target_slot_x.append(row_idx)
                    target_slot_y.append(col_idx)

        # target point in bev
        if (len(target_slot_x) > 0) and (len(target_slot_y) > 0):
            new_target_x = int(np.average(target_slot_x))
            new_target_y = int(np.average(target_slot_y))
            self.pre_target_point = self.get_target_point_ego_coord(pred_seg_img, [new_target_x, new_target_y])

    def get_target_point_ego_coord(self, pred_seg_img, target_point_pixel_idx):
        bev_shape = pred_seg_img.shape[0]
        x = -(target_point_pixel_idx[0] - bev_shape / 2)
        y = target_point_pixel_idx[1] - bev_shape / 2
        target_point_ego_coord = [x * self.cfg.bev_x_bound[2], y * self.cfg.bev_y_bound[2]]
        return target_point_ego_coord

    def init_agent(self):
        w = self.world.cam_config['width']
        h = self.world.cam_config['height']

        self.intrinsic_crop = update_intrinsics(
            torch.from_numpy(self.world.intrinsic).float(),
            (h - self.cfg.image_crop) / 2,
            (w - self.cfg.image_crop) / 2,
            scale_width=1,
            scale_height=1
        )
        self.intrinsic_crop = self.intrinsic_crop.unsqueeze(0).expand(4, 3, 3)

        veh2cam_dict = self.world.veh2cam_dict
        front_to_ego = torch.from_numpy(veh2cam_dict['rgb_front']).float().unsqueeze(0)
        left_to_ego = torch.from_numpy(veh2cam_dict['rgb_left']).float().unsqueeze(0)
        right_to_ego = torch.from_numpy(veh2cam_dict['rgb_right']).float().unsqueeze(0)
        rear_to_ego = torch.from_numpy(veh2cam_dict['rgb_rear']).float().unsqueeze(0)
        self.extrinsic = torch.cat([front_to_ego, left_to_ego, right_to_ego, rear_to_ego], dim=0)

        self.image_process = ProcessImage(self.cfg.image_crop)

        self.step = -1
        self.pre_target_point = None

    def save_atten_avg_map(self, data):
        atten = self.save_output.outputs[0].detach().cpu()
        # visualize_heads(atten)

        bev = data['segmentation']
        bev = bev.convert("RGB")
        # visualize_grid_to_grid(atten, 136, bev)
        grid_image, atten_avg = get_atten_avg_map(atten, 136, bev)
        grid_image = np.asarray(grid_image)[::-1, ...]
        atten_avg = np.asarray(atten_avg)[::-1, ...]
        return grid_image, atten_avg

    def tick(self):
        if self.net_eva.agent_need_init:
            self.init_agent()
            self.net_eva.agent_need_init = False

        self.step += 1

        # stop 1s for new eva
        if self.step < 30:
            self.player.apply_control(carla.VehicleControl())
            self.player.set_transform(self.net_eva.ego_transform)
            return

        if self.step % self.process_frequency == 0:
            data_frame = self.world.sensor_data_frame

            if not data_frame:
                return

            vehicle_transform = data_frame['veh_transfrom']
            imu_data = data_frame['imu']

            data = self.get_model_data(data_frame)

            # move data tensors to model device
            for k in ['image', 'intrinsics', 'extrinsics', 'target_point', 'ego_motion']:
                if k in data and isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(self.device)

            self.model.eval()
            with torch.no_grad():
                start_time = time.time()

                # model.predict now returns predicted trajectory (B, T, 2), segmentation, depth, bev_target
                pred_traj, pred_segmentation, _, target_bev = self.model.predict(data)

                # save first predicted point for next step BEV concat (use device-CPU numpy)
                try:
                    first_pt = pred_traj[0, 0, :].detach().cpu().numpy()
                    self.prev_pred_traj_point = first_pt
                except Exception:
                    self.prev_pred_traj_point = None

                end_time = time.time()
                self.net_eva.inference_time.append(end_time - start_time)

                # save segmentation/target for visualization (segmentation may be None)
                if pred_segmentation is not None:
                    try:
                        self.save_prev_target(pred_segmentation)
                    except Exception:
                        pass

                # convert predicted trajectory to low-level vehicle control using VehicleController
                # pred_traj expected shape: (B, T, 2)
                # get current speed from data_frame
                vehicle_velocity = data_frame['veh_velocity']
                current_speed = (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)) / 3.6  # convert km/h to m/s properly -> kept as m/s
                throttle, brake, steer, reverse = self.vehicle_controller.control(pred_traj, current_speed)

                self.trans_control.throttle = float(throttle)
                self.trans_control.brake = float(brake)
                self.trans_control.steer = float(steer)
                self.trans_control.reverse = bool(reverse)

                # set gear according to reverse decision
                try:
                    if reverse:
                        self.trans_control.gear = -1
                    else:
                        self.trans_control.gear = 1
                except Exception:
                    # some CARLA versions don't allow direct gear setting; keep reverse flag
                    pass

                # Log control outputs and predicted trajectory for debugging/verification
                try:
                    # predicted first point for debug
                    pred_first = None
                    pred_points_list = []
                    pred_seq_len = 0
                    if pred_traj is not None:
                        try:
                            # convert to numpy safely
                            if hasattr(pred_traj, 'detach'):
                                pj = pred_traj.detach().cpu().numpy()
                            else:
                                pj = np.array(pred_traj)
                            pred_seq_len = int(pj.shape[1]) if pj.ndim == 3 else int(pj.shape[0])
                            # flatten first up to 5 points
                            if pj.ndim == 3:
                                pts = pj[0]
                            else:
                                pts = pj
                            for i in range(min(5, pts.shape[0])):
                                pred_points_list.append((float(pts[i, 0]), float(pts[i, 1])))
                            if pts.shape[0] > 0:
                                pred_first = pred_points_list[0]
                        except Exception:
                            pred_first = None
                            pred_points_list = []
                            pred_seq_len = 0

                    logging.info('Pred first point=%s | throttle=%.3f brake=%.3f steer=%.3f reverse=%s gear=%s',
                                 str(pred_first), float(self.trans_control.throttle), float(self.trans_control.brake), float(self.trans_control.steer),
                                 bool(self.trans_control.reverse), getattr(self.trans_control, 'gear', 'N/A'))
                    # detect gear switch event
                    if not hasattr(self, '_last_reverse_state'):
                        self._last_reverse_state = bool(self.trans_control.reverse)
                    else:
                        if bool(self.trans_control.reverse) != self._last_reverse_state:
                            logging.info('Gear/reverse state changed: reverse %s -> %s', self._last_reverse_state, bool(self.trans_control.reverse))
                            self._last_reverse_state = bool(self.trans_control.reverse)

                    # write detailed CSV for later analysis
                    try:
                        csv_path = pathlib.Path(self.cfg.log_dir) / 'control_trace.csv'
                        logging.debug('Control CSV path: %s', str(csv_path))
                        header = 'timestamp,step,pred_first_x,pred_first_y,pred_seq_len,first5_points,throttle,brake,steer,reverse,gear,veh_speed,veh_control_throttle,veh_control_brake,veh_control_steer,target_x,target_y,image_hash\n'
                        if not csv_path.exists():
                            with open(csv_path, 'w') as fh:
                                fh.write(header)
                        # get vehicle speed (m/s)
                        try:
                            vel = self.player.get_velocity()
                            veh_speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
                        except Exception:
                            veh_speed = float('nan')
                        # get actual applied control from vehicle (may differ)
                        try:
                            actual_ctrl = self.player.get_control()
                            veh_ctrl_th = float(getattr(actual_ctrl, 'throttle', 0.0))
                            veh_ctrl_br = float(getattr(actual_ctrl, 'brake', 0.0))
                            veh_ctrl_st = float(getattr(actual_ctrl, 'steer', 0.0))
                        except Exception:
                            veh_ctrl_th = float('nan')
                            veh_ctrl_br = float('nan')
                            veh_ctrl_st = float('nan')

                        # compute target_point and image hash if available
                        first_x = '' if pred_first is None else f"{pred_first[0]:.6f}"
                        first_y = '' if pred_first is None else f"{pred_first[1]:.6f}"
                        target_x = ''
                        target_y = ''
                        image_hash = ''
                        try:
                            if 'target_point' in data:
                                tp = data['target_point']
                                if hasattr(tp, 'detach'):
                                    tp = tp.detach().cpu().numpy()
                                tp = np.array(tp).reshape(-1)
                                target_x = f"{float(tp[0]):.6f}"
                                target_y = f"{float(tp[1]):.6f}"
                        except Exception:
                            target_x = ''
                            target_y = ''
                        try:
                            import hashlib
                            if 'image' in data:
                                img = data['image']
                                if hasattr(img, 'detach'):
                                    arr = img.detach().cpu().numpy()
                                else:
                                    arr = np.array(img)
                                # use raw bytes for hash
                                h = hashlib.md5(arr.tobytes()).hexdigest()
                                image_hash = h
                        except Exception:
                            image_hash = ''
                        first5_str = '"' + ','.join([f'({x:.4f}:{y:.4f})' for (x, y) in pred_points_list]) + '"' if pred_points_list else ''
                        line = f"{time.time():.3f},{int(self.step)},{first_x},{first_y},{pred_seq_len},{first5_str},{float(self.trans_control.throttle):.6f},{float(self.trans_control.brake):.6f},{float(self.trans_control.steer):.6f},{int(bool(self.trans_control.reverse))},{getattr(self.trans_control,'gear','N/A')},{veh_speed:.6f},{veh_ctrl_th:.6f},{veh_ctrl_br:.6f},{veh_ctrl_st:.6f},{target_x},{target_y},{image_hash}\n"
                        with open(csv_path, 'a') as fh:
                            fh.write(line)
                    except Exception:
                        logging.exception('Failed to write control CSV')

                except Exception:
                    logging.exception('Failed to log control debug info')

                self.speed_limit(data_frame)

                if self.show_eva_imgs:
                    self.grid_image, self.atten_avg = self.save_atten_avg_map(data)
                    self.save_seg_img(pred_segmentation)
                    self.save_target_bev_img(target_bev)
                    self.display_imgs()

                self.save_output.clear()

            self.prev_xy_thea = [vehicle_transform.location.x,
                                 vehicle_transform.location.y,
                                 imu_data.compass if np.isnan(imu_data.compass) else 0]

        self.player.apply_control(self.trans_control)

    def speed_limit(self, data_frame):
        # if vehicle stops at initialization, give throttle until Gear turns to 1
        if data_frame['veh_control'].gear == 0:
            self.trans_control.throttle = 0.5

        speed = (3.6 * math.sqrt(
            data_frame['veh_velocity'].x ** 2 + data_frame['veh_velocity'].y ** 2 + data_frame['veh_velocity'].z ** 2))

        # limit the vehicle speed within 15km/h when reverse is False
        if not self.trans_control.reverse and speed >= 12:
            self.trans_control.throttle = 0.0

        # limit the vehicle speed within 8km/h when reverse is True
        if self.trans_control.reverse and speed >= 10:
            self.trans_control.throttle = 0.0

        # if brake and throttle both not on, and speed < 2 for more than 2 seconds, give it a small throttle for 1
        # second
        if self.trans_control.throttle < 1e-5 and self.trans_control.brake < 1e-5 and speed < 2.0:
            self.stop_count += 1
        else:
            self.stop_count = 0.0

        if self.stop_count > 10:  # 1s
            self.boost = True

        if self.boost:
            self.trans_control.throttle = 0.3
            self.boot_step += 1

        if self.boot_step > 10 or self.trans_control.brake > 1e-5:  # 1s
            self.boot_step = 0
            self.boost = False

    def get_model_data(self, data_frame):

        vehicle_transform = data_frame['veh_transfrom']
        imu_data = data_frame['imu']
        vehicle_velocity = data_frame['veh_velocity']

        data = {}

        target_point = convert_slot_coord(vehicle_transform, self.net_eva.eva_parking_goal)

        front_final, self.rgb_front = self.image_process(data_frame['rgb_front'])
        left_final, self.rgb_left = self.image_process(data_frame['rgb_left'])
        right_final, self.rgb_right = self.image_process(data_frame['rgb_right'])
        rear_final, self.rgb_rear = self.image_process(data_frame['rgb_rear'])

        images = [front_final, left_final, right_final, rear_final]
        images = torch.cat(images, dim=0)
        data['image'] = images.unsqueeze(0)

        data['extrinsics'] = self.extrinsic.unsqueeze(0)
        data['intrinsics'] = self.intrinsic_crop.unsqueeze(0)

        velocity = (3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2))
        data['ego_motion'] = torch.tensor([velocity, imu_data.accelerometer.x, imu_data.accelerometer.y],
                                          dtype=torch.float).unsqueeze(0).unsqueeze(0)

        if self.pre_target_point is not None:
            target_point = [self.pre_target_point[0], self.pre_target_point[1], target_point[2]]
        data['target_point'] = torch.tensor(target_point, dtype=torch.float).unsqueeze(0)

        # agent does not need to provide gt control during inference; model.predict will return predicted trajectory
        # keep compatibility with show_eva_imgs that may expect segmentation/depth images

        # if concat_traj_to_bev enabled in cfg, provide previous predicted traj point (if any)
        if getattr(self.cfg, 'concat_traj_to_bev', False):
            if self.prev_pred_traj_point is not None:
                # prev_pred_traj_point expected as (2,) numpy or torch
                if isinstance(self.prev_pred_traj_point, torch.Tensor):
                    data['traj_point'] = self.prev_pred_traj_point.unsqueeze(0).to(self.cfg.device)
                else:
                    data['traj_point'] = torch.tensor(self.prev_pred_traj_point, dtype=torch.float).unsqueeze(0).to(self.cfg.device)
            else:
                # when no previous predicted trajectory is available, provide a zero traj_point
                # so BEV encoder receives the expected extra channel (model was built with concat_traj_to_bev=True)
                # shape: (B=1, 2)
                data['traj_point'] = torch.zeros((1, 2), dtype=torch.float)

        if self.show_eva_imgs:
            img = encode_npy_to_pil(np.asarray(data_frame['topdown'].squeeze().cpu()))
            img = np.moveaxis(img, 0, 2)
            img = Image.fromarray(img)
            seg_gt = self.semantic_process(image=img, scale=0.5, crop=200, target_slot=target_point)
            seg_gt[seg_gt == 1] = 128
            seg_gt[seg_gt == 2] = 255
            data['segmentation'] = Image.fromarray(seg_gt)

        return data

    def traj_to_control(self, traj_tensor):
        """
        Convert predicted trajectory (relative xy points in ego frame) to a simple control command.
        traj_tensor: Tensor of shape (T, 2) or (1, T, 2) - we take the first future point as target.
        Returns: (throttle, brake, steer, reverse)
        """
        if isinstance(traj_tensor, torch.Tensor):
            traj = traj_tensor.detach().cpu().numpy()
        else:
            traj = traj_tensor

        # normalize shapes
        if traj.ndim == 3:
            traj = traj[0]

        # take first point (closest future point)
        if traj.shape[0] == 0:
            return 0.0, 0.0, 0.0, False

        x, y = float(traj[0][0]), float(traj[0][1])

        # heading error
        desired_yaw = math.atan2(y, x) if (x != 0.0 or y != 0.0) else 0.0
        # map yaw into steering [-1, 1] (assume max steering corresponds to 45 degrees)
        steer = max(-1.0, min(1.0, desired_yaw / (math.pi / 4)))

        # distance-based throttle: scale distance to [0,1] with a simple factor
        dist = math.hypot(x, y)
        max_speed_distance = 5.0  # meters -> tuneable
        throttle = max(0.0, min(1.0, dist / max_speed_distance))
        brake = 0.0
        reverse = False

        # if far behind (x < 0) consider reverse briefly
        if x < -0.2:
            reverse = True
            throttle = 0.0
            brake = min(1.0, dist / max_speed_distance)

        return throttle, brake, steer, reverse

    def get_eva_control(self):
        """Return a dict suitable for the UI show_control_info function.

        Keys: 'throttle', 'brake', 'steer', 'reverse'. Uses current trans_control if available
        and otherwise returns safe defaults.
        """
        # default values
        ctrl_dict = {'throttle': 0.0, 'brake': 0.0, 'steer': 0.0, 'reverse': False}
        try:
            tc = getattr(self, 'trans_control', None)
            if tc is not None:
                # Access attributes with safe fallback
                ctrl_dict['throttle'] = float(getattr(tc, 'throttle', 0.0)) if getattr(tc, 'throttle', None) is not None else 0.0
                ctrl_dict['brake'] = float(getattr(tc, 'brake', 0.0)) if getattr(tc, 'brake', None) is not None else 0.0
                ctrl_dict['steer'] = float(getattr(tc, 'steer', 0.0)) if getattr(tc, 'steer', None) is not None else 0.0
                ctrl_dict['reverse'] = bool(getattr(tc, 'reverse', False))
        except Exception:
            # on any unexpected error return defaults
            pass
        return ctrl_dict

    def draw_waypoints(self, waypoints):
        ego_t = self.world.player.get_transform()
        ego_loc = carla.Location(x=ego_t.location.x, y=ego_t.location.y, z=0.20)
        self.world.world.debug.draw_string(ego_loc, 'O', draw_shadow=True, color=carla.Color(255, 0, 0))

        wp_list = waypoints[0].tolist()
        for wp in wp_list:
            logging.info('wp: dx: %4f; dy: %4f;', wp[0], wp[1])
            loc = carla.Location(x=ego_t.location.x + wp[0], y=ego_t.location.y + wp[1], z=0.20)
            self.world.world.debug.draw_string(loc, 'O', draw_shadow=True, color=carla.Color(0, 255, 0))

    def render_predicted_traj(self, pred_traj, duration=0.2, point_size=0.05):
        """
        Draw predicted trajectory (relative ego frame points) into CARLA world for visualization.
        pred_traj: tensor or numpy array shape (B,T,2) or (T,2)
        duration: seconds the debug points/lines persist in CARLA
        point_size: size of drawn points
        """
        if pred_traj is None:
            return
        if hasattr(pred_traj, 'detach'):
            pred = pred_traj.detach().cpu().numpy()
        else:
            pred = np.array(pred_traj)

        if pred.ndim == 3:
            pred = pred[0]

        # get ego transform
        ego_t = self.world.player.get_transform()
        ego_loc = ego_t.location
        ego_rot = ego_t.rotation

        # helper to rotate local xy by yaw and add to ego world pos
        def local_to_world(x_local, y_local, ego_loc, ego_rot):
            # yaw in degrees -> radians
            yaw = math.radians(ego_rot.yaw)
            # in dataset/this project, x forward, y left
            wx = ego_loc.x + (x_local * math.cos(yaw) - y_local * math.sin(yaw))
            wy = ego_loc.y + (x_local * math.sin(yaw) + y_local * math.cos(yaw))
            return carla.Location(x=wx, y=wy, z=ego_loc.z + 0.2)

        # draw points and connecting lines
        prev_loc = None
        color_step = 255 // max(1, len(pred))
        for i, p in enumerate(pred):
            x_local = float(p[0])
            y_local = float(p[1])
            world_loc = local_to_world(x_local, y_local, ego_loc, ego_rot)
            # color gradient from green to red along trajectory
            r = int(min(255, i * color_step))
            g = int(max(0, 255 - i * color_step))
            self.world.world.debug.draw_point(world_loc, size=point_size, color=carla.Color(r, g, 0), life_time=duration)
            if prev_loc is not None:
                self.world.world.debug.draw_line(prev_loc, world_loc, thickness=0.05, color=carla.Color(r, g, 0), life_time=duration)
            prev_loc = world_loc
