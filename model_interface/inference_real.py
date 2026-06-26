from collections import OrderedDict, deque
import numpy as np
import threading
import time
import os

import rospy
import torch
import torchvision
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path
from std_srvs.srv import SetBool

from model_interface.model.parking_model_real import ParkingModelReal
from utils.camera_utils import CameraInfoParser, ProcessImage, get_normalized_torch_image, get_torch_intrinsics_or_extrinsics
from utils.config import InferenceConfiguration
from utils.pose_utils import PoseFlow, pose2customize_pose
from utils.ros_interface import RosInterface
from utils.traj_post_process import calculate_tangent, fitting_curve
from utils.trajectory_utils import detokenize_traj_point
from utils.evaluation import EvaluationManager


class ParkingInferenceModuleReal:
    def __init__(self, inference_cfg: InferenceConfiguration, ros_interface_obj: RosInterface,
                 eval_bag_path: str = None, eval_out_dir: str = None):
        self.cfg = inference_cfg
        self.model = None
        self.device = None

        self.images_tag = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear")

        self.ros_interface_obj = ros_interface_obj

        self.load_model(self.cfg.model_ckpt_path)

        self.BOS_token = self.cfg.train_meta_config.token_nums

        self.traj_start_point_info = Pose()
        self.traj_start_point_lock = threading.Lock()

        camera_info_obj = CameraInfoParser(task_index=-1, parser_dir=self.cfg.cam_info_dir)
        self.intrinsic, self.extrinsic = camera_info_obj.intrinsic, camera_info_obj.extrinsic
        self.EOS_token = self.cfg.train_meta_config.token_nums + self.cfg.train_meta_config.append_token - 2

        self.pub = rospy.Publisher("e2e_traj_pred_topic", Path, queue_size=1)
        # optional evaluation manager
        self.eval_manager = None
        if eval_bag_path is not None:
            out_dir = eval_out_dir if eval_out_dir is not None else os.path.join("log", "eval", str(int(time.time())))
            self.eval_manager = EvaluationManager(eval_bag_path, out_dir)

        # history buffer for past vehicle poses (in ego coordinates)
        # use history_len from training config if available
        hist_len = getattr(self.cfg.train_meta_config, 'history_len', None)
        self.use_history = bool(getattr(self.cfg.train_meta_config, 'use_history', False))
        if self.use_history and hist_len is not None:
            try:
                self.history_buffer = deque(maxlen=int(hist_len))
            except Exception:
                self.history_buffer = deque(maxlen=10)
        else:
            self.history_buffer = None

    def _append_current_pose_to_history(self):
        """Query ros_interface for current pose, transform into ego frame (relative to traj_start_point_info),
        and append (x,y) to history_buffer. If buffer not enabled, do nothing.
        """
        if not self.use_history or self.history_buffer is None:
            return
        curr_pose = self.ros_interface_obj.get_pose()
        if curr_pose is None:
            return
        # convert to CustomizePose (world coords)
        try:
            curr_custom = pose2customize_pose(curr_pose)
            # compute world->ego matrix using latest start pose
            ego2world = pose2customize_pose(self.traj_start_point_info)
            world2ego_mat = ego2world.get_homogeneous_transformation().get_inverse_matrix()
            curr_in_ego = curr_custom.get_pose_in_ego(world2ego_mat)
            self.history_buffer.append([curr_in_ego.x, curr_in_ego.y])
        except Exception:
            # fail-safe: ignore history update
            return

    def _get_history_tensor(self, device=None):
        """Return history tensor shaped (1, hist_len, 2) in float32 on provided device, or None if no history.
        If buffer shorter than maxlen, pad zeros at the front so the most recent entries are at the end.
        """
        if not self.use_history or self.history_buffer is None:
            return None
        import torch as _torch
        hist_list = list(self.history_buffer)
        maxlen = self.history_buffer.maxlen
        if len(hist_list) == 0:
            return None
        # pad at front
        if len(hist_list) < maxlen:
            pad_n = maxlen - len(hist_list)
            pad = [[0.0, 0.0]] * pad_n
            hist_list = pad + hist_list
        arr = _torch.tensor([hist_list], dtype=_torch.float32)
        if device is not None:
            arr = arr.to(device)
        return arr

    def predict(self, mode="service"):
        if mode == "topic":
            self.pub_topic()
        elif mode == "service":
            rospy.Service("/e2e_parking/srv_start", SetBool, self.pub_srv)
            rospy.spin()
        else:
            assert print("Can't support %s mode!".format(mode))

    def pub_srv(self, msg=None):
        self.get_start_pose()
        self.pub_path()
        return [True, "OK"]

    def pub_topic(self):
        while not self.ros_interface_obj.get_rviz_target():
            time.sleep(1)

        rate = rospy.Rate(self.cfg.trajectory_pub_frequency)
        while not rospy.is_shutdown():
            self.get_start_pose()
            self.pub_path()
            rate.sleep()

    def pub_path(self):
        images, intrinsics, extrinsics = self.get_format_images()
        # update history buffer with current pose before building data
        self._append_current_pose_to_history()
        data = {
            "image": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics
        }

        # attach history_traj if enabled
        hist_tensor = self._get_history_tensor(device=(images.device if hasattr(images, 'device') else None))
        if hist_tensor is not None:
            data['history_traj'] = hist_tensor
        else:
            # keep key for compatibility; downstream will ignore None
            data['history_traj'] = None

        target_point_info = self.ros_interface_obj.get_rviz_target()
        target_point_info.position.z = self.traj_start_point_info.position.z
        target_in_world = pose2customize_pose(target_point_info)

        ego2world = pose2customize_pose(self.traj_start_point_info)
        world2ego_mat = ego2world.get_homogeneous_transformation().get_inverse_matrix()
        target_in_ego = target_in_world.get_pose_in_ego(world2ego_mat)

        target_point = [[target_in_ego.x, target_in_ego.y]]

        data["target_point"] = torch.from_numpy(np.array(target_point).astype(np.float32))
        data["fuzzy_target_point"] = data["target_point"]
        start_token = [self.BOS_token]
        data["gt_traj_point_token"] = torch.tensor([start_token], dtype=torch.int64).cuda()

        self.model.eval()
        delta_predicts = self.inference(data)
        delta_predicts = fitting_curve(delta_predicts, num_points=self.cfg.train_meta_config.autoregressive_points,
                                       item_number=self.cfg.train_meta_config.item_number)
        traj_yaw_path = calculate_tangent(np.array(delta_predicts)[:, :2], mode="five_point")

        msg = Path()
        msg.header.frame_id = "base_link"
        for (point_item, traj_yaw) in zip(delta_predicts, traj_yaw_path):
            if self.cfg.train_meta_config.item_number == 2:
                x, y = point_item
            elif self.cfg.train_meta_config.item_number == 3:
                x, y, progress_bar = point_item
                if abs(progress_bar) < 1 - self.cfg.progress_threshold:
                    break
            msg.poses.append(self.get_posestamp_info(x, y, traj_yaw))
        msg.header.stamp = rospy.Time.now()
        # record prediction for evaluation if enabled
        if self.eval_manager is not None:
            try:
                self.eval_manager.record_prediction(msg, self.traj_start_point_info)
            except Exception as e:
                print(f"[Inference] evaluation manager record failed: {e}")
        self.pub.publish(msg)

    def inference(self, data):
        delta_predicts = []
        with torch.no_grad():
            if self.cfg.train_meta_config.decoder_method == "transformer":
                delta_predicts = self.inference_transformer(data)
            elif self.cfg.train_meta_config.decoder_method == "gru":
                delta_predicts = self.inference_gru(data)
            else:
                raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        delta_predicts = delta_predicts.tolist()
        return delta_predicts

    def inference_transformer(self, data):
        pred_traj_point, pred_segmentation, pred_depth, bev_target = self.model.predict_transformer(
            data, predict_token_num=self.cfg.train_meta_config.item_number * self.cfg.train_meta_config.autoregressive_points)
        pred_traj_point_update = pred_traj_point[0][1:]
        pred_traj_point_update = self.remove_invalid_content(pred_traj_point_update)

        delta_predicts = detokenize_traj_point(pred_traj_point_update, self.cfg.train_meta_config.token_nums,
                                               self.cfg.train_meta_config.item_number,
                                               self.cfg.train_meta_config.xy_max)

        return delta_predicts

    def inference_gru(self, data):
        delta_predicts = self.model.predict_gru(data)

        return delta_predicts

    def remove_invalid_content(self, pred_traj_point_update):
        finish_index = -1
        index_tensor = torch.where(
            pred_traj_point_update == self.cfg.train_meta_config.token_nums + self.cfg.train_meta_config.append_token - 2)[
            0]
        if len(index_tensor):
            finish_index = torch.where(pred_traj_point_update == self.EOS_token)[0][0].item()
            finish_index = finish_index - finish_index % self.cfg.train_meta_config.item_number
        if finish_index != -1:
            pred_traj_point_update = pred_traj_point_update[: finish_index]
        return pred_traj_point_update

    def get_posestamp_info(self, x, y, yaw):
        predict_pose = PoseStamped()
        pose_flow_obj = PoseFlow(att_input=[yaw, 0, 0], type="euler", deg_or_rad="deg")
        quad = pose_flow_obj.get_quad()
        predict_pose.pose.position = Point(x=x, y=y, z=0.0)
        predict_pose.pose.orientation = Quaternion(x=quad.x, y=quad.y, z=quad.z, w=quad.w)
        return predict_pose

    def get_start_pose(self):
        self.traj_start_point_lock.acquire()
        tmp_start_point_info = None
        cnt = 0
        while tmp_start_point_info == None:
            tmp_start_point_info = self.ros_interface_obj.get_pose()

            if cnt > 10: time.sleep(1)
            cnt += 1
        self.traj_start_point_info = tmp_start_point_info
        self.traj_start_point_lock.release()

    def get_images(self, img_tag):
        images = None
        cnt = 0
        while images == None:
            images = self.ros_interface_obj.get_images(img_tag)

            if cnt > 10: time.sleep(1)
            cnt += 1
        return images

    def load_model(self, parking_pth_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ParkingModelReal(self.cfg.train_meta_config)

        ckpt = torch.load(parking_pth_path, map_location='cuda:0')
        state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def get_format_images(self):
        process_width, process_height = int(self.cfg.train_meta_config.process_dim[0]), int(
            self.cfg.train_meta_config.process_dim[1])
        images, intrinsics, extrinsics = [], [], []
        for image_tag in self.images_tag:
            pil_image = self.torch2pillow()(self.get_images(image_tag))
            image_obj = ProcessImage(pil_image,
                                     self.intrinsic[image_tag],
                                     self.extrinsic[image_tag],
                                     target_size=(process_width, process_height))

            image_obj.resize_pil_image()
            images.append(get_normalized_torch_image(image_obj.resize_img))
            intrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.resize_intrinsics))
            extrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.extrinsics))

        images = torch.cat(images, dim=0).unsqueeze(0)
        intrinsics = torch.cat(intrinsics, dim=0).unsqueeze(0)
        extrinsics = torch.cat(extrinsics, dim=0).unsqueeze(0)

        return images, intrinsics, extrinsics

    def torch2pillow(self):
        return torchvision.transforms.transforms.ToPILImage()
