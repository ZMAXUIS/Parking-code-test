import os
from typing import List

import numpy as np
import torch
from shapely.geometry import LineString
from shapely.measurement import hausdorff_distance

from utils.common import get_json_content
from utils.pose_utils import CustomizePose
from loguru import logger
from datetime import datetime
import json
import threading

# file to record normalization issues (append-only)
_NORMALIZATION_ISSUES_FILE = os.path.join(os.getcwd(), 'normalization_issues.log')
_LOG_LOCK = threading.Lock()


def tokenize_traj_point(x, y, progress, token_nums, xy_max, progress_bar=1, context: str = None,
                        raise_threshold: float = 0.5):
    """
    Tokenize trajectory points
    :param x: [-xy_max, xy_max]
    :param y: [-xy_max, xy_max]
    :param progress: [-progress_bar, progress_bar]
    :return: tokenized control range [0, token_nums]
    """
    # compute normalized values
    valid_token = token_nums - 1
    x_normalize = (x + xy_max) / (2 * xy_max)
    y_normalize = (y + xy_max) / (2 * xy_max)
    progress_normalize = (progress + progress_bar) / (2 * progress_bar)

    # check ranges and always clamp to [0,1] to avoid DataLoader worker crash
    out_of_range = (x_normalize < 0.0 or x_normalize > 1.0 or
                    y_normalize < 0.0 or y_normalize > 1.0 or
                    progress_normalize < 0.0 or progress_normalize > 1.0)

    if out_of_range:
        # prepare diagnostic entry
        entry = {
            'time': datetime.now().isoformat(),
            'context': context,
            'x_raw': float(x), 'y_raw': float(y), 'progress_raw': float(progress),
            'xy_max': float(xy_max), 'progress_bar': float(progress_bar),
            'x_norm': float(x_normalize), 'y_norm': float(y_normalize), 'progress_norm': float(progress_normalize)
        }
        # append to diagnostics file (thread-safe); multiple workers append concurrently
        try:
            with _LOG_LOCK:
                with open(_NORMALIZATION_ISSUES_FILE, 'a') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception:
            # fallback to logger if file write fails
            logger.warning("Failed to write normalization diagnostic entry: {}", entry)

        # If values are wildly out of range, raise so user notices a real data issue
        max_deviation = max(abs(x_normalize - 0.5) - 0.5, abs(y_normalize - 0.5) - 0.5,
                            abs(progress_normalize - 0.5) - 0.5)
        # simpler: if any normalized value deviates from [0,1] by more than raise_threshold, consider it fatal
        if (x_normalize < -raise_threshold or x_normalize > 1.0 + raise_threshold or
                y_normalize < -raise_threshold or y_normalize > 1.0 + raise_threshold or
                progress_normalize < -raise_threshold or progress_normalize > 1.0 + raise_threshold):
            raise ValueError(f"Trajectory normalization grossly out of range: {entry}")

        # log a warning then clamp
        logger.warning(
            "Trajectory normalization out of [0,1], clamping. context={}, x_norm={:.6f}, y_norm={:.6f}, progress_norm={:.6f}",
            context, x_normalize, y_normalize, progress_normalize)

    # clamp values to [0,1]
    x_normalize = min(max(0.0, x_normalize), 1.0)
    y_normalize = min(max(0.0, y_normalize), 1.0)
    progress_normalize = min(max(0.0, progress_normalize), 1.0)

    return [int(x_normalize * valid_token), int(y_normalize * valid_token), int(progress_normalize * valid_token)]


def detokenize_traj_point(torch_list: torch.Tensor, token_nums, item_num, xy_max=10, progress_max=1):
    valid_token = token_nums - 1

    torch_list_process = torch_list.view(-1, item_num)
    ret_tensor = torch.zeros_like(torch_list_process, dtype=torch.float32)
    ret_tensor[:, :2] = (torch_list_process[:, :2] / valid_token) * 2 * xy_max - xy_max
    if (item_num > 2):
        ret_tensor[:, 2:] = (torch_list_process[:, 2:] / valid_token) * 2 * progress_max - progress_max
    return ret_tensor


class TrajectoryInfoParser:
    def __init__(self, task_index, task_path):
        self.task_index = task_index
        self.task_path = task_path
        self.total_frames = self._get_trajectory_num()
        self.trajectory_list = self.make_trajectory()
        self.progress_list = self.get_progress_list()
        self.candidate_target_pose = self.get_candidate_target_pose()

    def _get_trajectory_num(self) -> int:
        return len(os.listdir(os.path.join(self.task_path, "measurements")))

    def get_trajectory_point(self, point_index) -> CustomizePose:
        return self.trajectory_list[point_index]

    def get_progress(self, index) -> float:
        return self.progress_list[index]

    def _get_trajectory_direction(self, bias_threshold=30) -> str:
        direction = None
        delta_yaw = self.get_safe_yaw(self.trajectory_list[-1].yaw - self.trajectory_list[0].yaw)
        if 90 - bias_threshold < abs(delta_yaw) < 90 + bias_threshold:
            direction = "right" if delta_yaw > 0 else "left"
        else:
            raise ValueError(f"Don't support trajectory rotation angle '{delta_yaw}'!")
        return direction

    def get_candidate_target_pose(self) -> List[CustomizePose]:
        candidate_target_pose = []
        for trajectory_item in self.trajectory_list:
            if abs(trajectory_item.yaw - self.trajectory_list[-1].yaw < 1):
                candidate_target_pose.append(trajectory_item)
        return candidate_target_pose

    def get_random_candidate_target_pose(self) -> CustomizePose:
        candidate_target_pose_len = len(self.candidate_target_pose)
        candidate_target_pose = self.candidate_target_pose[np.random.choice(range(0, candidate_target_pose_len))]
        # noise = 0.4 * (2 * np.random.rand(*candidate_target_pose.shape) - 1) # 扰动从[-0.4m, 0.4m]
        # candidate_target_pose += noise
        return candidate_target_pose

    def get_precise_target_pose(self) -> CustomizePose:
        return self.trajectory_list[-1]

    def get_safe_yaw(self, yaw) -> int:
        if yaw <= -180:
            yaw += 360
        if yaw > 180:
            yaw -= 360
        return yaw

    def get_measurement_path(self, measurement_index) -> str:
        return os.path.join(self.task_path, "measurements", "{}.json".format(str(measurement_index).zfill(4)))

    def make_trajectory(self) -> List[CustomizePose]:
        trajectory_list = []
        for frame in range(0, self.total_frames):
            data = get_json_content(self.get_measurement_path(frame))
            cur_pose = CustomizePose(x=data["x"], y=data["y"], z=data["z"], roll=data["roll"], yaw=data["yaw"],
                                     pitch=data["pitch"])
            trajectory_list.append(cur_pose)
        return trajectory_list

    def get_progress_list(self) -> List[float]:
        distance_list = [0.0]
        for index in range(1, self.total_frames):
            distance_list.append(distance_list[-1] + self._get_backwark_delta_distance(index))
        progress_list = 1 - np.array(distance_list) / distance_list[-1]
        if self._get_trajectory_direction() == "left":
            progress_list = -progress_list
        return progress_list.tolist()

    def _get_backwark_delta_distance(self, index) -> float:
        delta_y = self.get_trajectory_point(index).y - self.get_trajectory_point(index - 1).y
        delta_x = self.get_trajectory_point(index).x - self.get_trajectory_point(index - 1).x
        return np.linalg.norm([delta_x, delta_y])


class TrajectoryDistance:
    def __init__(self, prediction_points_np, gt_points_np):
        self.prediction_points_np = prediction_points_np
        self.gt_points_np = gt_points_np

        self.cut_stop_segment()

    def cut_stop_segment(self, stop_threshold=0.001):
        distance_list = np.linalg.norm(self.gt_points_np[1:, :] - self.gt_points_np[:-1, :], axis=-1)

        threshold_bool_list = abs(distance_list) < stop_threshold

        stop_index = -1
        for index in range(0, len(threshold_bool_list)):
            inverse_index = len(threshold_bool_list) - index - 1
            if not threshold_bool_list[inverse_index]:
                stop_index = inverse_index + 1
                break
        self.prediction_points_np = self.prediction_points_np[:stop_index + 1]
        self.gt_points_np = self.gt_points_np[:stop_index + 1]

    def get_len(self):
        return self.gt_points_np.shape[0]

    def get_l2_distance(self):
        if self.gt_points_np.size == 0 or self.prediction_points_np.size == 0:
            return float('nan')
        l2_distance_list = np.linalg.norm(self.gt_points_np - self.prediction_points_np, axis=1)
        if l2_distance_list.size == 0:
            return float('nan')
        l2_distance = float(np.mean(l2_distance_list))
        return l2_distance

    def get_haus_distance(self):
        line_gt = LineString(self.gt_points_np)
        line_pred = LineString(self.prediction_points_np)
        haus_distance = hausdorff_distance(line_pred, line_gt)
        return haus_distance

    def get_fourier_difference(self):
        fd1 = self.compute_fourier_descriptor(self.gt_points_np, num_descriptors=10)
        fd2 = self.compute_fourier_descriptor(self.prediction_points_np, num_descriptors=10)
        fourier_difference = np.linalg.norm(fd1 - fd2)
        return fourier_difference

    def compute_fourier_descriptor(self, points, num_descriptors):
        n = points.shape[0]
        if n == 0:
            # return zeros to indicate no frequency content
            return np.zeros(num_descriptors, dtype=float)

        complex_points = points[:, 0].astype(float) + 1j * points[:, 1].astype(float)
        descriptors = np.fft.fft(complex_points)
        mags = np.abs(descriptors)
        # Take up to num_descriptors; if insufficient length, pad with zeros
        if mags.shape[0] >= num_descriptors:
            out = mags[:num_descriptors]
        else:
            out = np.zeros(num_descriptors, dtype=float)
            out[:mags.shape[0]] = mags
        return out
