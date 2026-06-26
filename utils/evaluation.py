import os
import json
import threading
import bisect
from typing import List

import numpy as np
from geometry_msgs.msg import PoseStamped
try:
    import rosbag
except Exception:
    rosbag = None

from utils.trajectory_utils import TrajectoryDistance
from utils.pose_utils import pose2customize_pose


class EvaluationManager:
    """Manager for online evaluation during inference.

    Usage:
      eval_mgr = EvaluationManager(bag_path, out_dir)
      eval_mgr.record_prediction(path_msg, start_pose)
    """

    def __init__(self, bag_path: str, out_dir: str, cfg=None):
        self.bag_path = bag_path
        self.out_dir = out_dir
        self.cfg = cfg or {}
        os.makedirs(self.out_dir, exist_ok=True)
        self.events_dir = os.path.join(self.out_dir, "events")
        os.makedirs(self.events_dir, exist_ok=True)

        # index for ego_pose: list of (time_sec, PoseStamped)
        self.pose_times = []
        self.pose_msgs = []
        self.bag = None
        self.lock = threading.Lock()

        if bag_path is not None and os.path.exists(bag_path) and rosbag is not None:
            try:
                self.bag = rosbag.Bag(bag_path)
                self._index_bag()
            except Exception as e:
                print(f"[EvaluationManager] failed to open bag {bag_path}: {e}")
                self.bag = None

        # running stats
        self.event_count = 0
        self.metrics_list = []
        self.one_step_list = []

        # pending previous prediction for one-step comparison
        self.pending_prev_event = None
        self.pending_prev_pred_world = None

    def _index_bag(self):
        for topic, msg, t in self.bag.read_messages(topics=["/ego_pose"]):
            # store ros::Time to float seconds
            try:
                ts = msg.header.stamp.to_sec()
            except Exception:
                # fallback to t
                ts = t.to_sec()
            self.pose_times.append(ts)
            self.pose_msgs.append(msg)

    def find_gt_poses(self, start_time: float, num_points: int) -> List[PoseStamped]:
        """Find next num_points pose stamped messages from bag starting at >= start_time."""
        if not self.pose_times:
            return []
        idx = bisect.bisect_left(self.pose_times, start_time)
        end_idx = min(idx + num_points, len(self.pose_msgs))
        return self.pose_msgs[idx:end_idx]

    def _pathmsg_to_np(self, path_msg) -> np.ndarray:
        pts = []
        for pose_stamped in path_msg.poses:
            p = pose_stamped.pose.position
            pts.append([p.x, p.y])
        return np.array(pts, dtype=np.float32)

    def _gt_msgs_to_ego_np(self, start_pose, gt_msgs: List[PoseStamped]) -> np.ndarray:
        ego2world = pose2customize_pose(start_pose)
        world2ego_mat = ego2world.get_homogeneous_transformation().get_inverse_matrix()
        pts = []
        for msg in gt_msgs:
            gt_pose = pose2customize_pose(msg.pose)
            gt_in_ego = gt_pose.get_pose_in_ego(world2ego_mat)
            pts.append([gt_in_ego.x, gt_in_ego.y])
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(pts, dtype=np.float32)

    def record_prediction(self, path_msg, start_pose):
        with self.lock:
            self.event_count += 1
            event_id = str(self.event_count).zfill(5)

            start_time = path_msg.header.stamp.to_sec()
            pred_np = self._pathmsg_to_np(path_msg)

            # compute world coord of first predicted point for one-step evaluation
            first_pred_world = None
            if pred_np.shape[0] > 0:
                try:
                    ego2world = pose2customize_pose(start_pose).get_homogeneous_transformation().get_matrix()
                    first = np.array([pred_np[0,0], pred_np[0,1], 0.0, 1.0])
                    first_world = ego2world @ first
                    first_pred_world = [float(first_world[0]), float(first_world[1])]
                except Exception:
                    first_pred_world = None

            gt_msgs = self.find_gt_poses(start_time, pred_np.shape[0])
            gt_np = self._gt_msgs_to_ego_np(start_pose, gt_msgs)

            note = ""
            if gt_np.shape[0] == 0:
                note = "no_gt"

            # align lengths by truncation to shortest
            min_len = min(pred_np.shape[0], gt_np.shape[0])
            if min_len == 0:
                metrics = {"l2_mean": None, "hausdorff": None, "fourier": None}
            else:
                pred_np_aligned = pred_np[:min_len]
                gt_np_aligned = gt_np[:min_len]
                td = TrajectoryDistance(pred_np_aligned, gt_np_aligned)
                metrics = {
                    "l2_mean": float(td.get_l2_distance()),
                    "hausdorff": float(td.get_haus_distance()),
                    "fourier": float(td.get_fourier_difference())
                }
                self.metrics_list.append(metrics)

            event = {
                "event_id": event_id,
                "start_time": start_time,
                "pred_num": int(pred_np.shape[0]),
                "gt_num": int(gt_np.shape[0]),
                "metrics": metrics,
                "note": note
            }

            with open(os.path.join(self.events_dir, f"{event_id}.json"), "w") as f:
                json.dump(event, f, indent=2)

            # if there is a pending previous prediction, compute one-step L2 between its first predicted world point and current start_pose
            if self.pending_prev_event is not None and self.pending_prev_pred_world is not None:
                try:
                    cur_x = start_pose.position.x
                    cur_y = start_pose.position.y
                    dx = self.pending_prev_pred_world[0] - cur_x
                    dy = self.pending_prev_pred_world[1] - cur_y
                    one_step_l2 = float(np.linalg.norm([dx, dy]))
                    # update previous event file
                    prev_path = os.path.join(self.events_dir, f"{self.pending_prev_event}.json")
                    try:
                        with open(prev_path, 'r') as pf:
                            prev_event = json.load(pf)
                    except Exception:
                        prev_event = {}
                    prev_event['one_step_l2'] = one_step_l2
                    with open(prev_path, 'w') as pf:
                        json.dump(prev_event, pf, indent=2)
                    self.one_step_list.append(one_step_l2)
                except Exception as e:
                    print(f"[EvaluationManager] failed to compute one-step l2: {e}")

            # set current as pending for next round
            self.pending_prev_event = event_id
            self.pending_prev_pred_world = first_pred_world

            # update summary
            self._update_summary()

    def _update_summary(self):
        if len(self.metrics_list) == 0:
            summary_metrics = None
        else:
            l2s = [m["l2_mean"] for m in self.metrics_list]
            hauss = [m["hausdorff"] for m in self.metrics_list]
            fds = [m["fourier"] for m in self.metrics_list]
            summary_metrics = {
                "l2_mean": float(np.mean(l2s)),
                "hausdorff": float(np.mean(hauss)),
                "fourier": float(np.mean(fds))
            }
        one_step_mean = float(np.mean(self.one_step_list)) if len(self.one_step_list) > 0 else None
        summary = {
            "total": self.event_count,
            "valid": len(self.metrics_list),
            "metrics_mean": summary_metrics,
            "one_step_l2_mean": one_step_mean
        }
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    pass
