import math
import time

import numpy as np


# ------------------------------------------------------------------
# Choose Option：'pure_pursuit' | 'lateral_pid'
LATERAL_CONTROLLER = ''lateral_pid'
# ------------------------------------------------------------------


class LongitudinalPID:
    def __init__(self, kp=1.0, ki=0.01, kd=0.0, dt=0.05, min_output=-3.0, max_output=3.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.min_output = min_output
        self.max_output = max_output

        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def step(self, target_speed, current_speed):
        # both speeds in m/s
        error = target_speed - current_speed
        self.integral += error * self.dt
        derivative = 0.0 if (self.prev_error is None) else (error - self.prev_error) / self.dt
        self.prev_error = error

        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        control = max(self.min_output, min(self.max_output, control))
        return control


class PurePursuit:
    def __init__(self, wheel_base=2.7, lookahead_gain=1.0, min_lookahead=2.0, max_steer_angle=0.7854):
        """
        wheel_base: vehicle wheel base in meters
        lookahead_gain: factor to compute lookahead distance from speed
        min_lookahead: minimum lookahead distance
        max_steer_angle: max steering (rad), default ~45deg
        """
        self.wheel_base = wheel_base
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_steer = max_steer_angle

    def get_steer(self, traj_xy, current_speed):
        """
        traj_xy: np array shape (T,2) in vehicle frame (x forward, y left)
        current_speed: m/s
        returns steering in [-1,1] as normalized fraction of max_steer
        """
        if traj_xy is None or len(traj_xy) == 0:
            return 0.0

        # choose lookahead distance proportional to speed
        ld = max(self.min_lookahead, self.lookahead_gain * current_speed)

        # find first point with distance >= ld
        dists = np.hypot(traj_xy[:, 0], traj_xy[:, 1])
        idx = np.searchsorted(dists, ld)
        if idx >= len(traj_xy):
            idx = len(traj_xy) - 1

        tx, ty = float(traj_xy[idx, 0]), float(traj_xy[idx, 1])

        if tx == 0 and ty == 0:
            alpha = 0.0
        else:
            alpha = math.atan2(ty, tx)

        # Pure pursuit steering angle
        # delta = atan2(2 * L * sin(alpha) / ld)
        # guard divide by zero
        if ld == 0:
            delta = 0.0
        else:
            delta = math.atan2(2.0 * self.wheel_base * math.sin(alpha), ld)

        # clamp
        delta = max(-self.max_steer, min(self.max_steer, delta))

        # normalize to [-1,1]
        steer_norm = delta / self.max_steer
        steer_norm = max(-1.0, min(1.0, steer_norm))
        return steer_norm


class LateralPID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.05, max_steer_angle=0.7854, lookahead=2.0):
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.max_steer = max_steer_angle
        self.lookahead = lookahead

        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def _select_target(self, traj_xy, current_speed):
       
        if traj_xy is None or len(traj_xy) == 0:
            return None
       
        ld = self.lookahead
        dists = np.hypot(traj_xy[:, 0], traj_xy[:, 1])
        idx = np.searchsorted(dists, ld)
        if idx >= len(traj_xy):
            idx = len(traj_xy) - 1
        return float(traj_xy[idx, 0]), float(traj_xy[idx, 1])

    def get_steer(self, traj_xy, current_speed):
      
        if traj_xy is None or len(traj_xy) == 0:
            return 0.0

        tgt = self._select_target(traj_xy, current_speed)
        if tgt is None:
            return 0.0
        tx, ty = tgt
        # desired heading in vehicle frame
        desired_heading = math.atan2(ty, tx)
        # heading error (vehicle heading = 0 in vehicle frame)
        error = desired_heading
        # wrap error to [-pi, pi]
        if error > math.pi:
            error -= 2.0 * math.pi
        elif error < -math.pi:
            error += 2.0 * math.pi

        # PID on heading error
        self.integral += error * self.dt
        derivative = 0.0 if (self.prev_error is None) else (error - self.prev_error) / self.dt
        self.prev_error = error

        control = self.kp * error + self.ki * self.integral + self.kd * derivative

        # control is in radians approx (if gains tuned that way). Clamp
        control = max(-self.max_steer, min(self.max_steer, control))

        steer_norm = control / self.max_steer
        steer_norm = max(-1.0, min(1.0, steer_norm))
        return steer_norm


class VehicleController:
    def __init__(self, cfg=None):
        # default controllers; can be tuned via cfg
        # longitudinal PID gains (tunable)
        if cfg is not None:
            kp = getattr(cfg, 'lon_kp', 1.0)
            ki = getattr(cfg, 'lon_ki', 0.0)
            kd = getattr(cfg, 'lon_kd', 0.0)
            wheel_base = getattr(cfg, 'wheel_base', 2.7)
            lookahead_gain = getattr(cfg, 'lookahead_gain', 1.0)
            min_lookahead = getattr(cfg, 'min_lookahead', 2.0)
            max_steer = getattr(cfg, 'max_steer_angle', 0.7854)
            # longitudinal parameters
            self.max_accel = getattr(cfg, 'max_accel', 3.0)
            # trajectory-derived speed estimation params
            self.traj_dt = getattr(cfg, 'traj_time_step', 0.1)
            self.traj_speed_points = getattr(cfg, 'traj_speed_points', 5)
            # steering saturation handling
            self.steer_saturation_threshold = getattr(cfg, 'steer_saturation_threshold', 0.95)
            self.slow_speed_on_high_steer = getattr(cfg, 'slow_speed_on_high_steer', 0.8)
            # steer rate limit
            self.max_steer_rate = getattr(cfg, 'max_steer_rate', 0.2)  # per control step
            # lateral controller selection
            lateral_ctrl_name = getattr(cfg, 'lateral_controller', None) or LATERAL_CONTROLLER
        else:
            kp, ki, kd = 1.0, 0.0, 0.0
            wheel_base = 2.7
            lookahead_gain = 1.0
            min_lookahead = 2.0
            max_steer = 0.7854
            self.max_accel = 3.0
            self.traj_dt = 0.1
            self.traj_speed_points = 5
            self.steer_saturation_threshold = 0.95
            self.slow_speed_on_high_steer = 0.8
            self.max_steer_rate = 0.2
            lateral_ctrl_name = LATERAL_CONTROLLER

        self.long_pid = LongitudinalPID(kp=kp, ki=ki, kd=kd, dt=0.05, min_output=-self.max_accel, max_output=self.max_accel)

        # instantiate lateral controller based on selection
        if lateral_ctrl_name == 'lateral_pid':
            # lateral pid gains can be provided via cfg: lat_kp, lat_ki, lat_kd, lat_dt, lat_lookahead
            lat_kp = getattr(cfg, 'lat_kp', 1.0)
            lat_ki = getattr(cfg, 'lat_ki', 0.0)
            lat_kd = getattr(cfg, 'lat_kd', 0.0)
            lat_dt = getattr(cfg, 'lat_dt', 0.05)
            lat_lookahead = getattr(cfg, 'lat_lookahead', 2.0)
            lat_max_steer = getattr(cfg, 'max_steer_angle', max_steer)
            self.lat_ctl = LateralPID(kp=lat_kp, ki=lat_ki, kd=lat_kd, dt=lat_dt, max_steer_angle=lat_max_steer, lookahead=lat_lookahead)
        else:
            # default to pure pursuit
            self.lat_ctl = PurePursuit(wheel_base=wheel_base, lookahead_gain=lookahead_gain,
                                       min_lookahead=min_lookahead, max_steer_angle=max_steer)

        self.max_accel = getattr(self, 'max_accel', 3.0)  # m/s^2

        # state for smoothing
        self.prev_steer = 0.0
        # expose last computed values for external logging / debugging
        self.last_target_speed = 0.0
        self.last_acc_cmd = 0.0

        # reverse detection/hysteresis state
        # number of future points to consider when deciding reverse
        import os
        # default values
        default_k = 5
        default_threshold = -0.05
        default_hysteresis = 2
        # allow overrides from env vars for quick comparison runs
        try:
            self._reverse_k = int(os.environ.get('REV_K', default_k))
        except Exception:
            self._reverse_k = default_k
        try:
            self._reverse_threshold = float(os.environ.get('REV_THRESHOLD', default_threshold))
        except Exception:
            self._reverse_threshold = default_threshold
        try:
            self._reverse_hysteresis_frames = int(os.environ.get('REV_HYSTERESIS', default_hysteresis))
        except Exception:
            self._reverse_hysteresis_frames = default_hysteresis
        # internal counter; positive -> leaning to reverse, negative -> leaning to forward
        self._reverse_counter = 0
        # cap for counter to avoid overflow
        self._reverse_counter_max = 5
        # current latched reverse state
        self._reverse_state = False

    def reset(self):
        self.long_pid.reset()
        self.prev_steer = 0.0
        # reset lateral controller state if supported
        try:
            self.lat_ctl.reset()
        except Exception:
            pass

    def control(self, traj, current_speed, desired_speed=None):
        """
        traj: predicted trajectory in vehicle frame (numpy array or torch tensor) shape (T,2) or (1,T,2)
        current_speed: current vehicle speed in m/s
        desired_speed: optional desired speed in m/s; if None, derive from traj points
        returns: throttle [0,1], brake [0,1], steer [-1,1], reverse(bool)
        """
        # normalize traj to numpy (T,2)
        if traj is None:
            return 0.0, 0.0, 0.0, False

        if hasattr(traj, 'detach'):
            traj = traj.detach().cpu().numpy()

        if traj.ndim == 3:
            traj = traj[0]

        # if trajectory is empty
        if traj.shape[0] == 0:
            return 0.0, 0.0, 0.0, False

        # compute desired speed heuristically if not provided: use multiple future points
        if desired_speed is None:
            k = min(self.traj_speed_points, traj.shape[0])
            if k <= 1:
                # fallback: distance to first point mapping
                first = traj[0]
                dist = math.hypot(float(first[0]), float(first[1]))
                target_speed = max(0.5, min(6.0, dist))
            else:
                # compute speeds between successive points
                deltas = traj[:k, :].copy()
                # compute pairwise distances over the k points
                dists = np.hypot(np.diff(deltas[:, 0]), np.diff(deltas[:, 1]))
                # if only one diff (k==2) handle
                if dists.size == 0:
                    speeds = np.array([0.0])
                else:
                    speeds = dists / max(1e-6, self.traj_dt)
                # robust estimate: take median or mean of positive speeds
                if speeds.size == 0:
                    target_speed = 0.5
                else:
                    # prevent spurious huge speeds
                    speeds = np.clip(speeds, 0.0, 10.0)
                    target_speed = float(max(0.5, np.mean(speeds)))
                target_speed = max(0.5, min(8.0, target_speed))
        else:
            target_speed = float(desired_speed)

        # lateral control
        steer = self.lat_ctl.get_steer(traj, current_speed)

        # if high steering demand, reduce longitudinal target speed for safety
        if abs(steer) >= self.steer_saturation_threshold:
            target_speed = min(target_speed, self.slow_speed_on_high_steer)

        # smooth steer rate (limit abrupt steer commands)
        steer_delta = steer - self.prev_steer
        if abs(steer_delta) > self.max_steer_rate:
            steer = self.prev_steer + math.copysign(self.max_steer_rate, steer_delta)
        # clamp steer in [-1,1]
        steer = max(-1.0, min(1.0, steer))

        # decide reverse based on majority of first few points being behind ego
        k_rev = min(self._reverse_k, traj.shape[0])
        # use median to be robust to outliers
        median_x = float(np.median(traj[:k_rev, 0]))
        # update hysteresis counter
        if median_x < self._reverse_threshold:
            self._reverse_counter = min(self._reverse_counter + 1, self._reverse_counter_max)
        else:
            self._reverse_counter = max(self._reverse_counter - 1, -self._reverse_counter_max)

        # determine reverse state based on counter and hysteresis
        if self._reverse_counter >= self._reverse_hysteresis_frames:
            reverse = True
        elif self._reverse_counter <= -self._reverse_hysteresis_frames:
            reverse = False
        else:
            # keep previous latched state if undecided
            reverse = bool(self._reverse_state)

        # latch state when counter crosses threshold
        if self._reverse_counter >= self._reverse_hysteresis_frames:
            self._reverse_state = True
        elif self._reverse_counter <= -self._reverse_hysteresis_frames:
            self._reverse_state = False

        # longitudinal PID compute acceleration command (m/s^2)
        acc_cmd = self.long_pid.step(target_speed, current_speed)
        # map to throttle/brake
        if acc_cmd >= 0:
            throttle = min(1.0, acc_cmd / self.max_accel)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -acc_cmd / self.max_accel)

        # update prev steer
        self.prev_steer = steer

        # save diagnostics
        self.last_target_speed = target_speed
        self.last_acc_cmd = acc_cmd

        return float(throttle), float(brake), float(steer), bool(reverse)
