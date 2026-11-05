"""Privileged expert driver used to teach the camera policy (behavior cloning).

It sees the car pose, speed and progress (simulator information that the camera policy never gets) and drives
the track center line with pure pursuit steering and a speed controller that follows the braking envelope of
SpeedProfile. Its actions are the labels for the CNN policy.
"""
import numpy as np

from racecar_gym.speed_profile import SpeedProfile


class ExpertDriver:
    WHEELBASE = 0.32  # estimated from the measured minimum turning radius
    MAX_STEER_ANGLE = 0.42 * 0.5  # max_steering_angle * steering_multiplier

    # steer_sign = -1: a positive steering command turns the car clockwise (measured).
    # The other defaults were picked by a grid search: 8/8 passes through the two sharpest corners.
    def __init__(self, profile: SpeedProfile = None, steer_sign: float = -1.0,
                 lookahead_base: float = 0.7, lookahead_gain: float = 0.0,
                 speed_scale: float = 0.6, speed_gain: float = 3.0, v_cap: float = 3.3,
                 speed_lookahead: float = 3.0):
        self.profile = profile if profile is not None else SpeedProfile()
        self.center = self.profile.center_world
        self.n = len(self.center)
        self.seg = self.profile.seg_len
        self.steer_sign = steer_sign
        self.lookahead_base = lookahead_base
        self.lookahead_gain = lookahead_gain
        self.speed_scale = speed_scale
        self.speed_gain = speed_gain
        self.v_cap = v_cap
        self.speed_lookahead = speed_lookahead
        self.last_index = None

    def _nearest(self, pos, progress):
        guess = int(progress * self.n) % self.n
        idx = (guess + np.arange(-20, 21)) % self.n
        d = np.linalg.norm(self.center[idx] - pos, axis=1)
        return int(idx[int(d.argmin())])

    def _advance(self, index, distance):
        steps, travelled = 0, 0.0
        while travelled < distance and steps < self.n:
            travelled += self.seg[(index + steps) % self.n]
            steps += 1
        return (index + steps) % self.n

    def act(self, info) -> np.ndarray:
        pose = np.asarray(info['pose'])
        pos, yaw = pose[:2], float(pose[5])
        speed = float(np.linalg.norm(np.asarray(info['velocity'])[:2]))
        index = self._nearest(pos, float(info['progress']))
        self.last_index = index

        # Pure pursuit steering toward a speed dependent look ahead point.
        ld = self.lookahead_base + self.lookahead_gain * speed
        target = self.center[self._advance(index, ld)]
        delta = target - pos
        alpha = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        steer_angle = np.arctan2(2.0 * self.WHEELBASE * np.sin(alpha), max(np.linalg.norm(delta), 1e-3))
        steer = float(np.clip(self.steer_sign * steer_angle / self.MAX_STEER_ANGLE, -1.0, 1.0))

        # Speed: the lowest allowed speed over the next few meters (the profile already includes braking).
        ahead = self._advance(index, self.speed_lookahead)
        span = np.arange(index, index + 1 + ((ahead - index) % self.n)) % self.n
        v_ref = min(self.v_cap, float(self.profile.v_allowed[span].min()) * self.speed_scale)
        motor = float(np.clip(self.speed_gain * (v_ref - speed), -1.0, 1.0))
        return np.array([motor, steer], dtype=np.float32)
